#!/usr/bin/env python
"""Read-only inspection of the raw retinal datasets (Phase 2).

This script NEVER modifies, moves, or deletes anything under the raw dataset
folder. It only reads files and writes reports to the output directory.

It reports, for every dataset folder found under the raw root:
  * directory hierarchy with recursive image counts
  * image counts, file extensions, formats, colour modes and dimensions
  * corrupted / unreadable images
  * CSV / Excel / other annotation files, their columns, sample rows and
    class distributions of low-cardinality columns
  * candidate patient-ID / eye columns
  * which table columns reference image filenames, images that no table
    references, and table rows that point to missing images
  * filenames that are ambiguous (same stem in several folders)
  * exact duplicates (SHA-256) and near-duplicate candidates (perceptual hash),
    including duplicates across datasets
  * disk usage and free space

Outputs (default: datasets/manifests/inspection/):
  inspection_report.md      human-readable summary -> share this one
  image_inventory.csv       one row per image file
  exact_duplicates.csv      groups of byte-identical images
  near_duplicate_pairs.csv  perceptual-hash pairs within the threshold
  tables_summary.json       detailed per-table information

Example (PowerShell, from the repository root):
  python scripts\\inspect_datasets.py --sample 30 --workers 2   # quick test
  python scripts\\inspect_datasets.py --workers 4                # full run
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

try:
    import imagehash
except ImportError:  # perceptual hashing is optional
    imagehash = None

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    tqdm = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif",
              ".ppm", ".pgm", ".webp", ".jp2"}
TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}
ANNOTATION_EXTS = {".json", ".xml", ".txt", ".mat", ".npy", ".npz"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz"}
ID_KEYWORDS = ("patient", "subject", "id", "eye", "laterality", "left",
               "right", "side", "od", "os")
MAX_VALUE_COUNTS_UNIQUE = 25

log = logging.getLogger("inspect_datasets")

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# --------------------------------------------------------------------------
# Image inspection (runs in worker processes; must stay top-level for Windows)
# --------------------------------------------------------------------------
def inspect_image(task: tuple[str, bool]) -> dict[str, Any]:
    """Inspect one image file without modifying it."""
    path_str, do_hash = task
    rec: dict[str, Any] = {
        "abs_path": path_str, "size_bytes": None, "format": "", "mode": "",
        "width": None, "height": None, "readable": False, "error": "",
        "sha256": "", "phash": "",
    }
    try:
        path = Path(path_str)
        rec["size_bytes"] = path.stat().st_size
        if do_hash:
            data = path.read_bytes()
            rec["sha256"] = hashlib.sha256(data).hexdigest()
            source: Any = io.BytesIO(data)
        else:
            source = path
        with Image.open(source) as im:
            rec["format"] = im.format or ""
            rec["mode"] = im.mode
            rec["width"], rec["height"] = im.size
            if do_hash:
                im.load()  # full decode: catches truncated/corrupted files
                if imagehash is not None:
                    try:
                        rec["phash"] = str(imagehash.phash(im))
                    except Exception as exc:  # hashing failure != unreadable
                        rec["error"] = f"phash failed: {type(exc).__name__}: {exc}"
            else:
                im.verify()  # cheaper structural check
        rec["readable"] = True
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def progress(iterable: Iterable, total: int, desc: str) -> Iterable:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="img")


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POPCOUNT8[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def filename_pattern(name: str) -> str:
    return re.sub(r"\d+", "#", name)


def read_table(path: Path) -> dict[str, pd.DataFrame]:
    """Read a CSV/TSV/Excel file into {sheet_name: DataFrame}."""
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=None)
    sep = "\t" if ext == ".tsv" else None
    last_exc: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return {"(csv)": pd.read_csv(path, sep=sep, engine="python",
                                         encoding=enc)}
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def normalise_ref(value: Any) -> tuple[str, str] | None:
    """Return (lowercase basename, lowercase stem) for a table cell value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    base = text.split("/")[-1].lower()
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return base, stem


# --------------------------------------------------------------------------
# Per-dataset analysis
# --------------------------------------------------------------------------
def scan_files(ds_root: Path) -> tuple[list[Path], list[Path], list[Path],
                                         list[Path], Counter]:
    images, tables, annotations, archives = [], [], [], []
    other_exts: Counter = Counter()
    for dirpath, _dirnames, filenames in os.walk(ds_root):
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                images.append(p)
            elif ext in TABLE_EXTS:
                tables.append(p)
            elif ext in ANNOTATION_EXTS:
                annotations.append(p)
            elif ext in ARCHIVE_EXTS:
                archives.append(p)
            else:
                other_exts[ext or "(no extension)"] += 1
    images.sort()
    tables.sort()
    return images, tables, annotations, archives, other_exts


def directory_tree(ds_root: Path, max_depth: int, max_lines: int) -> list[str]:
    """Indented tree with recursive image counts and direct file counts."""
    direct_imgs: Counter = Counter()
    direct_files: Counter = Counter()
    for dirpath, _d, filenames in os.walk(ds_root):
        rel = Path(dirpath).relative_to(ds_root)
        direct_files[rel] = len(filenames)
        direct_imgs[rel] = sum(Path(f).suffix.lower() in IMAGE_EXTS
                               for f in filenames)
    recursive: Counter = Counter()
    for rel, n in direct_imgs.items():
        parts = rel.parts
        for i in range(len(parts) + 1):
            recursive[Path(*parts[:i]) if i else Path(".")] += n
    lines = []
    for rel in sorted(direct_files, key=lambda r: [p.lower() for p in r.parts]):
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        name = ds_root.name if depth == 0 else rel.parts[-1]
        lines.append(f"{'  ' * depth}{name}/  [images (recursive): "
                     f"{recursive[rel if depth else Path('.')]}, files here: "
                     f"{direct_files[rel]}]")
        if len(lines) >= max_lines:
            lines.append("  ... (tree truncated; raise --tree-lines to see more)")
            break
    return lines


def summarise_table(df: pd.DataFrame, image_names: set[str],
                    image_stems: set[str], max_examples: int) -> dict[str, Any]:
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    info: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "head": df.head(5).to_string(max_colwidth=40),
        "missing_per_column": {str(c): int(v) for c, v in df.isna().sum().items() if v},
        "value_counts": {},
        "high_cardinality_columns": {},
        "id_like_columns": {},
        "filename_columns": {},
    }
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        if n_unique <= MAX_VALUE_COUNTS_UNIQUE:
            vc = series.value_counts(dropna=False)
            info["value_counts"][str(col)] = {str(k): int(v) for k, v in vc.items()}
        else:
            info["high_cardinality_columns"][str(col)] = n_unique
        lname = str(col).lower()
        tokens = set(re.split(r"[^a-z]+", lname))
        if any(k in tokens or (len(k) > 3 and k in lname) for k in ID_KEYWORDS):
            info["id_like_columns"][str(col)] = {
                "unique": n_unique,
                "rows": int(series.notna().sum()),
                "repeats": bool(n_unique < series.notna().sum()),
            }
        # Does this column reference image files?
        refs = [normalise_ref(v) for v in series.tolist()]
        refs = [r for r in refs if r is not None]
        if not refs:
            continue
        matched = [r for r in refs if r[0] in image_names or r[1] in image_stems]
        if matched and len(matched) / len(refs) >= 0.05:
            missing = [r[0] for r in refs
                       if not (r[0] in image_names or r[1] in image_stems)]
            info["filename_columns"][str(col)] = {
                "values": len(refs),
                "matched_to_images": len(matched),
                "unmatched": len(missing),
                "unmatched_examples": missing[:max_examples],
                "matched_stems": sorted({r[1] for r in matched}),
            }
    return info


def analyse_dataset(name: str, ds_root: Path, raw_root: Path,
                    args: argparse.Namespace) -> tuple[dict, list[dict]]:
    log.info("Scanning %s ...", ds_root)
    images, tables, annotations, archives, other_exts = scan_files(ds_root)
    all_images = list(images)
    total_found = len(images)
    if args.sample and len(images) > args.sample:
        rng = random.Random(args.seed)
        images = sorted(rng.sample(images, args.sample))

    do_hash = not args.no_hash
    tasks = [(str(p), do_hash) for p in images]
    records: list[dict] = []
    if args.workers <= 1:
        for t in progress(tasks, len(tasks), name):
            records.append(inspect_image(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rec in progress(pool.map(inspect_image, tasks, chunksize=16),
                                len(tasks), name):
                records.append(rec)

    for rec in records:
        p = Path(rec["abs_path"])
        rel = p.relative_to(ds_root)
        rec["dataset"] = name
        rec["rel_path"] = str(p.relative_to(raw_root))
        rec["subfolder"] = "/".join(rel.parts[:-1][:2]) or "."
        rec["filename"] = p.name
        rec["stem"] = p.stem
        rec["ext"] = p.suffix.lower()

    # Label matching always uses ALL images found, not just the sample.
    all_names = {p.name.lower() for p in all_images}
    stem_to_paths: dict[str, list[str]] = defaultdict(list)
    for p in all_images:
        stem_to_paths[p.stem.lower()].append(str(p.relative_to(ds_root)))
    all_stems = set(stem_to_paths)

    table_info: dict[str, Any] = {}
    referenced_stems: set[str] = set()
    for tpath in tables:
        key = str(tpath.relative_to(ds_root))
        try:
            sheets = read_table(tpath)
        except Exception as exc:
            table_info[key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        for sheet, df in sheets.items():
            tkey = key if sheet == "(csv)" else f"{key} [sheet: {sheet}]"
            try:
                s = summarise_table(df, all_names, all_stems, args.max_examples)
            except Exception as exc:
                table_info[tkey] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            for col in s["filename_columns"].values():
                referenced_stems.update(col.pop("matched_stems"))
            table_info[tkey] = s

    unreferenced = sorted(all_stems - referenced_stems)
    ambiguous = {s: paths for s, paths in stem_to_paths.items() if len(paths) > 1}

    readable = [r for r in records if r["readable"]]
    unreadable = [r for r in records if not r["readable"]]
    dims = Counter(f"{r['width']}x{r['height']}" for r in readable)
    by_sub: dict[str, Counter] = defaultdict(Counter)
    sub_counts: Counter = Counter()
    for r in readable:
        sub_counts[r["subfolder"]] += 1
        by_sub[r["subfolder"]][f"{r['width']}x{r['height']} {r['mode']}"] += 1

    summary = {
        "name": name,
        "root": str(ds_root),
        "images_found": total_found,
        "images_inspected": len(records),
        "sampled": bool(args.sample and total_found > len(records)),
        "readable": len(readable),
        "unreadable": len(unreadable),
        "unreadable_examples": [(r["rel_path"], r["error"]) for r in unreadable][:args.max_examples],
        "hash_warnings": [(r["rel_path"], r["error"]) for r in readable if r["error"]][:args.max_examples],
        "total_image_bytes": int(sum(r["size_bytes"] or 0 for r in records)),
        "extensions": dict(Counter(r["ext"] for r in records).most_common()),
        "formats": dict(Counter(r["format"] for r in readable).most_common()),
        "modes": dict(Counter(r["mode"] for r in readable).most_common()),
        "top_dimensions": dict(dims.most_common(10)),
        "n_distinct_dimensions": len(dims),
        "width_range": [min((r["width"] for r in readable), default=None),
                        max((r["width"] for r in readable), default=None)],
        "height_range": [min((r["height"] for r in readable), default=None),
                         max((r["height"] for r in readable), default=None)],
        "subfolders": {k: {"images": sub_counts[k],
                           "top_dims_modes": dict(by_sub[k].most_common(3))}
                       for k in sorted(sub_counts)},
        "filename_patterns": dict(Counter(filename_pattern(p.name) for p in images).most_common(15)),
        "laterality_in_filenames": {
            "left": sum(bool(re.search(r"(^|[^a-z])(left|os)([^a-z]|$)", p.stem.lower())) for p in images),
            "right": sum(bool(re.search(r"(^|[^a-z])(right|od)([^a-z]|$)", p.stem.lower())) for p in images),
        },
        "tables": table_info,
        "other_annotation_files": [str(p.relative_to(ds_root)) for p in annotations][:50],
        "n_other_annotation_files": len(annotations),
        "archives": [str(p.relative_to(ds_root)) for p in archives],
        "other_file_extensions": dict(other_exts.most_common()),
        "images_not_referenced_by_any_table": len(unreferenced),
        "unreferenced_examples": unreferenced[:args.max_examples],
        "ambiguous_stems": len(ambiguous),
        "ambiguous_examples": dict(list(ambiguous.items())[:args.max_examples]),
        "tree": directory_tree(ds_root, args.tree_depth, args.tree_lines),
    }
    return summary, records


# --------------------------------------------------------------------------
# Cross-dataset duplicate detection
# --------------------------------------------------------------------------
def find_exact_duplicates(records: list[dict]) -> pd.DataFrame:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["sha256"]:
            groups[r["sha256"]].append(r)
    rows = []
    gid = 0
    for sha, members in groups.items():
        if len(members) < 2:
            continue
        gid += 1
        for m in members:
            rows.append({"group_id": gid, "sha256": sha, "dataset": m["dataset"],
                         "rel_path": m["rel_path"]})
    return pd.DataFrame(rows, columns=["group_id", "sha256", "dataset", "rel_path"])


def find_near_duplicates(records: list[dict], threshold: int,
                         max_pairs: int) -> tuple[pd.DataFrame, int]:
    usable = [r for r in records if r["phash"]]
    if len(usable) < 2:
        return pd.DataFrame(columns=["distance", "dataset_a", "path_a",
                                     "dataset_b", "path_b"]), 0
    hashes = np.array([int(r["phash"], 16) for r in usable], dtype=np.uint64)
    shas = [r["sha256"] for r in usable]
    rows, total = [], 0
    iterator = range(len(usable) - 1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="near-duplicate search", unit="img")
    for i in iterator:
        dist = popcount64(hashes[i + 1:] ^ hashes[i])
        for j_off in np.nonzero(dist <= threshold)[0]:
            j = i + 1 + int(j_off)
            if shas[i] and shas[i] == shas[j]:
                continue  # exact duplicate, reported separately
            total += 1
            if len(rows) < max_pairs:
                rows.append({"distance": int(dist[j_off]),
                             "dataset_a": usable[i]["dataset"], "path_a": usable[i]["rel_path"],
                             "dataset_b": usable[j]["dataset"], "path_b": usable[j]["rel_path"]})
    df = pd.DataFrame(rows, columns=["distance", "dataset_a", "path_a",
                                     "dataset_b", "path_b"])
    return df.sort_values("distance", kind="stable"), total


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def build_report(summaries: list[dict], exact: pd.DataFrame, near: pd.DataFrame,
                 near_total: int, raw_root: Path, root_files: list[str],
                 args: argparse.Namespace, elapsed: float) -> str:
    L: list[str] = []
    L.append("# Dataset inspection report")
    L.append("")
    L.append(f"- Raw root: `{raw_root}`")
    L.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} (took {elapsed:.0f}s)")
    L.append(f"- Mode: {'SAMPLE of ' + str(args.sample) + ' images per dataset' if args.sample else 'FULL'}; "
             f"hashing {'off' if args.no_hash else 'on'}; "
             f"pHash available: {imagehash is not None}; near-dup threshold: {args.phash_threshold}")
    try:
        du = shutil.disk_usage(raw_root)
        L.append(f"- Disk free on this drive: {human_bytes(du.free)} of {human_bytes(du.total)}")
    except OSError:
        pass
    if root_files:
        L.append(f"- Files directly inside raw root (not in a dataset folder): {root_files}")
    L.append("")
    L.append("## Overview")
    L.append("")
    L.append("| Dataset | Images found | Inspected | Unreadable | Size | Tables | Not referenced by tables | Ambiguous stems |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        L.append(f"| {s['name']} | {s['images_found']} | {s['images_inspected']} | {s['unreadable']} | "
                 f"{human_bytes(s['total_image_bytes'])} | {len(s['tables'])} | "
                 f"{s['images_not_referenced_by_any_table']} | {s['ambiguous_stems']} |")
    L.append("")

    for s in summaries:
        L.append(f"## {s['name']}")
        L.append("")
        L.append("### Directory tree")
        L.append("```text")
        L.extend(s["tree"])
        L.append("```")
        L.append("")
        L.append("### Images")
        L.append(f"- Extensions: {s['extensions']}")
        L.append(f"- Formats: {s['formats']}")
        L.append(f"- Colour modes: {s['modes']}")
        L.append(f"- Distinct dimensions: {s['n_distinct_dimensions']}; width range {s['width_range']}, "
                 f"height range {s['height_range']}")
        L.append(f"- Most common dimensions: {s['top_dimensions']}")
        L.append(f"- Filename patterns (digits shown as #): {s['filename_patterns']}")
        L.append(f"- Filenames containing left/OS: {s['laterality_in_filenames']['left']}, "
                 f"right/OD: {s['laterality_in_filenames']['right']}")
        if s["archives"]:
            L.append(f"- ARCHIVES still present (contents NOT inspected): {s['archives']}")
        if s["other_file_extensions"]:
            L.append(f"- Other (non-image, non-table) file types: {s['other_file_extensions']}")
        if s["n_other_annotation_files"]:
            L.append(f"- Other annotation-like files ({s['n_other_annotation_files']}): "
                     f"{s['other_annotation_files'][:args.max_examples]}")
        L.append("")
        L.append("Per subfolder (first two levels): images, most common `WxH mode`")
        L.append("")
        for sub, v in s["subfolders"].items():
            L.append(f"- `{sub}`: {v['images']} | {v['top_dims_modes']}")
        if s["unreadable"]:
            L.append("")
            L.append(f"**Unreadable images ({s['unreadable']}):**")
            for path, err in s["unreadable_examples"]:
                L.append(f"- `{path}` -> {err}")
        if s["hash_warnings"]:
            L.append("")
            L.append("Hashing warnings:")
            for path, err in s["hash_warnings"]:
                L.append(f"- `{path}` -> {err}")
        L.append("")
        L.append("### Tables and labels")
        if not s["tables"]:
            L.append("- No CSV/Excel files found. Labels may come from folder names or filenames.")
        for tname, t in s["tables"].items():
            L.append("")
            L.append(f"#### `{tname}`")
            if "error" in t:
                L.append(f"- Could not read: {t['error']}")
                continue
            L.append(f"- Rows: {t['rows']}; columns ({len(t['columns'])}): {t['columns']}")
            if t["missing_per_column"]:
                L.append(f"- Missing values per column: {t['missing_per_column']}")
            if t["id_like_columns"]:
                L.append(f"- Possible patient/eye ID columns: {t['id_like_columns']}")
            if t["filename_columns"]:
                for col, fc in t["filename_columns"].items():
                    L.append(f"- Column `{col}` references images: {fc['matched_to_images']} of "
                             f"{fc['values']} values match an image file; {fc['unmatched']} do not"
                             + (f" (e.g. {fc['unmatched_examples']})" if fc["unmatched"] else ""))
            else:
                L.append("- No column clearly references image filenames.")
            if t["high_cardinality_columns"]:
                L.append(f"- High-cardinality columns (unique count): {t['high_cardinality_columns']}")
            for col, vc in t["value_counts"].items():
                L.append(f"- Distribution of `{col}`: {vc}")
            L.append("")
            L.append("First rows:")
            L.append("```text")
            L.append(t["head"])
            L.append("```")
        L.append("")
        L.append(f"- Images not referenced by any table: {s['images_not_referenced_by_any_table']}"
                 + (f" (e.g. {s['unreferenced_examples']})" if s["unreferenced_examples"] else ""))
        if s["ambiguous_stems"]:
            L.append(f"- Ambiguous filename stems (same name in several folders): {s['ambiguous_stems']}")
            for stem, paths in s["ambiguous_examples"].items():
                L.append(f"  - `{stem}`: {paths}")
        L.append("")

    L.append("## Duplicates")
    L.append("")
    if args.no_hash:
        L.append("- Skipped (--no-hash).")
    else:
        n_groups = exact["group_id"].nunique() if len(exact) else 0
        L.append(f"- Exact duplicate groups (identical bytes): {n_groups}, covering {len(exact)} files")
        if n_groups:
            cross = exact.groupby("group_id")["dataset"].nunique()
            L.append(f"- Exact groups spanning more than one dataset: {int((cross > 1).sum())}")
            for gid in list(exact["group_id"].unique())[:args.max_examples]:
                L.append(f"  - group {gid}: {exact.loc[exact.group_id == gid, 'rel_path'].tolist()}")
        L.append(f"- Near-duplicate candidate pairs (pHash distance <= {args.phash_threshold}, "
                 f"excluding exact duplicates): {near_total}")
        if len(near):
            combos = Counter(tuple(sorted((a, b))) for a, b in zip(near.dataset_a, near.dataset_b))
            L.append(f"- Pairs by dataset combination: {dict(combos)}")
            L.append(f"- Distance distribution: {near['distance'].value_counts().sort_index().to_dict()}")
            for _, row in near.head(args.max_examples).iterrows():
                L.append(f"  - d={row.distance}: `{row.path_a}` <-> `{row.path_b}`")
        if args.sample:
            L.append("- NOTE: sample mode only compares the sampled images.")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, default=repo_root / "datasets" / "raw",
                    help="Folder containing one subfolder per dataset.")
    ap.add_argument("--out-dir", type=Path,
                    default=repo_root / "datasets" / "manifests" / "inspection",
                    help="Where reports are written (never inside --raw-root).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Only inspect these dataset folders (case-insensitive).")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="Parallel worker processes (1 = no multiprocessing).")
    ap.add_argument("--sample", type=int, default=0,
                    help="Inspect only N random images per dataset (0 = all).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-hash", action="store_true",
                    help="Skip SHA-256/pHash (faster, but no duplicate detection).")
    ap.add_argument("--phash-threshold", type=int, default=6,
                    help="Max Hamming distance (0-64) for near-duplicate candidates.")
    ap.add_argument("--max-near-pairs", type=int, default=50000,
                    help="Cap on near-duplicate pairs written to CSV.")
    ap.add_argument("--tree-depth", type=int, default=4)
    ap.add_argument("--tree-lines", type=int, default=150)
    ap.add_argument("--max-examples", type=int, default=10)
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    raw_root: Path = args.raw_root.resolve()
    out_dir: Path = args.out_dir.resolve()

    if not raw_root.is_dir():
        log.error("Raw root not found: %s", raw_root)
        return 1
    if out_dir == raw_root or raw_root in out_dir.parents:
        log.error("--out-dir must not be inside the raw dataset folder.")
        return 1
    if args.no_hash is False and imagehash is None:
        log.warning("ImageHash not installed: near-duplicate detection disabled.")

    ds_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()], key=lambda d: d.name.lower())
    root_files = sorted(p.name for p in raw_root.iterdir() if p.is_file())
    if args.datasets:
        wanted = {d.lower() for d in args.datasets}
        ds_dirs = [d for d in ds_dirs if d.name.lower() in wanted]
        missing = wanted - {d.name.lower() for d in ds_dirs}
        if missing:
            log.warning("Requested dataset folders not found: %s", sorted(missing))
    if not ds_dirs:
        log.error("No dataset folders found under %s", raw_root)
        return 1
    log.info("Datasets to inspect: %s", [d.name for d in ds_dirs])

    start = time.time()
    summaries, all_records = [], []
    for d in ds_dirs:
        summary, records = analyse_dataset(d.name, d, raw_root, args)
        summaries.append(summary)
        all_records.extend(records)

    exact = find_exact_duplicates(all_records) if not args.no_hash else pd.DataFrame()
    near, near_total = (find_near_duplicates(all_records, args.phash_threshold, args.max_near_pairs)
                        if not args.no_hash else (pd.DataFrame(), 0))

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "rel_path", "subfolder", "filename", "ext", "size_bytes", "format", "mode",
            "width", "height", "readable", "error", "sha256", "phash"]
    pd.DataFrame(all_records)[cols].to_csv(out_dir / "image_inventory.csv", index=False)
    if not args.no_hash:
        exact.to_csv(out_dir / "exact_duplicates.csv", index=False)
        near.to_csv(out_dir / "near_duplicate_pairs.csv", index=False)
    with open(out_dir / "tables_summary.json", "w", encoding="utf-8") as fh:
        json.dump({s["name"]: s for s in summaries}, fh, indent=2, default=str)
    report = build_report(summaries, exact, near, near_total, raw_root, root_files,
                          args, time.time() - start)
    (out_dir / "inspection_report.md").write_text(report, encoding="utf-8")

    print()
    for s in summaries:
        print(f"{s['name']:<12} found={s['images_found']:<7} inspected={s['images_inspected']:<7} "
              f"unreadable={s['unreadable']:<5} tables={len(s['tables'])}")
    print(f"\nReport written to: {out_dir / 'inspection_report.md'}")
    print("Nothing under the raw dataset folder was modified.")
    return 0


if __name__ == "__main__":  # required for multiprocessing on Windows
    sys.exit(main())
