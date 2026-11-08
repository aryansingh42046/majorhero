
import pandas as pd
import hashlib
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

CSV = ROOT / "datasets/manifests/cross_dataset_version_comparison.csv"
OUTPUT = ROOT / "datasets/manifests/cross_dataset_selected_hash_check.csv"

TARGETS = [
    "2170_left.jpg",
    "1062_right.jpg",
    "158_right.jpg",
    "162_right.jpg",
    "2165_right.jpg",
]

def get_hash(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def main():
    df = pd.read_csv(CSV)

    selected = df[
        df["mdd_file"].isin(TARGETS)
    ].copy()

    results = []

    for _, row in selected.iterrows():

        mdd_path = ROOT / row["mdd_path"]
        odir_path = ROOT / row["odir_path"]

        if not mdd_path.exists() or not odir_path.exists():
            print("Missing file:", mdd_path, odir_path)
            continue

        mdd_hash = get_hash(mdd_path)
        odir_hash = get_hash(odir_path)

        with Image.open(mdd_path) as im:
            mdd_size = im.size

        with Image.open(odir_path) as im:
            odir_size = im.size

        results.append({
            "mdd_file": row["mdd_file"],
            "odir_file": row["odir_file"],
            "odir_version": row["odir_version"],
            "mdd_label": row["mdd_label"],
            "odir_label": row["odir_label"],
            "recomputed_distance": row["recomputed_distance"],
            "mdd_sha256": mdd_hash,
            "odir_sha256": odir_hash,
            "sha256_identical": mdd_hash == odir_hash,
            "mdd_dimensions": str(mdd_size),
            "odir_dimensions": str(odir_size),
            "mdd_path": str(mdd_path),
            "odir_path": str(odir_path),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT, index=False)

    print("\nHASH VERIFICATION RESULTS")
    print(result_df[
        [
            "mdd_file",
            "odir_file",
            "odir_version",
            "sha256_identical",
            "recomputed_distance"
        ]
    ].to_string(index=False))

    print("\nTotal comparisons:", len(result_df))
    print("Exact hash matches:", result_df["sha256_identical"].sum())
    print("Different hashes:", (~result_df["sha256_identical"]).sum())
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
