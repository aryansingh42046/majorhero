import csv
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import hashlib

root = Path.cwd()
csv_path = root / "datasets/manifests/cross_dataset_version_comparison.csv"
out_path = root / "datasets/manifests/cross_dataset_review_next20.jpg"

with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

def full_path(value):
    return (root / value).resolve()

# Keep distinct path pairs; sort using the recomputed distance.
seen = set()
candidates = []
for row in rows:
    mp = full_path(row["mdd_path"])
    op = full_path(row["odir_path"])
    key = (str(mp).lower(), str(op).lower())
    if key in seen or not mp.is_file() or not op.is_file():
        continue
    seen.add(key)
    try:
        distance = int(float(row["recomputed_distance"]))
    except (ValueError, TypeError):
        continue
    candidates.append((distance, row, mp, op))

candidates.sort(key=lambda x: x[0])
candidates = candidates[:20]

thumb_w, thumb_h = 250, 220
label_h = 72
row_h = thumb_h + label_h
sheet = Image.new("RGB", (thumb_w * 2, max(1, len(candidates)) * row_h), "white")
draw = ImageDraw.Draw(sheet)

for i, (dist, row, mp, op) in enumerate(candidates):
    y = i * row_h
    for col, path in enumerate((mp, op)):
        try:
            im = Image.open(path).convert("RGB")
            im = ImageOps.contain(im, (thumb_w - 10, thumb_h - 10))
            x = col * thumb_w + (thumb_w - im.width) // 2
            yy = y + (thumb_h - im.height) // 2
            sheet.paste(im, (x, yy))
        except Exception:
            draw.text((col * thumb_w + 10, y + 10), "Could not load image", fill="red")

    draw.text((5, y + thumb_h + 3),
              f"#{i+1} d={dist} | MDD: {row['mdd_file']} | label={row['mdd_label']}",
              fill="black")
    draw.text((5, y + thumb_h + 23),
              f"ODIR: {row['odir_file']} | label={row['odir_label']} | {row['odir_version']}",
              fill="black")
    draw.text((5, y + thumb_h + 43),
              f"Pair: {mp.name}  <>  {op.name}",
              fill="black")

sheet.save(out_path, quality=92)
print(f"Created: {out_path}")
print(f"Unique path pairs shown: {len(candidates)}")
