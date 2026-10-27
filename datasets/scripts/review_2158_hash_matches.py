import csv
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

root = Path.cwd()
source = root / "datasets/manifests/cross_dataset_version_comparison.csv"
output = root / "datasets/manifests/2158_hash_collision_review.jpg"

with source.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

matches = []
seen = set()

for r in rows:
    try:
        d = int(float(r["recomputed_distance"]))
    except (ValueError, TypeError):
        continue

    if r["mdd_file"].lower() != "2158_right.jpg" or d > 1:
        continue

    mdd = (root / r["mdd_path"]).resolve()
    odir = (root / r["odir_path"]).resolve()
    key = (str(mdd).lower(), str(odir).lower())

    if key in seen or not mdd.is_file() or not odir.is_file():
        continue

    seen.add(key)
    matches.append((d, r, mdd, odir))

matches.sort(key=lambda x: x[0])

w, h, label_h = 500, 420, 100
sheet = Image.new("RGB", (w * 2, max(1, len(matches)) * (h + label_h)), "white")
draw = ImageDraw.Draw(sheet)

for i, (distance, r, mdd, odir) in enumerate(matches):
    y = i * (h + label_h)

    for col, path in enumerate((mdd, odir)):
        try:
            with Image.open(path) as im:
                im = ImageOps.contain(im.convert("RGB"), (w - 12, h - 12))
            x = col * w + (w - im.width) // 2
            yy = y + (h - im.height) // 2
            sheet.paste(im, (x, yy))
        except Exception as e:
            draw.text((col * w + 10, y + 10), f"Image error: {e}", fill="red")

    draw.text((5, y + h + 4),
              f"Pair {i+1} | dHash distance: {distance}", fill="black")
    draw.text((5, y + h + 24),
              f"MDD: {r['mdd_file']} | label: {r['mdd_label']}", fill="black")
    draw.text((5, y + h + 44),
              f"ODIR: {r['odir_file']} | label: {r['odir_label']}", fill="black")
    draw.text((5, y + h + 64),
              f"ODIR version: {r['odir_version']}", fill="black")

sheet.save(output, quality=95)
print(f"Created: {output}")
print(f"Unique pairs shown: {len(matches)}")
