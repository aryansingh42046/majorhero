
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[2]

CSV = ROOT / "datasets/manifests/duplicate_resolution_decisions.csv"
OUTPUT = ROOT / "datasets/manifests/duplicate_review_contact_sheet.jpg"

df = pd.read_csv(CSV)
groups = list(df.groupby(["dataset", "group_id"]))

W, H = 240, 240
CELL_W = W * 2
CELL_H = H + 65
COLUMNS = 2

total_rows = math.ceil(len(groups) / COLUMNS)

sheet = Image.new(
    "RGB",
    (CELL_W * COLUMNS, CELL_H * total_rows),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, ((dataset, group_id), group) in enumerate(groups):

    row = index // COLUMNS
    col = index % COLUMNS

    x_start = col * CELL_W
    y_start = row * CELL_H

    labels = ", ".join(group["label"].astype(str))

    draw.text(
        (x_start + 5, y_start + 5),
        f"{dataset} | Group {group_id} | Labels: {labels}",
        fill="black"
    )

    for image_index, (_, record) in enumerate(group.iterrows()):

        path = Path(record["image_path"])

        if not path.exists():
            print("MISSING:", path)
            continue

        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((W, H))

        x = x_start + image_index * W
        y = y_start + 25

        sheet.paste(image, (x, y))

        draw.text(
            (x + 5, y + H + 5),
            str(record["filename"]),
            fill="black"
        )

    print(f"Added {dataset} Group {group_id}")

sheet.save(OUTPUT, quality=95)

print("\nAll groups included:", len(groups))
print("Saved:", OUTPUT)
