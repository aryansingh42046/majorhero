
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]

CSV = ROOT / "datasets/manifests/cross_dataset_version_comparison.csv"
OUTPUT = ROOT / "datasets/manifests/focused_cross_dataset_review.jpg"

TARGETS = [
    "938_left.jpg",
    "158_right.jpg",
    "162_right.jpg",
    "2165_right.jpg",
    "2170_left.jpg",
    "2158_right.jpg"
]

SIZE = (220, 220)
ROW_HEIGHT = 290


def load_image(path):
    path = ROOT / path

    if not path.exists():
        return None

    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail(SIZE)
    return im


def main():
    df = pd.read_csv(CSV)

    selected = df[
        (df["mdd_file"].isin(TARGETS)) &
        (df["mdd_file"] == df["odir_file"])
    ].copy()

    # Keep one row per ODIR version for each MDD image.
    selected = selected.drop_duplicates(
        subset=["mdd_file", "odir_version"]
    )

    sheet = Image.new(
        "RGB",
        (SIZE[0] * 3, ROW_HEIGHT * len(TARGETS)),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, filename in enumerate(TARGETS):

        group = selected[selected["mdd_file"] == filename]

        if group.empty:
            print("No matching comparison:", filename)
            continue

        y = index * ROW_HEIGHT

        draw.text(
            (5, y + 5),
            filename,
            fill="black"
        )

        # MDD image
        record = group.iloc[0]
        mdd = load_image(record["mdd_path"])

        if mdd:
            sheet.paste(mdd, (5, y + 35))

        draw.text((5, y + 260), "MDD", fill="black")

        for version_index, version in enumerate(
            ["original", "preprocessed"]
        ):

            matches = group[group["odir_version"] == version]

            if matches.empty:
                continue

            record = matches.iloc[0]
            odir = load_image(record["odir_path"])

            x = SIZE[0] * (version_index + 1) + 5

            if odir:
                sheet.paste(odir, (x, y + 35))

            draw.text(
                (x, y + 260),
                f"ODIR {version}",
                fill="black"
            )

    sheet.save(OUTPUT, quality=95)
    print("Created:", OUTPUT)


if __name__ == "__main__":
    main()
