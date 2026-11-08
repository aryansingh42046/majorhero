
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "datasets/manifests/cross_dataset_version_comparison.csv"

OUTPUT_DIR = ROOT / "datasets/manifests"
OUTPUT_PATH = OUTPUT_DIR / "cross_dataset_comparison_sheet.jpg"

IMAGE_SIZE = (230, 230)
CARDS_PER_PAGE = 12
COLUMNS = 2
CARD_HEIGHT = 340


def resolve_path(value):
    path = Path(str(value))

    if path.is_absolute():
        return path

    return ROOT / path


def load_image(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail(IMAGE_SIZE)
    return image


def main():

    df = pd.read_csv(CSV_PATH)

    # Sort by recomputed distance, lowest first.
    df["recomputed_distance"] = pd.to_numeric(
        df["recomputed_distance"], errors="coerce"
    )

    df = df.sort_values(
        "recomputed_distance",
        na_position="last"
    )

    # Keep the first 24 comparisons for manual review.
    df = df.head(24).reset_index(drop=True)

    pages = math.ceil(len(df) / CARDS_PER_PAGE)

    font_title = None

    for page in range(pages):

        subset = df.iloc[
            page * CARDS_PER_PAGE:
            (page + 1) * CARDS_PER_PAGE
        ]

        rows = math.ceil(len(subset) / COLUMNS)

        sheet = Image.new(
            "RGB",
            (COLUMNS * 720, rows * CARD_HEIGHT),
            "white"
        )

        draw = ImageDraw.Draw(sheet)

        for index, record in enumerate(subset.itertuples()):

            row = index // COLUMNS
            col = index % COLUMNS

            x = col * 720
            y = row * CARD_HEIGHT

            mdd_path = resolve_path(record.mdd_path)
            odir_path = resolve_path(record.odir_path)

            draw.text(
                (x + 5, y + 5),
                f"MDD: {record.mdd_file} | Label: {record.mdd_label}",
                fill="black"
            )

            draw.text(
                (x + 5, y + 25),
                f"ODIR: {record.odir_file} | Label: {record.odir_label}",
                fill="black"
            )

            draw.text(
                (x + 5, y + 45),
                f"Version: {record.odir_version} | "
                f"Distance: {record.recomputed_distance}",
                fill="black"
            )

            # MDD image
            try:
                image = load_image(mdd_path)
                sheet.paste(image, (x + 5, y + 75))
            except Exception as error:
                draw.text(
                    (x + 5, y + 100),
                    f"MDD IMAGE ERROR: {error}",
                    fill="red"
                )

            # ODIR image
            try:
                image = load_image(odir_path)
                sheet.paste(image, (x + 350, y + 75))
            except Exception as error:
                draw.text(
                    (x + 350, y + 100),
                    f"ODIR IMAGE ERROR: {error}",
                    fill="red"
                )

            draw.text(
                (x + 5, y + 315),
                "LEFT: MDD                    RIGHT: ODIR",
                fill="black"
            )

        output = OUTPUT_DIR / (
            f"cross_dataset_comparison_page_{page + 1}.jpg"
        )

        sheet.save(output, quality=95)

        print("Created:", output)

    print("\nTotal comparisons reviewed:", len(df))
    print("Total pages:", pages)


if __name__ == "__main__":
    main()
