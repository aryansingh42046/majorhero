
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
rows = []

def add_image(path, dataset, label, patient_id=""):
    rows.append({
        "dataset": dataset,
        "image_path": str(path.resolve()),
        "filename": path.name,
        "label": label,
        "patient_id": str(patient_id),
    })

# ACRIMA: labels come from class folder names
for path in (RAW / "ACRIMA").rglob("*"):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        parent_names = [part.lower() for part in path.parts]
        label = ""
        if "glaucoma" in parent_names:
            label = "glaucoma"
        elif "non glaucoma" in parent_names:
            label = "non_glaucoma"
        if label:
            add_image(path, "ACRIMA", label)

# IDRiD: use the dataset's supplied CSV

# IDRiD: match labels to actual image filenames
idrid_root = RAW / "IDRid"
idrid_csvs = list(idrid_root.rglob("idrid_labels.csv"))

if idrid_csvs:
    csv_path = idrid_csvs[0]
    df = pd.read_csv(csv_path)

    # Index image files by filename stem
    image_files = [
        p for p in idrid_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and p.name.lower() != "idrid_labels.csv"
    ]

    image_index = {}
    for p in image_files:
        image_index.setdefault(p.stem.lower(), p)

    for _, row in df.iterrows():
        image_id = str(row["id_code"]).strip()
        diagnosis = row["diagnosis"]

        # Prefer the exact image ID; do not accidentally use *test.jpg copies
        image_path = image_index.get(image_id.lower())

        if image_path is not None:
            add_image(
                image_path,
                "IDRiD",
                diagnosis,
                image_id
            )
        else:
            print(f"IDRiD image not found: {image_id}")


# MDD: use class folder names, not severity grades
for path in (RAW / "MDD").rglob("*"):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        parent_names = [part.lower() for part in path.parts]
        if "amd" in parent_names:
            add_image(path, "MDD", "amd")
        elif "cataract" in parent_names:
            add_image(path, "MDD", "cataract")

# ODIR: one row per individual eye image
odir_csvs = list((RAW / "ODIR").rglob("full_df.csv"))
if odir_csvs:
    df = pd.read_csv(odir_csvs[0])
    image_root = RAW / "ODIR"
    for _, row in df.drop_duplicates(
        subset=["ID", "filename"]
    ).iterrows():
        filename = str(row["filename"])
        matches = list(image_root.rglob(filename))
        if matches:
            add_image(
                matches[0], "ODIR", row["labels"], row["ID"]
            )

manifest = pd.DataFrame(rows)
if not manifest.empty:
    manifest = manifest.drop_duplicates(
        subset=["dataset", "image_path"]
    )
    output_path = OUT / "dataset_manifest.csv"
    manifest.to_csv(output_path, index=False)
    print("Manifest saved:", output_path)
    print("\nImages recorded per dataset:")
    print(manifest.groupby("dataset").size().to_string())
else:
    print("No images were recorded. Check dataset paths and label files.")
