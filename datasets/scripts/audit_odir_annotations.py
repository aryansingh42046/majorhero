
from pathlib import Path
import pandas as pd

ROOT = Path("datasets/raw/ODIR/odir dataset")
CSV_PATH = ROOT / "full_df.csv"
XLSX_PATH = ROOT / "ODIR-5K" / "ODIR-5K" / "data.xlsx"
OUTPUT = Path("datasets/manifests/odir_annotation_audit.txt")

LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]

csv = pd.read_csv(CSV_PATH)
xlsx = pd.read_excel(XLSX_PATH)

csv["ID"] = pd.to_numeric(csv["ID"]).astype(int)
xlsx["ID"] = pd.to_numeric(xlsx["ID"]).astype(int)

for col in LABELS:
    csv[col] = pd.to_numeric(csv[col], errors="coerce")
    xlsx[col] = pd.to_numeric(xlsx[col], errors="coerce")

merged = csv[["ID", "filename"] + LABELS].merge(
    xlsx[["ID"] + LABELS],
    on="ID",
    how="left",
    suffixes=("_csv", "_xlsx"),
    validate="many_to_one",
)

mismatches = 0
for col in LABELS:
    mismatches += int(
        merged[f"{col}_csv"].ne(merged[f"{col}_xlsx"]).sum()
    )

train_dir = ROOT / "ODIR-5K" / "ODIR-5K" / "Training Images"
test_dir = ROOT / "ODIR-5K" / "ODIR-5K" / "Testing Images"

def image_names(folder):
    return {
        p.name for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

train_names = image_names(train_dir)
test_names = image_names(test_dir)
csv_names = set(csv["filename"].dropna().astype(str))
missing_names = train_names - csv_names

rows_per_patient = csv.groupby("ID").size()
excel_ids = set(xlsx["ID"])
csv_ids = set(csv["ID"])

lines = [
    "ODIR ANNOTATION AUDIT",
    "=====================",
    f"CSV eye records: {len(csv)}",
    f"Excel patient records: {len(xlsx)}",
    f"CSV unique patients: {len(csv_ids)}",
    f"Excel-only patients: {len(excel_ids - csv_ids)}",
    f"Original training images: {len(train_names)}",
    f"CSV-referenced training images: {len(train_names & csv_names)}",
    f"Training images absent from CSV: {len(missing_names)}",
    f"Original testing images: {len(test_names)}",
    f"CSV filenames found in testing folder: {len(test_names & csv_names)}",
    f"Rows with label mismatches vs Excel: {mismatches}",
    f"Patients with one CSV eye record: {int((rows_per_patient == 1).sum())}",
    f"Patients with two CSV eye records: {int((rows_per_patient == 2).sum())}",
    "",
    "Limitations:",
    "- Label agreement does not establish eye-specific ground truth.",
    "- Original split provenance remains unverified.",
    "- Do not infer labels for unannotated records.",
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved report: {OUTPUT}")
