import csv
from pathlib import Path

out = Path("datasets/manifests/manual_duplicate_decisions.csv")
out.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "mdd_path",
    "odir_path",
    "decision",
    "evidence",
    "sha256_identical",
    "notes"
]

mdd = (
    r"datasets\raw\MDD\mdd dataset\AMDNet23 Fundus Image Dataset for Age-Related Macular Degeneration Disease Detection"
    r"\AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection"
    r"\AMDNet23 Dataset\train\cataract\2158_right.jpg"
)

odir = r"datasets\raw\ODIR\odir dataset\preprocessed_images\2158_right.jpg"

row = {
    "mdd_path": mdd,
    "odir_path": odir,
    "decision": "LIKELY_DUPLICATE",
    "evidence": "MANUAL_VISUAL_REVIEW",
    "sha256_identical": "false",
    "notes": "Images appear to show the same underlying photograph; file hashes differ. Retain both source files; group for leakage prevention after review."
}

existing = []
if out.exists():
    with out.open("r", newline="", encoding="utf-8-sig") as f:
        existing = list(csv.DictReader(f, delimiter=","))

key = (row["mdd_path"].lower(), row["odir_path"].lower())
existing = [
    r for r in existing
    if (r.get("mdd_path", "").lower(), r.get("odir_path", "").lower()) != key
]
existing.append(row)

with out.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(existing)

print(f"Saved review decisions: {out}")
print(f"Total recorded decisions: {len(existing)}")
