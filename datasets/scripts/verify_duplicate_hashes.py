import csv
import hashlib
from pathlib import Path

root = Path.cwd()
source = root / "datasets/manifests/duplicate_resolution_decisions.csv"
output = root / "datasets/manifests/exact_duplicate_hash_verification.csv"

with source.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

groups = {}
for row in rows:
    groups.setdefault(row["group_id"], []).append(row)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

results = []
for group_id, members in groups.items():
    if len(members) != 2:
        results.append({
            "group_id": group_id,
            "dataset": members[0]["dataset"],
            "file1": members[0]["filename"],
            "file2": "",
            "hash1": "",
            "hash2": "",
            "sha256_identical": "UNVERIFIED",
            "notes": f"Expected 2 records; found {len(members)}"
        })
        continue

    a, b = members
    pa = Path(a["image_path"])
    pb = Path(b["image_path"])

    if not pa.is_absolute():
        pa = root / pa
    if not pb.is_absolute():
        pb = root / pb

    if not pa.is_file() or not pb.is_file():
        results.append({
            "group_id": group_id,
            "dataset": a["dataset"],
            "file1": a["filename"],
            "file2": b["filename"],
            "hash1": "",
            "hash2": "",
            "sha256_identical": "MISSING_FILE",
            "notes": f"File1 exists={pa.is_file()}; file2 exists={pb.is_file()}"
        })
        continue

    ha, hb = sha256(pa), sha256(pb)
    results.append({
        "group_id": group_id,
        "dataset": a["dataset"],
        "file1": a["filename"],
        "file2": b["filename"],
        "hash1": ha,
        "hash2": hb,
        "sha256_identical": str(ha == hb),
        "notes": "Hash equality checks byte identity only"
    })

fields = [
    "group_id", "dataset", "file1", "file2",
    "hash1", "hash2", "sha256_identical", "notes"
]
with output.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"Created: {output}")
print(f"Groups checked: {len(results)}")
for key in ("True", "False", "MISSING_FILE", "UNVERIFIED"):
    print(f"{key}: {sum(r['sha256_identical'] == key for r in results)}")
