import csv
from pathlib import Path

root = Path.cwd()
source = root / "datasets/manifests/duplicate_resolution_decisions.csv"
hash_report = root / "datasets/manifests/exact_duplicate_hash_verification.csv"
output = root / "datasets/manifests/duplicate_representative_plan.csv"

with source.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

with hash_report.open("r", encoding="utf-8-sig", newline="") as f:
    hashes = list(csv.DictReader(f))

hash_by_group = {r["group_id"]: r for r in hashes}
groups = {}
for r in rows:
    groups.setdefault(r["group_id"], []).append(r)

result = []
for group_id, members in groups.items():
    members = sorted(members, key=lambda r: r["filename"].lower())
    verification = hash_by_group.get(group_id, {})
    exact = verification.get("sha256_identical") == "True"
    conflict = any(r["label_conflict"].lower() == "true" for r in members)

    # Prefer a clean filename only as a proposal, not a final automatic decision.
    clean = [r for r in members if "(1)" not in r["filename"] and not r["filename"].lower().startswith("copy of ")]
    proposed = clean[0] if clean else members[0]

    if conflict:
        status = "HOLD_LABEL_CONFLICT"
        proposed_name = ""
        note = "Do not select a representative until labels are verified."
    elif exact:
        status = "PROPOSED_NOT_FINAL"
        proposed_name = proposed["filename"]
        note = "Exact hashes match. Review proposal before using in modelling manifest."
    else:
        status = "HOLD_HASH_NOT_CONFIRMED"
        proposed_name = ""
        note = "Hashes did not confirm an exact duplicate."

    result.append({
        "group_id": group_id,
        "dataset": members[0]["dataset"],
        "files": " | ".join(r["filename"] for r in members),
        "labels": " | ".join(r["label"] for r in members),
        "sha256_identical": verification.get("sha256_identical", "UNVERIFIED"),
        "proposed_representative": proposed_name,
        "decision_status": status,
        "decision_note": note
    })

fields = [
    "group_id", "dataset", "files", "labels", "sha256_identical",
    "proposed_representative", "decision_status", "decision_note"
]
with output.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(result)

print(f"Created: {output}")
print(f"Groups listed: {len(result)}")
for status in sorted(set(r["decision_status"] for r in result)):
    print(f"{status}: {sum(r['decision_status'] == status for r in result)}")
