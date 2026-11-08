
import pandas as pd
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DECISIONS = ROOT / "datasets/manifests/duplicate_resolution_decisions.csv"
PLAN = ROOT / "datasets/manifests/duplicate_representative_plan.csv"

OUT_DECISIONS = ROOT / "datasets/manifests/duplicate_resolution_decisions_final.csv"
OUT_PLAN = ROOT / "datasets/manifests/duplicate_representative_plan_final.csv"


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():

    decisions = pd.read_csv(DECISIONS)
    plan = pd.read_csv(PLAN)

    # Validate expected columns
    required_decisions = {
        "group_id", "dataset", "filename", "image_path",
        "label", "decision_status", "include_in_training",
        "decision_note"
    }

    required_plan = {
        "group_id", "dataset", "files", "labels",
        "sha256_identical", "proposed_representative",
        "decision_status"
    }

    if not required_decisions.issubset(decisions.columns):
        raise ValueError("Unexpected columns in duplicate decisions CSV.")

    if not required_plan.issubset(plan.columns):
        raise ValueError("Unexpected columns in representative plan CSV.")

    if len(plan) != 24 or len(decisions) != 48:
        raise ValueError("Expected exactly 24 groups and 48 records.")

    # Work on copies. Original audit files remain untouched.
    final_decisions = decisions.copy()
    final_plan = plan.copy()

    # Validate every group before writing anything.
    approved = 0
    held = 0

    for _, group_plan in plan.iterrows():

        group_id = group_plan["group_id"]
        dataset = group_plan["dataset"]

        rows = final_decisions[
            (final_decisions["group_id"] == group_id)
            & (final_decisions["dataset"] == dataset)
        ]

        if len(rows) != 2:
            raise ValueError(
                f"Group {group_id}: expected 2 records, found {len(rows)}"
            )

        paths = [Path(p) for p in rows["image_path"]]

        if not all(p.exists() for p in paths):
            raise FileNotFoundError(
                f"Group {group_id}: one or more image files are missing."
            )

        # Verify that both files are byte-identical.
        hashes = [sha256(p) for p in paths]
        identical = hashes[0] == hashes[1]

        labels_match = rows["label"].nunique(dropna=False) == 1

        # Conflicting-label groups must remain on hold.
        if not labels_match:

            final_decisions.loc[rows.index, "decision_status"] = (
                "HOLD_LABEL_CONFLICT"
            )

            final_decisions.loc[rows.index, "include_in_training"] = False

            final_decisions.loc[rows.index, "decision_note"] = (
                "Held: identical image files have conflicting labels. "
                "Verify source annotations before modelling."
            )

            final_plan.loc[
                final_plan["group_id"] == group_id,
                "decision_status"
            ] = "HOLD_LABEL_CONFLICT"

            held += 1
            print(f"Group {group_id}: HOLD - conflicting labels")
            continue

        # Matching labels are not enough; hashes must also match.
        if not identical:
            raise ValueError(
                f"Group {group_id}: SHA256 mismatch. "
                "Cannot approve this duplicate pair."
            )

        representative = str(
            group_plan["proposed_representative"]
        ).strip()

        matching_rows = rows[
            rows["filename"].astype(str) == representative
        ]

        if len(matching_rows) != 1:
            raise ValueError(
                f"Group {group_id}: representative filename "
                f"{representative} was not uniquely found."
            )

        representative_index = matching_rows.index[0]
        duplicate_indices = [
            idx for idx in rows.index if idx != representative_index
        ]

        # Keep representative eligible for a later data split.
        final_decisions.loc[
            representative_index, "decision_status"
        ] = "KEEP_REPRESENTATIVE_PENDING_SPLIT"

        final_decisions.loc[
            representative_index, "include_in_training"
        ] = True

        final_decisions.loc[
            representative_index, "decision_note"
        ] = (
            "Approved representative: exact SHA256 match and matching "
            "label. Eligible for later train/validation/test assignment; "
            "not assigned to a split yet."
        )

        # Exclude redundant copy from modelling.
        final_decisions.loc[
            duplicate_indices, "decision_status"
        ] = "EXCLUDE_EXACT_DUPLICATE"

        final_decisions.loc[
            duplicate_indices, "include_in_training"
        ] = False

        final_decisions.loc[
            duplicate_indices, "decision_note"
        ] = (
            f"Excluded from modelling because it is byte-identical "
            f"to representative {representative}. Raw file retained."
        )

        final_plan.loc[
            final_plan["group_id"] == group_id,
            "decision_status"
        ] = "FINALIZED_KEEP_ONE"

        approved += 1
        print(
            f"Group {group_id}: APPROVED - "
            f"keep {representative}"
        )

    # Save new files only after all groups pass validation.
    final_decisions.to_csv(OUT_DECISIONS, index=False)
    final_plan.to_csv(OUT_PLAN, index=False)

    print("\n========== FINAL SUMMARY ==========")
    print("Approved matching-label groups:", approved)
    print("Groups held for label conflicts:", held)
    print("Representatives retained:", approved)
    print("Redundant copies excluded:", approved)
    print("Records held:", held * 2)
    print("\nCreated:")
    print(OUT_DECISIONS)
    print(OUT_PLAN)
    print("\nOriginal CSV files and raw images were not modified.")


if __name__ == "__main__":
    main()
