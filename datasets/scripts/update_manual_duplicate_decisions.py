
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMPARISON = ROOT / "datasets/manifests/cross_dataset_version_comparison.csv"
EXISTING = ROOT / "datasets/manifests/manual_duplicate_decisions.csv"

OUTPUT = ROOT / "datasets/manifests/manual_duplicate_decisions_reviewed.csv"

TARGETS = [
    "938_left.jpg",
    "158_right.jpg",
    "162_right.jpg",
]

def main():
    existing = pd.read_csv(EXISTING)
    comparisons = pd.read_csv(COMPARISON)

    additions = []

    for filename in TARGETS:
        candidates = comparisons[
            (comparisons["mdd_file"] == filename) &
            (comparisons["odir_file"] == filename) &
            (comparisons["odir_version"] == "preprocessed")
        ]

        if candidates.empty:
            raise ValueError(f"No preprocessed ODIR match found: {filename}")

        # Select one candidate only after checking matching labels.
        candidates = candidates.drop_duplicates(
            subset=["mdd_path", "odir_path"]
        )

        if len(candidates) != 1:
            raise ValueError(
                f"Multiple candidates for {filename}; manual review needed."
            )

        row = candidates.iloc[0]

        if not Path(row["mdd_path"]).exists():
            raise FileNotFoundError(row["mdd_path"])

        if not Path(row["odir_path"]).exists():
            raise FileNotFoundError(row["odir_path"])

        addition = {
            "mdd_path": row["mdd_path"],
            "odir_path": row["odir_path"],
            "decision": "LIKELY_DUPLICATE",
            "evidence": "MANUAL_VISUAL_REVIEW",
            "sha256_identical": False,
            "notes": (
                "Strong visual correspondence in manually reviewed "
                "comparison sheet. Hashes differ; not byte-identical. "
                "Retain source files and group for leakage prevention "
                "pending final verification."
            )
        }

        additions.append(addition)

    new_rows = pd.DataFrame(additions)

    # Avoid adding an existing MDD-ODIR path pair twice.
    existing_pairs = set(zip(existing["mdd_path"], existing["odir_path"]))

    new_rows = new_rows[
        ~new_rows.apply(
            lambda r: (r["mdd_path"], r["odir_path"]) in existing_pairs,
            axis=1
        )
    ]

    reviewed = pd.concat([existing, new_rows], ignore_index=True)

    reviewed.to_csv(OUTPUT, index=False)

    print("Existing decisions:", len(existing))
    print("New decisions added:", len(new_rows))
    print("Total reviewed records:", len(reviewed))
    print("\nDecision counts:")
    print(reviewed["decision"].value_counts().to_string())
    print("\nCreated:", OUTPUT)


if __name__ == "__main__":
    main()
