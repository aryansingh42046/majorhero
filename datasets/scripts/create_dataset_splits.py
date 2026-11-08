
from pathlib import Path
import ast
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "datasets" / "manifests" / "split_grouping_manifest.csv"

OUTPUT_PATH = ROOT / "datasets" / "manifests" / "final_dataset_splits.csv"
REPORT_PATH = ROOT / "datasets" / "manifests" / "split_distribution_report.csv"

RANDOM_STATE = 42

ODIR_LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def parse_odir_labels(value):
    """Convert ODIR label strings such as ['N', 'D'] into lists."""

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        result = ast.literal_eval(str(value))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        raise ValueError(f"Could not parse ODIR label: {value}")


def split_three_way(groups, label_column, seed=RANDOM_STATE):
    """
    Split independent groups into approximately
    80% train, 10% validation, 10% test.
    """

    groups = groups.copy()

    train_groups, temporary_groups = train_test_split(
        groups,
        test_size=0.20,
        random_state=seed,
        stratify=groups[label_column]
    )

    validation_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=0.50,
        random_state=seed,
        stratify=temporary_groups[label_column]
    )

    result = {}

    for group_id in train_groups["split_group_id"]:
        result[group_id] = "train"

    for group_id in validation_groups["split_group_id"]:
        result[group_id] = "validation"

    for group_id in test_groups["split_group_id"]:
        result[group_id] = "test"

    return result


def make_group_table(df, label_column="label"):
    """Create one row per independent split group."""

    return (
        df.groupby("split_group_id", as_index=False)
        .agg({label_column: "first"})
    )


# --------------------------------------------------
# LOAD AND VALIDATE MANIFEST
# --------------------------------------------------

print("\nLoading dataset manifest...")

df = pd.read_csv(MANIFEST_PATH)

required_columns = [
    "dataset",
    "label",
    "patient_id",
    "source_split_status",
    "modelling_status",
    "split_group_id"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df[df["modelling_status"] == "ELIGIBLE_PENDING_SPLIT"].copy()

if df.empty:
    raise ValueError("No eligible records found.")

if df["split_group_id"].isna().any():
    raise ValueError("Some eligible records have missing split_group_id.")

if df["split_group_id"].duplicated().all():
    print("Note: Multiple images may belong to the same split group.")

print("Eligible records:", len(df))
print(df["dataset"].value_counts().to_string())


# --------------------------------------------------
# GLOBAL SPLIT ASSIGNMENTS
# --------------------------------------------------

group_split = {}


# --------------------------------------------------
# 1. ODIR: PATIENT-LEVEL MULTI-LABEL SPLITTING
# --------------------------------------------------

print("\nSplitting ODIR by patient...")

odir = df[df["dataset"] == "ODIR"].copy()

odir["parsed_labels"] = odir["label"].apply(parse_odir_labels)

# Combine both eyes' disease labels for each patient.
patient_rows = []

for group_id, group in odir.groupby("split_group_id"):

    combined_labels = set()

    for labels in group["parsed_labels"]:
        combined_labels.update(labels)

    patient_rows.append({
        "split_group_id": group_id,
        "labels": sorted(combined_labels)
    })

patient_table = pd.DataFrame(patient_rows)

if patient_table.empty:
    raise ValueError("No ODIR patient groups found.")

Y = np.array([
    [int(label in labels) for label in ODIR_LABELS]
    for labels in patient_table["labels"]
])

if not np.all(Y.sum(axis=0) > 0):
    raise ValueError("At least one ODIR label has no positive patient groups.")

# First: 80% train, 20% temporary.
splitter_1 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, temp_idx = next(
    splitter_1.split(np.zeros(len(Y)), Y)
)

# Second: divide the temporary 20% into 10% validation and 10% test.
Y_temp = Y[temp_idx]

splitter_2 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE
)

val_relative_idx, test_relative_idx = next(
    splitter_2.split(np.zeros(len(Y_temp)), Y_temp)
)

validation_idx = temp_idx[val_relative_idx]
test_idx = temp_idx[test_relative_idx]

for idx in train_idx:
    group_split[patient_table.iloc[idx]["split_group_id"]] = "train"

for idx in validation_idx:
    group_split[patient_table.iloc[idx]["split_group_id"]] = "validation"

for idx in test_idx:
    group_split[patient_table.iloc[idx]["split_group_id"]] = "test"

print("ODIR patient groups:", len(patient_table))


# --------------------------------------------------
# 2. ACRIMA: PRESERVE ORIGINAL TEST SET
# --------------------------------------------------

print("\nSplitting ACRIMA...")

acrima = df[df["dataset"] == "ACRIMA"].copy()

acrima_test = acrima[
    acrima["source_split_status"] == "source_test"
]

acrima_train = acrima[
    acrima["source_split_status"] == "source_train"
]

if len(acrima_test) != 140:
    raise ValueError(
        f"Expected 140 original ACRIMA test images, found {len(acrima_test)}"
    )

for group_id in acrima_test["split_group_id"].unique():
    group_split[group_id] = "test"

acrima_groups = make_group_table(acrima_train)

acrima_train_groups, acrima_val_groups = train_test_split(
    acrima_groups,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=acrima_groups["label"]
)

for group_id in acrima_train_groups["split_group_id"]:
    group_split[group_id] = "train"

for group_id in acrima_val_groups["split_group_id"]:
    group_split[group_id] = "validation"

print("ACRIMA original test images:", len(acrima_test))


# --------------------------------------------------
# 3. IDRiD: IMAGE/GROUP-LEVEL STRATIFIED SPLITTING
# --------------------------------------------------

print("\nSplitting IDRiD...")

idrid = df[df["dataset"] == "IDRiD"].copy()

idrid_groups = make_group_table(idrid)

idrid_assignments = split_three_way(
    idrid_groups,
    label_column="label"
)

group_split.update(idrid_assignments)

print("IDRiD groups:", len(idrid_groups))


# --------------------------------------------------
# 4. MDD: PROPAGATE LINKED ODIR SPLITS FIRST
# --------------------------------------------------

print("\nSplitting MDD...")

mdd = df[df["dataset"] == "MDD"].copy()

# MDD images linked to ODIR patients must inherit
# the exact same split as their corresponding ODIR patient.
linked_mdd = mdd[
    mdd["split_group_id"].isin(group_split.keys())
]

for group_id in linked_mdd["split_group_id"].unique():
    if group_id not in group_split:
        raise ValueError(f"Missing ODIR assignment for {group_id}")

print("Linked MDD records:", len(linked_mdd))

# Split only MDD groups that are not linked to ODIR.
unlinked_mdd = mdd[
    ~mdd["split_group_id"].isin(group_split.keys())
]

mdd_groups = make_group_table(unlinked_mdd)

if not mdd_groups.empty:
    mdd_assignments = split_three_way(
        mdd_groups,
        label_column="label"
    )

    group_split.update(mdd_assignments)

print("Independent MDD groups:", len(mdd_groups))


# --------------------------------------------------
# 5. ASSIGN SPLITS TO ALL ELIGIBLE IMAGES
# --------------------------------------------------

print("\nCreating final split manifest...")

df["split"] = df["split_group_id"].map(group_split)

if df["split"].isna().any():
    missing = df[df["split"].isna()][
        ["dataset", "filename", "split_group_id"]
    ]

    raise ValueError(
        "Some records have no split assignment:\n"
        + missing.to_string(index=False)
    )


# --------------------------------------------------
# 6. LEAKAGE VALIDATION
# --------------------------------------------------

print("\nChecking for group leakage...")

group_split_counts = df.groupby("split_group_id")["split"].nunique()

if (group_split_counts > 1).any():
    raise ValueError("LEAKAGE DETECTED: A group appears in multiple splits.")

# Every linked MDD–ODIR pair must have the same split.
linked = df[df["cross_dataset_duplicate_group_id"].notna()]

for cross_id, group in linked.groupby("cross_dataset_duplicate_group_id"):

    if group["split"].nunique() != 1:
        raise ValueError(
            f"Cross-dataset leakage detected in {cross_id}"
        )

    if set(group["dataset"]) != {"MDD", "ODIR"}:
        raise ValueError(
            f"Unexpected datasets in cross-dataset group {cross_id}"
        )

# Verify ACRIMA's original test records remain untouched.
if not (
    df[
        (df["dataset"] == "ACRIMA")
        & (df["source_split_status"] == "source_test")
    ]["split"] == "test"
).all():
    raise ValueError("ACRIMA original test set was not preserved.")

print("No split-group leakage detected.")
print("All six linked groups checked.")


# --------------------------------------------------
# 7. SAVE FINAL MANIFEST
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nFinal split counts:")

print(
    df.groupby(["dataset", "split"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)


# --------------------------------------------------
# 8. SAVE CLASS DISTRIBUTION REPORT
# --------------------------------------------------

report_rows = []

for dataset in df["dataset"].unique():

    subset = df[df["dataset"] == dataset]

    for split_name in ["train", "validation", "test"]:

        split_df = subset[subset["split"] == split_name]

        if dataset == "ODIR":

            for label in ODIR_LABELS:

                positive_count = sum(
                    label in parse_odir_labels(value)
                    for value in split_df["label"]
                )

                report_rows.append({
                    "dataset": dataset,
                    "split": split_name,
                    "label": label,
                    "positive_images": positive_count
                })

        else:

            counts = split_df["label"].value_counts()

            for label, count in counts.items():

                report_rows.append({
                    "dataset": dataset,
                    "split": split_name,
                    "label": label,
                    "positive_images": count
                })

report = pd.DataFrame(report_rows)
report.to_csv(REPORT_PATH, index=False)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\nSUCCESS!")
print("Final manifest:", OUTPUT_PATH)
print("Distribution report:", REPORT_PATH)
print("Random seed:", RANDOM_STATE)
