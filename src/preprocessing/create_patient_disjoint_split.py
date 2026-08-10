import pandas as pd
from sklearn.model_selection import train_test_split


INPUT = (
    "../data/processed/cbis_ddsm/"
    "cbis_ddsm_full_image_manifest.csv"
)

OUTPUT = (
    "../data/processed/cbis_ddsm/"
    "cbis_ddsm_full_image_manifest_split.csv"
)

RANDOM_STATE = 42


df = pd.read_csv(INPUT)


# ============================================================
# Build participant-level summary
# ============================================================

def participant_label(group):
    """
    If any image for the participant is malignant,
    classify the participant as malignant for stratification.
    """
    if (group["binary_pathology"] == "MALIGNANT").any():
        return "MALIGNANT"

    return "BENIGN"


def participant_type(group):
    """
    Summarize lesion types for stratification.
    """

    has_mass = group["contains_mass"].any()
    has_calc = group["contains_calcification"].any()

    if has_mass and has_calc:
        return "mass+calc"

    elif has_mass:
        return "mass"

    elif has_calc:
        return "calc"

    return "unknown"


participant_records = []

for patient_id, group in df.groupby("patient_id"):

    participant_records.append({
        "patient_id":
            patient_id,

        "participant_pathology":
            participant_label(group),

        "participant_type":
            participant_type(group),

        "n_images":
            len(group),
    })


patients = pd.DataFrame(participant_records)


# ============================================================
# Combined stratification key
# ============================================================

patients["stratum"] = (
    patients["participant_pathology"]
    + "_"
    + patients["participant_type"]
)


print("=" * 72)
print("PARTICIPANT SUMMARY")
print("=" * 72)

print("\nParticipants:")
print(len(patients))

print("\nParticipant pathology:")
print(
    patients["participant_pathology"]
    .value_counts()
)

print("\nParticipant lesion type:")
print(
    patients["participant_type"]
    .value_counts()
)

print("\nStrata:")
print(
    patients["stratum"]
    .value_counts()
)


# ============================================================
# Split 70 / 15 / 15
# ============================================================

train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=patients["stratum"],
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_patients["stratum"],
)


# ============================================================
# Assign split
# ============================================================

split_map = {}

for patient_id in train_patients["patient_id"]:
    split_map[patient_id] = "train"

for patient_id in val_patients["patient_id"]:
    split_map[patient_id] = "validation"

for patient_id in test_patients["patient_id"]:
    split_map[patient_id] = "test"


df["split"] = df["patient_id"].map(split_map)


# ============================================================
# Validation
# ============================================================

print("\n" + "=" * 72)
print("SPLIT SUMMARY")
print("=" * 72)


print("\nParticipants per split:")

for split in [
    "train",
    "validation",
    "test"
]:

    subset = df[df["split"] == split]

    print(
        f"{split:12s}: "
        f"{subset['patient_id'].nunique():4d} participants, "
        f"{len(subset):4d} images"
    )


print("\nImage pathology per split:")

print(
    pd.crosstab(
        df["split"],
        df["binary_pathology"]
    )
)


print("\nLesion type per split:")

print(
    pd.crosstab(
        df["split"],
        [
            df["contains_mass"],
            df["contains_calcification"]
        ]
    )
)


# ============================================================
# Leakage checks
# ============================================================

train_ids = set(
    df[df["split"] == "train"]["patient_id"]
)

val_ids = set(
    df[df["split"] == "validation"]["patient_id"]
)

test_ids = set(
    df[df["split"] == "test"]["patient_id"]
)


print("\nPatient overlap checks:")

print(
    "train ∩ validation:",
    len(train_ids & val_ids)
)

print(
    "train ∩ test:",
    len(train_ids & test_ids)
)

print(
    "validation ∩ test:",
    len(val_ids & test_ids)
)


assert len(train_ids & val_ids) == 0
assert len(train_ids & test_ids) == 0
assert len(val_ids & test_ids) == 0


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)