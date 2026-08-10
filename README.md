# Multimodal Mammography XAI

An explainable multimodal breast-cancer classification project using the **CBIS-DDSM** mammography dataset.

The project is being developed for the University of London CM3070 Final Project. The planned system combines:

- a convolutional neural network (CNN) branch for full mammograms;
- radiomics features extracted from annotated lesions;
- multimodal fusion of CNN and radiomics features; and
- explainable AI methods such as Grad-CAM, Grad-CAM++, and Integrated Gradients.

The current implementation has completed the main **CBIS-DDSM data preparation, path resolution, DICOM conversion, quality assurance, deduplication, and patient-disjoint dataset splitting** stages.

## Current status

The CBIS-DDSM download was first verified using `metadata.csv`.

| Processing stage | Result |
|---|---:|
| Metadata records | 6,775 |
| Valid metadata directories | 6,775 / 6,775 |
| DICOM files discovered | 10,239 |
| Case-description rows | 3,568 |
| Full mammograms resolved | 3,568 / 3,568 |
| Cropped images resolved | 3,568 / 3,568 |
| ROI masks resolved | 3,248 / 3,568 |
| Ambiguous final matches | 0 |
| Reversed crop/ROI CSV references corrected | 1 |
| Unique full mammograms | 3,103 |
| Unique participants | 1,566 |
| Benign full mammograms | 1,728 |
| Malignant full mammograms | 1,375 |

The 320 unavailable ROI masks are all from the calcification test subset.

Quality assurance confirmed that all converted full mammograms and cropped images were valid and that all 3,248 available ROI masks contained foreground pixels.

## Patient-disjoint split

The original CBIS-DDSM mass and calcification train/test files are individually patient-disjoint, but combining them introduces cross-task participant overlap:

- `calc_train` vs `calc_test`: 0 participants
- `mass_train` vs `mass_test`: 0 participants
- `calc_train` vs `mass_test`: 18 participants
- `mass_train` vs `calc_test`: 13 participants

A new patient-disjoint split was therefore created for the combined full-mammogram experiment.

| Split | Participants | Images | Benign | Malignant |
|---|---:|---:|---:|---:|
| Train | 1,096 | 2,193 | 1,215 | 978 |
| Validation | 235 | 456 | 255 | 201 |
| Test | 235 | 454 | 258 | 196 |
| **Total** | **1,566** | **3,103** | **1,728** | **1,375** |

There is zero participant overlap between the three new splits.

## Recommended repository structure

```text
multimodal-mammography-xai/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                    # not tracked by Git
│   │   └── cbis_ddsm/
│   ├── interim/                # generated manifests / intermediate files
│   └── processed/              # converted images; normally not tracked
│       └── cbis_ddsm/
│           ├── full_mammograms/
│           ├── cropped/
│           ├── roi_masks/
│           └── manifests/
│
├── src/
│   ├── preprocessing/
│   │   ├── checking_paths.py
│   │   ├── find_dicom_files.py
│   │   ├── inspect_one_case.py
│   │   ├── classify_dicom_images.py
│   │   ├── resolve_cbisd_paths.py
│   │   ├── resolve_cbisd_paths_v2.py
│   │   ├── convert_cbisd_images.py
│   │   ├── qa_processed_images.py
│   │   ├── build_full_image_manifest.py
│   │   └── create_patient_disjoint_split.py
│   │
│   ├── models/
│   ├── radiomics/
│   ├── fusion/
│   ├── xai/
│   └── utils/
│
├── notebooks/
│   ├── exploration/
│   ├── modelling/
│   └── evaluation/
│
├── models/                     # saved checkpoints; usually ignored
├── results/
│   ├── figures/
│   ├── metrics/
│   └── xai/
│
├── tests/
└── docs/
    └── report/
```

The directory names under `src/` contain reusable Python code. Notebooks are intended for exploration, experiments, and presentation rather than the main data pipeline.

## Data

The raw dataset is **not included in this GitHub repository** because CBIS-DDSM is large and is distributed separately.

Download CBIS-DDSM from The Cancer Imaging Archive (TCIA), then place the downloaded collection under:

```text
data/raw/cbis_ddsm/
```

Do not commit the TCIA downloader, downloaded ZIP files, DICOM data, converted mammograms, virtual environments, or trained model weights to Git.

## Main preprocessing pipeline

The preprocessing work currently follows this sequence:

```text
metadata.csv
    ↓
verify local paths
    ↓
discover DICOM files
    ↓
inspect and classify DICOM image roles
    ↓
resolve CBIS-DDSM CSV paths using Study/Series UIDs
    ↓
correct crop/ROI role inconsistencies
    ↓
convert DICOM images
    ↓
quality assurance
    ↓
deduplicate full mammograms
    ↓
create image-level labels
    ↓
create patient-disjoint train/validation/test split
```

### Important CBIS-DDSM path issue

The case-description CSV paths do not directly match the physical filenames in the downloaded data. The local download contains UUID-style DICOM filenames and numerical directory names.

The resolver therefore uses DICOM metadata, especially:

- `StudyInstanceUID`
- `SeriesInstanceUID`
- `SeriesDescription`
- image role

rather than relying on the original `000000.dcm` filenames.

One calcification training record (`P_00474`) was found to have crop and ROI references reversed. The final resolver uses the actual DICOM image role to resolve this safely.

## Key manifests

The main generated manifests are:

```text
data/processed/cbis_ddsm/manifests/
├── cbis_ddsm_processed_manifest.csv
├── cbis_ddsm_qa_manifest.csv
├── cbis_ddsm_full_image_manifest.csv
└── cbis_ddsm_full_image_manifest_split.csv
```

`cbis_ddsm_full_image_manifest_split.csv` is the current main manifest for the combined full-mammogram classification experiment.

## Environment setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Main packages currently used include:

```text
pandas
numpy
pydicom
Pillow
tqdm
scikit-learn
pylibjpeg
pylibjpeg-libjpeg
pylibjpeg-openjpeg
tensorflow
```

The exact versions should be pinned in `requirements.txt` before final submission.

## Running the preprocessing scripts

Run scripts from the repository root so that relative paths are consistent.

Example:

```bash
python src/preprocessing/checking_paths.py
python src/preprocessing/find_dicom_files.py
python src/preprocessing/classify_dicom_images.py
python src/preprocessing/resolve_cbisd_paths_v2.py
python src/preprocessing/convert_cbisd_images.py
python src/preprocessing/qa_processed_images.py
python src/preprocessing/build_full_image_manifest.py
python src/preprocessing/create_patient_disjoint_split.py
```

The older `resolve_cbisd_paths.py` is retained for development history, but `resolve_cbisd_paths_v2.py` is the current resolver.

## Planned next stages

The next stages are:

1. full-mammogram background removal;
2. left/right orientation normalisation;
3. aspect-ratio-preserving resizing;
4. CNN baseline training;
5. radiomics feature extraction;
6. multimodal fusion;
7. Grad-CAM, Grad-CAM++, and Integrated Gradients;
8. quantitative XAI comparison with ROI masks; and
9. final model evaluation.

## Reproducibility

To keep the project reproducible:

- raw DICOM data is never modified;
- derived images are written to `data/processed/`;
- all important preprocessing decisions are recorded in CSV manifests;
- full-mammogram duplicates are removed at the source-DICOM level;
- the combined classification experiment uses a patient-disjoint split; and
- random seeds should be fixed for all training and splitting scripts.

## Notes on GitHub storage

This repository should contain **code, configuration files, documentation, small manifests, and selected result figures**.

Large files should not be committed directly. In particular, exclude:

- raw CBIS-DDSM data;
- converted full mammograms;
- converted cropped images;
- ROI mask image folders;
- TCIA download archives;
- `.venv/` or `venv/`;
- model checkpoints;
- TensorBoard logs;
- cache files.

If selected model checkpoints are later needed for release, use Git LFS or a separate release artifact.

## License and dataset attribution

The source code in this repository is covered by the repository `LICENSE`.

CBIS-DDSM is a separate third-party dataset and remains subject to its own terms and attribution requirements. The dataset itself is not redistributed in this repository.
