# RSNA Knee Abnormality Detection: verified overview

> Snapshot: 2026-08-09. Primary sources only. Counts below were verified against Kaggle's live API/CLI and the small CSV metadata files; the DICOM archive was **not** downloaded.

## At a glance

| Item | Verified detail |
| --- | --- |
| Competition | [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) |
| Host / category | Radiological Society of North America (RSNA) / Research |
| Goal | Predict a probability for each of 12 abnormalities for every knee MRI study. Training pairs DICOM MRI series with the study's original radiology report. [Kaggle description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/description) |
| Prize pool | USD 77,000 (Kaggle competition metadata) |
| Timeline | Start July 30, 2026; entry and team-merger deadline October 15; final submission deadline October 22 at 23:59 UTC. [Kaggle timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline) |
| Submission mode | Kaggle Notebook only; CPU or GPU runtime at most 9 hours, internet off, and output named `submission.csv`. Publicly available external data and pretrained models are allowed. [Kaggle code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements) |

RSNA describes this as its first challenge to combine medical images and radiology-report text. Its purpose is decision support for knee MRI interpretation across varied scanners, protocols, sites, and populations. [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)

## Prediction targets and metric

The output is a 12-label probability vector per `StudyInstanceUID`. The official definitions are in the [Kaggle data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data):

| Group | Target | Meaning |
| --- | --- | --- |
| Ligament | `ACL` | Anterior cruciate ligament injury |
| Ligament | `MCL` | Medial collateral ligament injury |
| Meniscus | `Medial Meniscus` | Medial meniscus tear |
| Meniscus | `Lateral Meniscus` | Lateral meniscus tear |
| Osteoarthritis | `Medial OA` | Medial tibiofemoral-compartment OA |
| Osteoarthritis | `Lateral OA` | Lateral tibiofemoral-compartment OA |
| Osteoarthritis | `PF OA` | Patellofemoral OA |
| Other | `Effusion` | Joint effusion / excess fluid |
| Other | `Synovitis` | Inflammation of the joint lining |
| Other | `Baker's` | Baker's cyst |
| Other | `Contusion` | Bone contusion / bone bruise |
| Other | `Fracture` | Fracture |

Kaggle scores the **unweighted macro-average ROC AUC** over the 12 targets, so every target contributes equally:

$$
\text{score}=\frac{1}{12}\sum_{i=1}^{12}\operatorname{AUC}_i
$$

The submission columns are `StudyInstanceUID` followed by the targets in the order shown above. [Kaggle evaluation and submission format](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)

## Live dataset snapshot

| Public item | Verified count / size | Notes |
| --- | ---: | --- |
| Training studies | 4,407 | One unique study per `train.csv` row; every row has a nonblank report. |
| Fully labeled training studies | 58 (1.316%) | The same 58 rows have all 12 binary labels; the other 4,349 have none of the 12 labels. |
| Training MRI series | 24,371 | 3-14 series per study; median 5, mean 5.53. |
| Training DICOM slices | 819,078 | Total DICOM count minus the 557 public-example test DICOMs. |
| Public test example | 3 studies, 15 series, 557 DICOMs | These are placeholders, not the scored test set. |
| Hidden scored test | About 1,300 studies | Kaggle replaces the three example studies and series metadata during scoring. |
| All currently listed files | 819,640 | 819,635 `.dcm` plus 5 `.csv` files. |
| Aggregate listed size | 569,763,893,144 bytes | 569.76 GB decimal / 530.63 GiB; sum of file bytes reported by Kaggle's data-summary API. |
| Compressed archive transfer size | 265,018,885,676 bytes | 265.02 GB decimal / 246.82 GiB; `Content-Length` from Kaggle's authenticated competition-download response, closed without reading its body. |

The exact counts in this table are derived from the official files/API as of the snapshot date; the written [Kaggle data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) is the source for the hidden-test estimate and replacement behavior.

### File layout

```text
train.csv
train_series.csv
train_series/
  <StudyInstanceUID>/
    <SeriesInstanceUID>/
      <SOPInstanceUID>.dcm
test.csv
test_series.csv
test_series/
  <StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
sample_submission.csv
```

- `train.csv` currently has 14 columns: `StudyInstanceUID`, `Report`, and the 12 targets. The report is free text and may be multilingual.
- `train_series.csv` has one row per MRI acquisition: `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane` (`Sagittal`, `Coronal`, or `Axial`). The current plane counts are 9,864 sagittal, 8,609 coronal, and 5,898 axial series.
- Each DICOM is one slice. Kaggle says a series typically has 20-45 slices (median 30), with a long tail into the hundreds.
- `test.csv`, `test_series.csv`, and `test_series/` are only a three-study public example. They are replaced by the hidden data inside the scoring run.
- `sample_submission.csv` contains the UID plus 12 probability columns, initially set to `0.5`.
- Transfer syntaxes vary: Explicit VR Little Endian, Implicit VR Little Endian, JPEG Lossless, and JPEG 2000. Kaggle says each DICOM was reduced to an allowlist of 86 metadata tags. [Kaggle data notes](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)

## Important implementation caveats

1. **Most labels must be derived from reports.** Only 58/4,407 studies have supplied expert labels; 4,349 have a report but no target values. This makes report parsing, translation, label extraction, and confidence-aware weak supervision central to the training pipeline.
2. **Reports are training-side data.** The published test schema contains only `StudyInstanceUID` plus image-series metadata, not a report. A final predictor must therefore work from MRI at inference time unless Kaggle changes the live schema.
3. **The documentation and live CSV disagree on `PatientSex`.** Kaggle's written data description lists a possibly blank `PatientSex` field, but the current official `train.csv` does not contain that column. Treat the live CSV schema as authoritative and avoid hard-coding the documented field.
4. **Public provenance summaries have changed.** One RSNA page says the source collection spans 16 institutions, while a later [RSNA member update](https://www.rsna.org/news/announcements/member-updates) says 19. The stable claims are that the data are international, multisite, and multilingual; use Kaggle's live files for implementation decisions.
5. **The public test is not representative in size.** Local dry runs see only three studies; the scored notebook receives about 1,300. Memory, runtime, batching, and failure handling need to be designed for that hidden scale.

## Verification provenance

The page text was retrieved through Kaggle's first-party competition API exposed by the CLI:

```bash
rtk kaggle competitions list -s "RSNA Knee Abnormality Detection" -v
rtk kaggle competitions pages rsna-knee-abnormality-detection --page-name Description --content
rtk kaggle competitions pages rsna-knee-abnormality-detection --page-name Evaluation --content
rtk kaggle competitions pages rsna-knee-abnormality-detection --page-name data-description --content
```

Only `train.csv`, `train_series.csv`, `test.csv`, `test_series.csv`, and `sample_submission.csv` were downloaded for row/schema checks. Aggregate counts came from Kaggle SDK calls to `GetCompetitionDataFilesSummary` and `ListDataTreeFiles`. The archive transfer size came from the headers returned by `DownloadDataFiles`; the response was closed immediately, without downloading the archive.

