# DSDN: dual-head pomegranate fruit health screening and disease typing

Code, data splits and full experimental outputs for the article
**"A Dual-Head Network with Explicit Health Screening for Interpretable Pomegranate Disease Diagnosis"** (submitted to the International Journal of Intelligent Systems).

DSDN uses a shared ImageNet-pretrained EfficientNet-B0 backbone with two projection heads:

- a **healthy head** (z_h) with a binary *healthy / diseased* classifier (screening);
- a **symptom head** (z_s) with a five-class disease classifier (typing).

Training combines classification, binary, feature-reconstruction and orthogonality losses.
At inference the two outputs are combined by an explicit rule: an image is reported as *Healthy*
when P(diseased) < τ; otherwise the disease type is the most probable of the four disease classes.
Each head has its own Grad-CAM++ attribution map. The model has **5,530,211 parameters**.

## Repository contents

| Path | Content |
|---|---|
| `dsdn.py` | Complete pipeline: data cache, splits, model, training (resumable), evaluation, probing, attribution, statistics, figures |
| `dataset_audit.py` | Dataset audit: duplicates, capture time and sessions, shortcut and leakage checks |
| `group_split.py` | Fruit-grouped split (reproduces `splits/split_grouped.json` exactly) |
| `DSDN_colab.ipynb` | Run order on Google Colab (T4) |
| `splits/` | All split files used in the article (image keys `class/filename`) and `fruit_groups.csv` |
| `results/runs/<split>__<config>__s<seed>/` | Per run: `epoch_log.csv` (every loss component per epoch), `metrics.json`, `predictions_val.csv`, `predictions_test.csv` (per-image probabilities, binary P(diseased), per-image FSS, gated prediction, binary/five-class disagreement flag), and where applicable `probe.json`, `scs_per_image.csv` |
| `results/tables/` | All tables reported in the article |
| `results/figures/` | All figures (each panel as a separate file plus a stacked version; PNG and PDF) |
| `results/pilot_report.txt` | End-to-end pipeline check on a T4 |
| `external_eval.py` | Evaluation without retraining on labeled external datasets (and `yolo_crop` for bounding-box datasets) |
| `external_gradcam.py` | Grad-CAM++ maps of both heads and SCS on external images |
| `results/external/` | Per-image outputs, confusion tables and summaries for both external datasets (Table 14); `gradcam/`: SCS per external class (Supplementary Table S1) |
| `.zenodo.json`, `CITATION.cff` | Archive and citation metadata |

Model checkpoints (`best.pth`) are archived separately at Zenodo (DOI to be added).

## Data

The images are **not** redistributed here. Download the *Pomegranate Fruit Diseases Dataset for Deep
Learning Models* (Pakruddin & Hemavathy, Mendeley Data, doi:10.17632/b6s2rkpmvh.1; data article:
Data in Brief 54 (2024) 110284) and place the five class folders in `data/Pomegranate Diseases Dataset/`.
See the dataset page for its licence.

The released split files identify images by `class/filename`, so they work with any copy of the dataset.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10–3.12, PyTorch 2.x and timm 1.x on an NVIDIA T4 (Google Colab).

## Reproducing the article

```bash
# 1. resized image cache (resumable; kept in a persistent folder and restored locally)
python dsdn.py cache --data "data/Pomegranate Diseases Dataset" --cache /tmp/cache256 --cache_drive cache256

# 2. pipeline check (all lines should read PASS)
python dsdn.py pilot --data "data/Pomegranate Diseases Dataset" --cache /tmp/cache256 --out out \
                     --split splits/split_random.json

# 3. one training run (resumes automatically if interrupted)
python dsdn.py train --cache /tmp/cache256 --out out --split splits/split_random.json --config full --seed 42
```

`DSDN_colab.ipynb` runs all 29 trainings and every analysis in order. Each training run takes about
18 min on a T4, with early stopping after 10 epochs without improvement in validation accuracy.

### Configurations (`--config`)

| Config | Description |
|---|---|
| `full` | DSDN (α, δ, β, γ) = (0.5, 0.2, 0.1, 0.2) |
| `no_ortho`, `no_recon` | Orthogonality or reconstruction loss removed |
| `matched_ctrl` | Both heads and the binary loss kept; orthogonality and reconstruction removed (same parameter count as `full`) |
| `no_healthy`, `no_disentangle` | Single (symptom) head, with or without reconstruction |
| `plain_effb0` | EfficientNet-B0 with a single five-class output |
| `bl_*` | Nine ImageNet-pretrained baselines (ResNet-50/101, DenseNet-121, EfficientNet-B0/B3, MobileNetV3-L, ConvNeXt-T, Swin-T, ViT-B/16) |
| `sens_a*`, `sens_b*`, `sens_g*` | Loss-weight sensitivity (one coefficient changed at a time) |

### Where each result comes from

| Article item | Command | Output |
|---|---|---|
| Fig. 2 | `fig2` | `fig2_two_stage.*` |
| Fig. 4, 5, 6, 7, 8; Supplementary Fig. S1 | `figures` | `results/figures/fig4*` … `fig8*`, `figS3_pca_stacked.*` (Fig. S1) |
| Fig. 9 (correct and misclassified examples) | `fig9` | `fig9_correct.*`, `fig9_misclassified.*` |
| Table 7 (parameters, T4 latency) | `benchmark` | `table7_complexity.csv` |
| Table 8 (comparison, bootstrap CI, McNemar, Bonferroni) | `stats` | `table8_comparison.csv` |
| Table 9 (class-wise F1) | `stats` | `table9_classwise_f1.csv` |
| Table 10 (loss-weight sensitivity) | `stats` | `table10_sensitivity.csv` |
| Ablation, matched-capacity control, grouped split | `summarize` | `table_ablation_and_splits.csv` |
| Direct vs gated inference (τ chosen on validation) | `summarize` | `table_screening_gated_vs_direct.csv` |
| Linear probing of z_h, z_s and backbone features | `probe`, `summarize` | `table_latent_probing.csv` |
| Spatial Concentration Score (whole test set) | `scs` | `table_scs_by_class.csv`, `runs/*/scs_per_image.csv` |
| Class-wise F1 and FSS | `summarize` | `table_classwise_f1_fss.csv` |
| Five-fold cross-validation (fixed test set) | `summarize` | `table_cv.csv` |
| Duplicate / capture-burst overlap per split | `dupcheck` | `table_duplicate_audit.csv`, `table_day_by_subset_*.csv` |
| Acquisition sites per class and subset | `farms` | `table_site_by_class.csv`, `table_site_by_subset_*.csv` |
| Table 14 (external evaluation) | `external_eval.py` | `results/external/<dataset>/summary.csv`, `confusion__*.csv`, `per_image__*.csv`; `screening_auroc.csv` |
| Supplementary Fig. S2, Table S1 | `external_gradcam.py` | `figS2*_external_gradcam*.png`, `results/external/gradcam/ext_scs_by_class_*.csv` |

## Main results (test set)

| | Value |
|---|---|
| DSDN accuracy, random split, 3 seeds | 98.95 ± 0.13 % |
| DSDN accuracy, five-fold cross-validation | 98.75 ± 0.54 % |
| DSDN accuracy, fruit-grouped split | 98.57 % |
| Nine baselines, same split and protocol | 98.30–99.61 %; no difference significant after Bonferroni correction |
| Gated vs direct five-class output | 99.00 ± 0.15 % vs 98.95 ± 0.13 % |
| Latency (batch 1, T4) | 8.8 ms |

Linear probing shows that both z_h and z_s retain disease-type information (balanced accuracy 0.99 for
both). The orthogonality term therefore yields near-orthogonal latent vectors (FSS ≈ 0.998) rather than
a separation of task information. All images come from one season and one device, and acquisition day is
partly confounded with class (capture day per image: `splits/fruit_groups.csv`).

## External evaluation (no retraining)

`external_eval.py` applies trained checkpoints to labelled external datasets that are not redistributed here:
the Halabja pomegranate dataset (Mohammed et al. 2025, https://doi.org/10.5281/zenodo.15856012) and
MangoFruitBD (Islam et al. 2026, https://doi.org/10.17632/bhrz29mkmr.2; fruit crops cut from the YOLO boxes
with `external_eval.py yolo_crop`, 20 % margin, EXIF-rotated images skipped). Outputs are in `results/external/`.

| | Pomegranate (n = 2,178) | Mango crops (n = 1,994) |
|---|---|---|
| DSDN screening AUROC, 3 seeds | 0.88 ± 0.04 | 0.85 ± 0.03 |
| DSDN healthy specificity at τ = 0.5 | 0.47 ± 0.22 | 0.60 ± 0.12 |
| DSDN non-healthy sensitivity at τ = 0.5 | 0.97 ± 0.01 | 0.86 ± 0.03 |
| DSDN shared-class recall | Anthracnose 0.35 ± 0.05 | Alternaria 0.04 ± 0.03; Anthracnose 0.04 ± 0.02 |

The screening output keeps a threshold-free separation of healthy and non-healthy fruit but needs a
recalibrated threshold; disease typing does not transfer.

## Notes

- Binary label convention: 0 = diseased, 1 = healthy.
- Grad-CAM++ target layer: `encoder.blocks[6][0].conv_pwl` (shared by both maps).
- FSS = 1 − |cos(z_s, z_h)|, averaged over images.
- Determinism: the training order and augmentation are seeded per epoch, so a resumed run reproduces an uninterrupted run.

## Citation

See `CITATION.cff`.

## Licence

Code: MIT (see `LICENSE`). Result files in `results/` and `splits/`: CC BY 4.0.
