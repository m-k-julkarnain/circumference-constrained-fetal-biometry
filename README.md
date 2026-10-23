# Circumference-constrained landmark decoding for fetal ultrasound biometry

Code for the paper *"Circumference-Constrained Landmark Decoding for Domain-Robust Fetal
Ultrasound Biometry: A Multicentre Leave-One-Domain-Out Study"*.

A single ResNet34 U-Net predicts endpoint heatmaps plus an auxiliary filled-ellipse mask (built from the
annotated diameters, no segmentation labels needed). At inference an ellipse is fitted to the predicted
mask, checked by a plausibility gate (size, axis ratio, position), and the endpoints are snapped to the
ellipse boundary; implausible fits fall back to free decoding. The **same trained model** is evaluated with
both decoders, so any difference comes from decoding alone.

## Data
Multicentre fetal biometry benchmark (Di Vece et al., *Sci. Rep.* 16:17405, 2026):
images and annotations at <https://doi.org/10.5522/04/30819911>, benchmark code at
<https://github.com/surgical-vision/Multicentre-Fetal-Biometry>. Download it and point `--data-root`
(or the `FETALBIO_DATA` environment variable) at the folder that contains `annotations/` and `images/`.

## Install
```bash
pip install -r requirements.txt        # tested with Python 3.12 and PyTorch 2.10; a CUDA GPU is recommended for training
```

## Repository layout
```
circumference-constrained-fetal-biometry/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/fetalbio/
│   ├── config.py        constants (image size, heatmap stride, optimisation, folds per anatomy)
│   ├── data.py          annotation loading, ellipse geometry, augmentation, heatmap/mask targets, LODO folds
│   ├── model.py         ResNet34 U-Net and loss
│   ├── decoding.py      free decoding and circumference-constrained decoding (+ plausibility gate)
│   ├── reference_decoders.py  best-pair and mask-free geometric decoders, parameterised gate and its variants
│   ├── evaluate.py      inference with flip-TTA and per-image metrics; evaluate_all = every decoder on one output
│   ├── train.py         training loop (shared by every experiment; options mask_head, frac, select)
│   ├── analysis.py      seed-averaged tables, Wilcoxon/Holm, GA error, cluster bootstrap
│   └── utils.py
├── scripts/
│   ├── train_lodo_mc.py             main experiment: 3 anatomies x (LODO folds + MC) x seeds
│   ├── merge_per_seed.py            merge per-seed CSVs into <Anatomy>_per_image.csv
│   ├── export_free_med.py           free-decoding MED per image (Table 4b)
│   ├── analyze.py                   main-result tables -> outputs/results/tables/
│   ├── run_decoder_comparison.py    all decoders + gate sensitivity on the saved baseline checkpoints
│   ├── run_factorial_size_shift.py  2x2 (augmentation x mask head) and training-size x shift studies
│   ├── analyze_extended.py          decoder, gate, QC, tertile, paired-CI, 2x2 and size-shift tables
│   ├── audit_data.py                annotation geometry and subject-ID audits
│   ├── run_aug_ablation.py          Experiment 1: augmentation ablation (Table 6)
│   ├── run_head_single_source.py    Experiment 2: single-source head transfers (Table 7)
│   ├── regenerate_from_checkpoint.py  rebuild per-image CSVs from checkpoints (inference only)
│   └── make_qualitative_figure.py   qualitative decoding figure
└── results/             (optional: final per-image CSVs to publish; see results/README.md)
```

## Reproducing the paper
Run from the repository root. Every training script is resumable (finished runs are skipped).
```bash
D=/path/to/FetalBiometry-Multicentre-Landmarks-2026

# 1. main experiment (one process per seed is convenient on free GPU quotas)
python scripts/train_lodo_mc.py --data-root $D --seeds 0
python scripts/train_lodo_mc.py --data-root $D --seeds 1
python scripts/train_lodo_mc.py --data-root $D --seeds 2

# 2. merge seeds, free-decoding MED, tables
python scripts/merge_per_seed.py
python scripts/export_free_med.py --data-root $D
python scripts/analyze.py

# 3. additional experiments
python scripts/run_aug_ablation.py        --data-root $D
python scripts/run_head_single_source.py  --data-root $D

# 4. decoder comparison, supplementary studies and extended tables
python scripts/run_decoder_comparison.py   --data-root $D
python scripts/run_factorial_size_shift.py --data-root $D      # ~ 5 GPU-hours on a T4-class GPU for Abdomen + Head
python scripts/analyze_extended.py
python scripts/audit_data.py               --data-root $D

# 5. qualitative figure
python scripts/make_qualitative_figure.py --data-root $D
```
Outputs: `outputs/checkpoints/<Anatomy>/<fold>/baseline_s<seed>/best.pth` and
`outputs/results/*.csv`; tables are written to `outputs/results/tables/`.

On Kaggle: add the dataset, enable the GPU, `!git clone <repo-url> && cd <repo> && pip install -r requirements.txt`,
then run the commands above with `--data-root` set to the dataset path and `--out /kaggle/working/outputs`.

## Manuscript table -> script
Table numbers are those of the submitted manuscript (check against the final PDF); the `analyze.py` file names keep the original numbering
(`table4_...`, `table9_...`), so use this map.

| Manuscript table | Produced by | Output file |
|---|---|---|
| 1 Source domains and folds | `audit_data.py`, `data.py` | `geometry_audit.csv`, `subject_id_audit.csv` |
| 2 Decoder comparison (free / best-pair / mask-free / constrained) | `run_decoder_comparison.py` -> `analyze_extended.py` | `decoders_nme.csv`, `paired_ci.csv` |
| 3 Training size x shift | `run_factorial_size_shift.py` -> `analyze_extended.py` | `size_shift.csv` |
| 4 Augmentation x mask head x decoder | `run_factorial_size_shift.py` -> `analyze_extended.py` | `factorial_2x2.csv` |
| 5 Augmentation ablation | `run_aug_ablation.py` -> `analyze.py` | `<Anatomy>_augablation_seed*.csv` |
| 6 MED by decoder | `export_free_med.py`, `analyze_extended.py` | `table4b_med_dice.csv`, `med_sdr.csv` |
| 7 Official multicentre split vs benchmark | `train_lodo_mc.py` -> `analyze.py` | `table4_constrained_vs_free.csv` |
| 8 Single-source transfer | `run_head_single_source.py` | `Head_xdomain.csv` |
| 9 Gestational-age error | `analyze.py`, `analyze_extended.py` | `table9_ga_days.csv`, `circ_ga.csv` |
| 10 Inter-observer band | `analyze.py` | `table10_observer_band.csv` |
| 11 Failure analysis | `analyze.py`, `analyze_extended.py` (`qc_fallback`) | `table12_failure.csv`, `qc_fallback.csv` |
| 12 Failure by tertile | `analyze_extended.py` | `failure_tertile.csv` |
| 13 Bootstrap CIs | `analyze.py` | `table13_bootstrap.csv` |
| 14 Seed variance | `analyze.py` | `table14_seed_variance.csv` |
| Gate sensitivity (text) | `analyze_extended.py` | `gate_sensitivity.csv`, `gate_mask_path_share_pct.csv` |

## Protocol summary
| | |
|---|---|
| Folds | Head: LODO-FP / HC18 / UCL + MC; Abdomen and Femur: LODO-FP / UCL + MC |
| Validation | subject-grouped 10% of the training pool (`GroupShuffleSplit`), per seed |
| Seeds | 0, 1, 2; metrics are averaged over seeds **per image** before any statistic |
| Input / targets | 256x256 letterbox + CLAHE(3.0, 8x8); 128x128 heatmaps (sigma 2.5) + ellipse mask (head, abdomen) |
| Loss | weighted MSE (1+10h)x100 + 2x(BCE + soft Dice) |
| Optimiser | AdamW (wd 1e-4), OneCycle peak 3e-4, batch 16, 40 epochs, clip 5.0, AMP |
| Checkpoint | lowest validation median NME after a 10-epoch warm-up; flip TTA at test time |
| Statistics | Wilcoxon signed-rank + Holm; image-level and subject-level cluster bootstrap |

## Notes
* Circumference uses Ramanujan's first approximation; gestational age uses the Hadlock (1984) HC-only formula.
  Both are interpretability aids, not validated clinical tools.
* Subject identifiers for the cluster bootstrap are the image-name prefix when `SubjectID` is missing.
* This is a retrospective methodological study, not a clinical validation.

## Citation
```
@article{julkarnain2026circumference,
  title   = {Circumference-Constrained Landmark Decoding for Domain-Robust Fetal Ultrasound Biometry:
             A Multicentre Leave-One-Domain-Out Study},
  author  = {Julkarnain, M K and Jibon, Ferdaus Anam},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Under review. Code archive: Zenodo DOI to be added on release}
}
```
