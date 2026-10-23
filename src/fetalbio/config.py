"""Global constants and per-anatomy configuration.

All values are those used for the experiments reported in the paper.
"""
import os

import torch

# ---- image / heatmap geometry -------------------------------------------------
S = 256            # letterboxed input size (pixels)
STRIDE = 2         # input -> heatmap stride
HS = S // STRIDE   # heatmap size (128)
SIGMA = 2.5        # Gaussian sigma on the heatmap grid

# ---- optimisation ---------------------------------------------------------------
EPOCHS = 40
BS = 16
LR = 3e-4          # OneCycle peak learning rate
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
TTA = True         # horizontal-flip test-time augmentation
SEEDS = (0, 1, 2)

# ---- evaluation constants ---------------------------------------------------------
SDR_THRESH_MM = (2, 4, 6, 8, 10)
FAIL_MM = 10.0                 # failure = seed-averaged endpoint error > 10 mm
INTEROBS_LO, INTEROBS_HI = 4.9, 11.1   # cited inter-observer variability band (%)

# ---- anatomies ----------------------------------------------------------------------
ANATOMIES = ("Abdomen", "Femur", "Head")
CFG = {
    "Head":    dict(meas=["bpd", "ofd"],  doms=["FP", "HC18", "UCL"], aux=True),
    "Abdomen": dict(meas=["tad", "apad"], doms=["FP", "UCL"],         aux=True),
    "Femur":   dict(meas=["fl"],          doms=["FP", "UCL"],         aux=False),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset root (folder that contains `annotations/` and `images/`).
# Override with the FETALBIO_DATA environment variable or the --data-root CLI flag.
DATA_ROOT = os.environ.get("FETALBIO_DATA", "data/FetalBiometry-Multicentre-Landmarks-2026")


def set_data_root(path: str) -> None:
    global DATA_ROOT
    DATA_ROOT = path
