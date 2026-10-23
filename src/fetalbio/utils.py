"""Small shared helpers: seeding, output layout, resumable CSV append."""
import os
import random

import numpy as np
import pandas as pd
import torch


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def results_dir(out):
    p = os.path.join(out, "results")
    os.makedirs(p, exist_ok=True)
    return p


def ckpt_path(out, anat, fold, tag, seed):
    """<out>/checkpoints/<Anatomy>/<fold>/<tag>_s<seed>/best.pth"""
    p = os.path.join(out, "checkpoints", anat, fold, f"{tag}_s{seed}")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, "best.pth")


def append_row(csv_path, row):
    """Append one result row to a CSV (creating it if needed). Used for resumable experiments."""
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


def fmt(row):
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
