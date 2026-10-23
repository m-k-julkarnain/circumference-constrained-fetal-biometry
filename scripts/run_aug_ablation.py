"""Experiment 1 -- augmentation ablation (Abdomen and Femur, LODO folds, baseline architecture).

Trains with and without augmentation (CLAHE and letterboxing kept in both arms) and appends one
row per (anatomy, fold, augment, seed) to <out>/results/<Anatomy>_augablation_seed<seed>.csv.
"""
import os

import pandas as pd
import torch
from _common import base_parser, parse

from fetalbio import config as C
from fetalbio.data import DS, get_folds
from fetalbio.evaluate import evaluate
from fetalbio.train import train_model
from fetalbio.utils import append_row, fmt, results_dir

PLAN = [("Abdomen", "LODO_FP"), ("Abdomen", "LODO_UCL"), ("Femur", "LODO_FP"), ("Femur", "LODO_UCL")]
a = parse(base_parser(__doc__))
R = results_dir(a.out)

for anat in sorted({x for x, _ in PLAN}):
    dd, folds = get_folds(anat)
    FD = {f: (tr, te) for f, tr, te in folds}
    for seed in a.seeds:
        csv = f"{R}/{anat}_augablation_seed{seed}.csv"
        for a_, fold in PLAN:
            if a_ != anat:
                continue
            tr, te = FD[fold]
            for augment in (True, False):
                RR = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
                if len(RR) and ((RR.fold == fold) & (RR.augment == augment)).any():
                    print("skip (done)", anat, fold, "augment=", augment, "seed", seed)
                    continue
                print(f"\n>>> {anat} | {fold} | augment={augment} | seed {seed}")
                model, be, minutes, _ = train_model(tr, anat, seed, augment=augment)
                res = evaluate(model, DS(te, anat, False), C.TTA)
                rv = res[res.valid == 1]
                row = dict(anatomy=anat, fold=fold, augment=augment, seed=seed, n_test=len(te),
                           best_epoch=be, nme_valid=rv.nme.mean(), nme_med=rv.nme.median(),
                           circ_pct=rv.circ_err_pct.mean() if "circ_err_pct" in rv else float("nan"),
                           train_min=minutes)
                print(fmt(row))
                append_row(csv, row)
                del model
                torch.cuda.empty_cache()
print("done")
