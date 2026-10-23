"""Main experiment: train the baseline model on every LODO fold and the official MC split.

For each (anatomy, fold, seed) this saves
  <out>/checkpoints/<Anatomy>/<fold>/baseline_s<seed>/best.pth
and appends the per-image test metrics (constrained and free decoding) to
  <out>/results/<Anatomy>_per_image_seed<seed>.csv
Finished (fold, seed) combinations are skipped, so the script can be restarted.

Example:  python scripts/train_lodo_mc.py --data-root /path/to/data --anatomies Abdomen --seeds 0
"""
import os

import pandas as pd
import torch
from _common import base_parser, parse

from fetalbio import config as C
from fetalbio.data import DS, get_folds
from fetalbio.evaluate import evaluate
from fetalbio.train import train_model
from fetalbio.utils import ckpt_path, fmt, results_dir

p = base_parser(__doc__)
p.add_argument("--anatomies", nargs="+", default=list(C.ANATOMIES), choices=list(C.ANATOMIES))
p.add_argument("--folds", nargs="+", default=None, help="restrict to some folds, e.g. LODO_FP MC")
a = parse(p)
R = results_dir(a.out)

for anat in a.anatomies:
    dd, folds = get_folds(anat)
    for seed in a.seeds:
        csv = f"{R}/{anat}_per_image_seed{seed}.csv"
        done = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
        for fold, tr, te in folds:
            if a.folds and fold not in a.folds:
                continue
            if len(done) and ((done.fold == fold) & (done.variant == "baseline")).any():
                print("skip (done)", anat, fold, seed)
                continue
            print(f"\n>>> {anat} | {fold} | baseline | seed {seed}")
            model, be, minutes, ntr = train_model(tr, anat, seed, augment=True)
            torch.save(model.state_dict(), ckpt_path(a.out, anat, fold, "baseline", seed))
            dte = DS(te, anat, False)
            res = evaluate(model, dte, C.TTA)
            res = res.assign(anatomy=anat, fold=fold, variant="baseline", seed=seed,
                             domain=dte.df.domain.iloc[res.i].values)
            rv = res[res.valid == 1]
            print(fmt(dict(n_test=len(dte), best_epoch=be, nme=rv.nme.mean(), nme_free=rv.nme_free.mean(),
                           minutes=minutes)))
            done = pd.concat([done, res], ignore_index=True)
            done.to_csv(csv, index=False)
            del model
            torch.cuda.empty_cache()
print("done")
