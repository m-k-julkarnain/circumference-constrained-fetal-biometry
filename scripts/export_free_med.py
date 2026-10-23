"""Export free-decoding MED (mm) per image for every checkpoint -> <out>/results/free_med_per_image.csv."""
import torch
import pandas as pd
from _common import base_parser, parse

from fetalbio import config as C
from fetalbio.data import DS, get_folds
from fetalbio.evaluate import free_med
from fetalbio.model import Net
from fetalbio.utils import ckpt_path, results_dir

p = base_parser(__doc__)
p.add_argument("--anatomies", nargs="+", default=list(C.ANATOMIES))
a = parse(p)
R = results_dir(a.out)
rows = []
for anat in a.anatomies:
    nc = len(C.CFG[anat]["meas"]) + int(C.CFG[anat]["aux"])
    dd, folds = get_folds(anat)
    for fold, tr, te in folds:
        dte = DS(te, anat, False)
        for seed in a.seeds:
            m = Net(nc).to(C.DEVICE)
            m.load_state_dict(torch.load(ckpt_path(a.out, anat, fold, "baseline", seed), map_location=C.DEVICE))
            rows.append(free_med(m, dte, C.TTA).assign(anatomy=anat, fold=fold, seed=seed))
            del m
            torch.cuda.empty_cache()
            print("done", anat, fold, seed)
pd.concat(rows, ignore_index=True).to_csv(f"{R}/free_med_per_image.csv", index=False)
