"""Re-create per-image result rows from saved checkpoints (inference only, no retraining).

Useful if a per-seed CSV was lost. Writes <out>/results/<Anatomy>_per_image_seed<seed>.csv.
"""
import torch
import pandas as pd
from _common import base_parser, parse

from fetalbio import config as C
from fetalbio.data import DS, get_folds
from fetalbio.evaluate import evaluate
from fetalbio.model import Net
from fetalbio.utils import ckpt_path, results_dir

p = base_parser(__doc__)
p.add_argument("--anatomies", nargs="+", default=list(C.ANATOMIES))
a = parse(p)
R = results_dir(a.out)
for anat in a.anatomies:
    c = C.CFG[anat]
    nc = len(c["meas"]) + int(c["aux"])
    dd, folds = get_folds(anat)
    for seed in a.seeds:
        parts = []
        for fold, tr, te in folds:
            model = Net(nc).to(C.DEVICE)
            model.load_state_dict(torch.load(ckpt_path(a.out, anat, fold, "baseline", seed), map_location=C.DEVICE))
            dte = DS(te, anat, False)
            res = evaluate(model, dte, C.TTA)
            parts.append(res.assign(anatomy=anat, fold=fold, variant="baseline", seed=seed,
                                    domain=dte.df.domain.iloc[res.i].values))
            print(f"{anat}/{fold}/seed{seed}: {len(res)} rows, mean NME {res.nme.mean():.4f}")
            del model
            torch.cuda.empty_cache()
        pd.concat(parts, ignore_index=True).to_csv(f"{R}/{anat}_per_image_seed{seed}.csv", index=False)
