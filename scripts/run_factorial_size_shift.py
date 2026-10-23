"""Supplementary training studies (resumable; one CSV per run).

E2  2x2 augmentation x mask-head ablation (fraction 1.0):
      Abdomen: LODO_FP, LODO_UCL, MC, seeds 0-2;   Head: LODO_FP, LODO_HC18, LODO_UCL, seed 0.
E3  training-size x shift study (full model, fractions 0.25 / 0.5 / 1.0):
      Abdomen: LODO_FP, LODO_UCL, IND_FP, IND_UCL (seeds 0-2); Head: LODO_FP, LODO_HC18, IND_HC18, IND_FP (seed 0).
In all runs the checkpoint is selected on the validation FREE-decoding median NME, so variants with and
without a mask head are selected by the same rule. The validation set is identical for all fractions.
Writes <out>/results/runs/<Anatomy>_<fold>_a<aug>_m<mask>_f<frac%>_s<seed>.csv
"""
import os
import time

from _common import base_parser, parse

from fetalbio import config as C
from fetalbio.data import DS, get_fold
from fetalbio.evaluate import evaluate_all
from fetalbio.train import train_model
from fetalbio.utils import results_dir

p = base_parser(__doc__)
p.add_argument("--only", choices=list(C.ANATOMIES), default=None, help="run one anatomy per session")
p.add_argument("--head-seeds", type=int, nargs="+", default=[0])
a = parse(p)
D = os.path.join(results_dir(a.out), "runs")
os.makedirs(D, exist_ok=True)

abd_seeds, head_seeds = list(a.seeds), list(a.head_seeds)
plan = []
for anat, folds, seeds in (("Abdomen", ["LODO_FP", "LODO_UCL", "MC"], abd_seeds),
                           ("Head", ["LODO_FP", "LODO_HC18", "LODO_UCL"], head_seeds)):
    for fold in folds:
        for aug in (1, 0):
            for mk in (1, 0):
                plan += [(anat, fold, aug, mk, 1.0, s) for s in seeds]
SIZE = {"Abdomen": ["LODO_FP", "LODO_UCL", "IND_FP", "IND_UCL"], "Head": ["LODO_FP", "LODO_HC18", "IND_HC18", "IND_FP"]}
for anat, folds in SIZE.items():
    for fold in folds:
        for fr in (.25, .5, 1.0):
            plan += [(anat, fold, 1, 1, fr, s) for s in (abd_seeds if anat == "Abdomen" else head_seeds)]
plan = list(dict.fromkeys(plan))
plan.sort(key=lambda t: ({"Abdomen": 0, "Head": 1}[t[0]], t[4] != 1.0, t[1], t[2:]))   # cheap first

t0 = time.time()
for anat, fold, aug, mk, fr, seed in plan:
    if a.only and anat != a.only:
        continue
    out = f"{D}/{anat}_{fold}_a{aug}_m{mk}_f{int(fr * 100)}_s{seed}.csv"
    if os.path.exists(out):
        continue
    print(f">>> {os.path.basename(out)}  ({(time.time() - t0) / 60:.0f} min elapsed)")
    tr, te = get_fold(anat, fold)
    model, be, _, ntr = train_model(tr, anat, seed, augment=bool(aug), verbose=False,
                                    mask_head=bool(mk), frac=fr, select="free")
    r = evaluate_all(model, DS(te, anat, False), bool(mk), None, C.TTA)
    r.assign(anatomy=anat, fold=fold, aug=aug, mask=mk, frac=fr, seed=seed, best_epoch=be, n_train=ntr).to_csv(out, index=False)
    del model
