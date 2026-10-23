"""Decoder comparison on the EXISTING baseline checkpoints (inference only; resumable).

Evaluates every decoder on the same network output: free, best-pair (strong), mask-free geometric
(consist), constrained (cons, with landmark-ellipse fallback) and strict constrained (cons_strict),
plus the plausibility-gate sensitivity variants (+-25 %, amax = 0.45).
Needs <out>/checkpoints/ from train_lodo_mc.py. Writes <out>/results/decoders/<Anatomy>_<fold>_s<seed>.csv
"""
import os

import torch
from _common import base_parser, parse

from fetalbio import config as C
from fetalbio.data import DS, get_folds
from fetalbio.evaluate import evaluate_all
from fetalbio.model import Net
from fetalbio.reference_decoders import gate_variants
from fetalbio.utils import ckpt_path, results_dir

p = base_parser(__doc__)
p.add_argument("--anatomies", nargs="+", default=list(C.ANATOMIES), choices=list(C.ANATOMIES))
a = parse(p)
D = os.path.join(results_dir(a.out), "decoders")
os.makedirs(D, exist_ok=True)
GV = gate_variants()
for anat in a.anatomies:
    c = C.CFG[anat]
    nc = len(c["meas"]) + int(c["aux"])
    for fold, tr, te in get_folds(anat)[1]:
        dte = DS(te, anat, False)
        for seed in a.seeds:
            out = f"{D}/{anat}_{fold}_s{seed}.csv"
            if os.path.exists(out):
                continue
            ck = ckpt_path(a.out, anat, fold, "baseline", seed)
            if not os.path.exists(ck):
                print("MISSING checkpoint", ck)
                continue
            m = Net(nc).to(C.DEVICE)
            m.load_state_dict(torch.load(ck, map_location=C.DEVICE))
            r = evaluate_all(m, dte, True, GV if c["aux"] else None, C.TTA)
            r.assign(anatomy=anat, fold=fold, seed=seed).to_csv(out, index=False)
            print("done", out, len(r))
            del m
            torch.cuda.empty_cache()
