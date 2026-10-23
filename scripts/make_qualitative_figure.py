"""Qualitative decoding figure: ground truth, free decoding, constrained decoding and fitted ellipse.

Panels: (a) Abdomen/LODO_FP gain case, (b) Head/LODO_HC18 mask-path case, (c) Head/LODO_HC18 fallback case.
Example images are drawn at random among matching cases (--rng-seed); the chosen images are saved to
<out>/results/qualitative_shortlist.csv so that the exact figure can be reproduced.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from _common import base_parser, parse
from matplotlib.lines import Line2D

from fetalbio import config as C
from fetalbio.data import DS, ellipse_boundary, get_folds
from fetalbio.decoding import decode_free, guided_decode, robust_ellipse
from fetalbio.evaluate import evaluate, infer
from fetalbio.model import Net
from fetalbio.utils import ckpt_path, results_dir

p = base_parser(__doc__)
p.add_argument("--rng-seed", type=int, default=0)
a = parse(p)
R = results_dir(a.out)
np.random.seed(a.rng_seed)
PANELS = [("(a)", "Abdomen", "LODO_FP", "gain"), ("(b)", "Head", "LODO_HC18", "spurious"),
          ("(c)", "Head", "LODO_HC18", "fallback")]


def load_model(anat, fold, seed):
    nc = len(C.CFG[anat]["meas"]) + int(C.CFG[anat]["aux"])
    m = Net(nc).to(C.DEVICE)
    m.load_state_dict(torch.load(ckpt_path(a.out, anat, fold, "baseline", seed), map_location=C.DEVICE))
    return m.eval()


def pick(anat, fold, mode):
    dd, folds = get_folds(anat)
    te = next(e for f, t, e in folds if f == fold)
    ds = DS(te, anat, False)
    cands = []
    for seed in a.seeds:
        m = load_model(anat, fold, seed)
        r = evaluate(m, ds, C.TTA)
        r = r[r.valid == 1].copy()
        r["gain"] = r.nme_free - r.nme
        if mode == "gain":
            s = r[r.gain > 0]
        elif mode == "spurious":
            s = r[(r.circumference_source == "mask") & (r.nme < 0.06)]
        else:
            s = r[r.circumference_source != "mask"]
        cands.extend([(seed, row) for _, row in s.iterrows()])
        del m
        torch.cuda.empty_cache()
    if not cands:
        raise RuntimeError(f"no candidate for {anat} {fold} {mode}")
    seed, row = cands[np.random.randint(len(cands))]
    return load_model(anat, fold, seed), ds, seed, row


fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
recs = []
for a_, (tag, anat, fold, mode) in zip(ax, PANELS):
    m, ds, seed, row = pick(anat, fold, mode)
    K = ds.K
    i = int(row.i)
    b = ds[i]
    o = infer(m, b["img"][None].to(C.DEVICE))[0]
    Pf = decode_free(o[None, :K])[0]
    el, src = robust_ellipse(torch.sigmoid(o[K]).numpy(), Pf)
    Pg = guided_decode(o[:K].numpy(), K, el) if el is not None else Pf
    G = b["pts"].numpy()
    a_.imshow(b["img"][0], cmap="gray")
    a_.axis("off")
    if el is not None:
        e = ellipse_boundary(*el)
        a_.plot(e[:, 0], e[:, 1], "--", c="yellow", lw=1.2)
    for k in range(K):
        a_.plot(G[k, :, 0], G[k, :, 1], "--o", c="lime", lw=1.6, ms=4)
        a_.plot(Pf[k, :, 0], Pf[k, :, 1], "--x", c="orange", lw=1.4, ms=6)
        a_.plot(Pg[k, :, 0], Pg[k, :, 1], "-", c="red", lw=1.6)
    a_.set_title(f"{tag} {anat} {fold}, seed {seed}\nNME free {row.nme_free:.3f} -> constrained {row.nme:.3f} [{src}]",
                 fontsize=9)
    recs.append(dict(panel=tag, anatomy=anat, fold=fold, seed=seed, image_name=row.image_name, nme=row.nme,
                     nme_free=row.nme_free, source=src, gt=json.dumps(np.round(G, 2).tolist()),
                     free=json.dumps(np.round(Pf, 2).tolist()), constrained=json.dumps(np.round(Pg, 2).tolist()),
                     ellipse_cx_cy_a_b_angle=json.dumps(None if el is None else [round(float(v), 2) for v in el])))
    del m
    torch.cuda.empty_cache()
fig.legend(handles=[Line2D([], [], c="lime", ls="--", marker="o"), Line2D([], [], c="orange", ls="--", marker="x"),
                    Line2D([], [], c="red"), Line2D([], [], c="yellow", ls="--")],
           labels=["Ground truth", "Free decoding", "Constrained decoding", "Fitted ellipse"],
           loc="lower center", ncol=4, fontsize=9)
plt.tight_layout(rect=[0, .06, 1, 1])
plt.savefig(f"{R}/qualitative_decoding.png", dpi=300)
pd.DataFrame(recs).to_csv(f"{R}/qualitative_shortlist.csv", index=False)
print(pd.DataFrame(recs)[["panel", "anatomy", "fold", "seed", "image_name", "nme_free", "nme", "source"]].to_string(index=False))
