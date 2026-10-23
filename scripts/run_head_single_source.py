"""Experiment 2 -- head single-source cross-domain runs (the benchmark's own protocol).

Each pair trains on ONE domain (full domain, subject-grouped 10% validation) and tests on another:
  FP->UCL, HC18->UCL (protocol-match check), UCL->HC18, UCL->FP, FP->HC18, HC18->FP.
Appends one row per (pair, seed) to <out>/results/Head_xdomain.csv.
"""
import os

import pandas as pd
import torch
from _common import base_parser, parse

from fetalbio import config as C
from fetalbio.data import DS, load_df
from fetalbio.evaluate import evaluate
from fetalbio.train import train_model
from fetalbio.utils import append_row, fmt, results_dir

PLAN = [("FP_to_UCL", "FP", "UCL"), ("HC18_to_UCL", "HC18", "UCL"),
        ("UCL_to_HC18", "UCL", "HC18"), ("UCL_to_FP", "UCL", "FP"),
        ("FP_to_HC18", "FP", "HC18"), ("HC18_to_FP", "HC18", "FP")]
BENCH = {"FP_to_UCL": (0.38, 0.22), "HC18_to_UCL": (0.15, 0.19), "UCL_to_HC18": (0.17, 0.13),
         "UCL_to_FP": (0.10, 0.09), "FP_to_HC18": (0.08, 0.08), "HC18_to_FP": (0.06, 0.06)}
p = base_parser(__doc__)
a = parse(p)
R = results_dir(a.out)
csv = f"{R}/Head_xdomain.csv"

for pair, trd, ted in PLAN:
    tr, te = load_df(trd, "Head"), load_df(ted, "Head")
    for seed in a.seeds:
        XR = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
        if len(XR) and ((XR.pair == pair) & (XR.seed == seed)).any():
            print("skip (done)", pair, seed)
            continue
        print(f"\n>>> Head | {pair} | seed {seed}")
        model, be, minutes, ntr = train_model(tr, "Head", seed, augment=True)
        res = evaluate(model, DS(te, "Head", False), C.TTA)
        rv = res[res.valid == 1]
        res.to_csv(f"{R}/Head_xdomain_perimage_{pair}_seed{seed}.csv", index=False)
        row = dict(pair=pair, train_domain=trd, test_domain=ted, seed=seed, n_train=ntr, n_test=len(te),
                   n_valid=len(rv), best_epoch=be, nme=rv.nme.mean(), nme_bpd=rv.nme_bpd.mean(),
                   nme_ofd=rv.nme_ofd.mean(), nme_free=rv.nme_free.mean(), nme_med=rv.nme.median(),
                   circ_pct=rv.circ_err_pct.mean(), train_min=minutes)
        print(fmt(row))
        append_row(csv, row)
        del model
        torch.cuda.empty_cache()

XR = pd.read_csv(csv).drop_duplicates(["pair", "seed"], keep="last")
g = XR.groupby("pair").agg(n_seeds=("seed", "nunique"), n_test=("n_test", "first"),
                           nme_mean=("nme", "mean"), nme_sd=("nme", lambda x: x.std(ddof=1)),
                           bpd=("nme_bpd", "mean"), ofd=("nme_ofd", "mean"), free=("nme_free", "mean")).round(4)
g["bench_bpd"] = [BENCH[i][0] for i in g.index]
g["bench_ofd"] = [BENCH[i][1] for i in g.index]
print("\n=== Head single-source cross-domain: ours (seed mean) vs benchmark ===")
print(g.to_string())
g.to_csv(f"{R}/Head_xdomain_summary.csv")
