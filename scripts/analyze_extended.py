"""Extended analyses from the decoder-comparison and supplementary-run CSVs (no GPU).

Needs  <out>/results/decoders/*.csv  (run_decoder_comparison.py)  and optionally
       <out>/results/runs/*.csv      (run_factorial_size_shift.py).
Writes <out>/results/tables_extended/<name>.csv and prints each table:
  decoders_nme      NME per decoder, reduction vs free, Holm-corrected Wilcoxon       (decoder comparison)
  med_sdr           MED and SDR@2/4/10 mm per decoder
  circ_ga           circumference error, 4.9/11.1 % band, GA error per decoder
  gate_sensitivity  constrained NME and mask-path share under the gate variants
  qc_fallback       fallback flag as failure detector (precision, recall, AUROC)
  failure_tertile   failure rate by head-circumference tertile
  paired_ci         paired Wilcoxon (Holm) + subject-level bootstrap CIs, constrained vs free / strong / consist
  factorial_2x2     augmentation x mask-head ablation
  size_shift        training-size x shift study, plus OLS of gain on log(n_train) + shift
"""
import glob
import os

import numpy as np
import pandas as pd
from _common import base_parser, parse
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from fetalbio.utils import results_dir

p = base_parser(__doc__)
p.add_argument("--n-boot", type=int, default=2000)
a = parse(p)
R = results_dir(a.out)
T = os.path.join(R, "tables_extended")
os.makedirs(T, exist_ok=True)
pd.set_option("display.width", 250, "display.max_columns", 60)


def save(df, name):
    df.to_csv(os.path.join(T, name + ".csv"), index=False)
    print(f"\n=========== {name} ===========")
    print(df.to_string(index=False))


def load(sub):
    fs = sorted(glob.glob(os.path.join(R, sub, "*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def seed_avg(d, keys):
    return d.groupby(keys + ["image_name"], as_index=False).mean(numeric_only=True)


had = lambda mm: 8.96 + 0.54 * (mm / 10) + 0.0003 * (mm / 10) ** 3   # Hadlock head-circumference GA (weeks)

B = load("decoders")
if B.empty:
    raise SystemExit("no decoder CSVs found; run run_decoder_comparison.py first")
B = B[B.valid == 1]
DEC = [c[:-4] for c in B.columns if c.endswith("_nme") and not c.startswith("gv_")]
SB = seed_avg(B, ["anatomy", "fold"])
SB["subj"] = SB.image_name.astype(str).str.split("_").str[0]

# ---- decoders_nme ---------------------------------------------------------------------------
rows = []
for (an, f), g in SB.groupby(["anatomy", "fold"]):
    r = dict(anatomy=an, fold=f, n=len(g), **{d: round(g[f"{d}_nme"].mean(), 4) for d in DEC
                                              if g[f"{d}_nme"].notna().any()})
    for d in ("strong", "consist", "cons", "cons_strict"):
        if d in r and (g[f"{d}_nme"] != g.free_nme).any():
            r[f"red_{d}_%"] = round(100 * (r["free"] - r[d]) / r["free"], 1)
            r[f"p_{d}"] = wilcoxon(g[f"{d}_nme"], g.free_nme)[1]
    rows.append(r)
TA = pd.DataFrame(rows)
for d in ("strong", "consist", "cons", "cons_strict"):
    if f"p_{d}" in TA:
        m = TA[f"p_{d}"].notna()
        TA.loc[m, f"p_{d}"] = multipletests(TA.loc[m, f"p_{d}"], method="holm")[1]
save(TA.round(4), "decoders_nme")

# ---- med_sdr -----------------------------------------------------------------------------------
rows = []
for (an, f), g in SB.groupby(["anatomy", "fold"]):
    for d in DEC:
        if f"{d}_med" not in g or g[f"{d}_med"].isna().all():
            continue
        h = g[g[f"{d}_med"].notna()]
        rows.append(dict(anatomy=an, fold=f, decoder=d, n=len(h), MED_mm=round(h[f"{d}_med"].mean(), 2),
                         **{f"SDR@{t}": round(100 * (h[f"{d}_med"] <= t).mean(), 1) for t in (2, 4, 10)}))
save(pd.DataFrame(rows), "med_sdr")

# ---- circ_ga -----------------------------------------------------------------------------------
rows = []
for (an, f), g in SB.groupby(["anatomy", "fold"]):
    if an == "Femur":
        continue
    for d in DEC:
        if f"{d}_circ_pct" not in g or g[f"{d}_circ_pct"].isna().all():
            continue
        h = g.dropna(subset=[f"{d}_circ_pct"])
        e = h[f"{d}_circ_pct"]
        r = dict(anatomy=an, fold=f, decoder=d, n=len(h), circ_err_mean=round(e.mean(), 2),
                 le_4_9=round(100 * (e <= 4.9).mean(), 1), le_11_1=round(100 * (e <= 11.1).mean(), 1))
        if an == "Head":
            ga = 7 * (had(h[f"{d}_circ_mm"]) - had(h.circ_gt_mm)).abs()
            r.update(ga_mean=round(ga.mean(), 2), ga_median=round(ga.median(), 2), ga_p90=round(ga.quantile(.9), 2))
        rows.append(r)
save(pd.DataFrame(rows), "circ_ga")

# ---- gate_sensitivity --------------------------------------------------------------------------
gv = sorted({c[3:-4] for c in B.columns if c.startswith("gv_") and c.endswith("_nme")})
if gv:
    rows = []
    for (an, f), g in SB[SB.anatomy != "Femur"].groupby(["anatomy", "fold"]):
        r = dict(anatomy=an, fold=f)
        for v in gv:
            r[v] = round(g[f"gv_{v}_nme"].mean(), 4)
        r["max_abs_dev_vs_base_%"] = round(100 * max(abs(r[v] - r["base"]) for v in gv) / r["base"], 1)
        rows.append(r)
    save(pd.DataFrame(rows), "gate_sensitivity")
    sh = (seed_avg(B[B.anatomy != "Femur"], ["anatomy", "fold"]).groupby(["anatomy", "fold"])
          [[f"gv_{v}_mask" for v in gv]].mean().mul(100).round(1).reset_index())
    save(sh, "gate_mask_path_share_pct")

# ---- qc_fallback -------------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score  # noqa: E402

Z = B[B.anatomy != "Femur"].copy()
Z["flag"] = (Z.src != "mask").astype(int)
Z["fail"] = (Z.cons_med > 10).astype(int)
Z = Z.dropna(subset=["cons_med"])
tp = int(((Z.flag == 1) & (Z.fail == 1)).sum())
save(pd.DataFrame([dict(n=len(Z), flagged=int(Z.flag.sum()), failures=int(Z.fail.sum()),
                        precision=round(tp / max(Z.flag.sum(), 1), 3), recall=round(tp / max(Z.fail.sum(), 1), 3),
                        auroc=round(roc_auc_score(Z.fail, Z.flag), 3) if Z.fail.nunique() > 1 else np.nan)]),
     "qc_fallback")

# ---- failure_tertile ---------------------------------------------------------------------------
H = SB[SB.anatomy == "Head"].dropna(subset=["circ_gt_mm", "cons_med"]).copy()
H["tertile"] = pd.qcut(H.circ_gt_mm, 3, labels=["T1 small", "T2", "T3 large"])
save(H.groupby(["fold", "tertile"], observed=True)
     .apply(lambda g: pd.Series(dict(n=len(g), fail_pct=round(100 * (g.cons_med > 10).mean(), 1))),
            include_groups=False).reset_index(), "failure_tertile")

# ---- paired_ci (subject-level cluster bootstrap) -------------------------------------------------
rng = np.random.default_rng(0)
SBn = SB[SB.anatomy != "Femur"]


def boot_red(g, idx, ref, new):
    s = g.groupby("subj")[[ref, new]].sum()
    r_, n_ = s[ref].values, s[new].values
    return 100 * (r_[idx].sum(1) - n_[idx].sum(1)) / r_[idx].sum(1)


rows = []
for (an, f), g in SBn.groupby(["anatomy", "fold"]):
    k = g.subj.nunique()
    idx = rng.integers(0, k, (a.n_boot, k))
    r = dict(anatomy=an, fold=f, n=len(g), subjects=k)
    for ref in ("free", "strong", "consist"):
        pt = 100 * (g[f"{ref}_nme"].mean() - g.cons_nme.mean()) / g[f"{ref}_nme"].mean()
        bs = boot_red(g, idx, f"{ref}_nme", "cons_nme")
        r[f"cons_vs_{ref}_%"] = round(pt, 1)
        r[f"CI_{ref}"] = f"[{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}]"
        r[f"p_{ref}"] = wilcoxon(g.cons_nme, g[f"{ref}_nme"])[1] if (g.cons_nme != g[f"{ref}_nme"]).any() else np.nan
    rows.append(r)
P = pd.DataFrame(rows)
for c in ("p_free", "p_strong", "p_consist"):
    P[c] = multipletests(P[c].fillna(1), method="holm")[1]
save(P.round(4), "paired_ci")

# ---- supplementary runs ------------------------------------------------------------------------
Rn = load("runs")
if Rn.empty:
    print("\n(no runs/ CSVs: run run_factorial_size_shift.py for the 2x2 and size-shift tables)")
else:
    Rn = Rn[Rn.valid == 1]
    RS = seed_avg(Rn, ["anatomy", "fold", "aug", "mask", "frac", "seed"])
    cols = {d: (f"{d}_nme", "mean") for d in DEC if f"{d}_nme" in RS}
    r1 = RS[RS.frac == 1.0].groupby(["anatomy", "fold", "aug", "mask", "seed"]).agg(**cols).reset_index()
    T2 = r1.groupby(["anatomy", "fold", "aug", "mask"]).mean(numeric_only=True).drop(columns="seed").round(4)
    T2["n_seeds"] = r1.groupby(["anatomy", "fold", "aug", "mask"]).seed.nunique()
    save(T2.reset_index(), "factorial_2x2")

    S3 = (RS[(RS.aug == 1) & (RS["mask"] == 1)].groupby(["anatomy", "fold", "frac", "seed"])
          .agg(free=("free_nme", "mean"), cons=("cons_nme", "mean"), strong=("strong_nme", "mean"),
               n_train=("n_train", "max")).reset_index())
    S3["gain_vs_free_%"] = 100 * (S3.free - S3.cons) / S3.free
    S3["gain_vs_strong_%"] = 100 * (S3.strong - S3.cons) / S3.strong
    TH = (S3.groupby(["anatomy", "fold", "frac"])
          .agg(n_train=("n_train", "mean"), free=("free", "mean"), cons=("cons", "mean"),
               gain_free=("gain_vs_free_%", "mean"), gain_strong=("gain_vs_strong_%", "mean"),
               seeds=("seed", "nunique")).round(3).reset_index())
    save(TH, "size_shift")
    import statsmodels.formula.api as smf
    S3["logn"] = np.log(S3.n_train.astype(float))
    S3["shift"] = S3.fold.str.startswith("LODO").astype(int)
    for an, g in S3.groupby("anatomy"):
        if len(g) >= 5:
            print(f"\nOLS {an}: gain_vs_free ~ log(n_train) + shift (shift=1 for LODO, 0 in-domain)")
            print(smf.ols("Q('gain_vs_free_%') ~ logn + shift", g).fit().summary().tables[1])
