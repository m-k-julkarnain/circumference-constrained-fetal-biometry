"""Seed-averaged analysis of the per-image result files (no training, no GPU).

Rules used everywhere (paper, Section 5):
  * baseline variant, valid == 1 images only;
  * metrics are averaged over the three seeds PER IMAGE before any summary or test;
  * mm-based metrics (SDR, MED, failure rate, circumference, GA) use only images with a
    pixel-to-millimetre scale.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from .config import ANATOMIES, FAIL_MM, INTEROBS_HI, INTEROBS_LO, SDR_THRESH_MM


def hadlock_ga_weeks(hc_mm):
    """Hadlock et al. 1984 (Radiology 152:497-501), HC-only: GA[wk] = 8.96 + 0.54 HC + 0.0003 HC^3, HC in cm."""
    hc_cm = hc_mm / 10.0
    return 8.96 + 0.54 * hc_cm + 0.0003 * hc_cm ** 3


def nm(fold):
    return fold.replace("_", "-")


def load_raw(results_dir, anat):
    """Baseline, valid rows of <anat>_per_image.csv with seeds asserted complete."""
    d = pd.read_csv(os.path.join(results_dir, f"{anat}_per_image.csv"))
    d = d[(d.variant == "baseline") & (d.valid == 1)].copy()
    assert d.groupby("fold").seed.nunique().min() == 3, f"{anat}: every fold needs 3 seeds"
    d["mean_mm_err"] = d[[c for c in d.columns if c.startswith("med_mm_")]].mean(axis=1)
    return d


def seed_average(df):
    """Average numeric columns over seeds per (fold, image); keep the modal circumference source."""
    num = df.groupby(["fold", "image_name"], as_index=False).mean(numeric_only=True)
    if "circumference_source" in df.columns:
        mode = df.groupby(["fold", "image_name"]).circumference_source.agg(lambda s: s.mode().iat[0])
        num = num.merge(mode.reset_index(), on=["fold", "image_name"], how="left")
    return num


def load_all(results_dir):
    raw = {a: load_raw(results_dir, a) for a in ANATOMIES}
    return raw, {a: seed_average(d) for a, d in raw.items()}


# ---------------------------------------------------------------------------- tables
def constrained_vs_free(SA):
    """Table 4: NME constrained vs free, Wilcoxon signed-rank with Holm correction."""
    rows = []
    for anat, df in SA.items():
        for fold, g in df.groupby("fold"):
            c, f = g.nme.values, g.nme_free.values
            if len(c) > 5 and (c != f).any():
                _, p = wilcoxon(c, f)
                rows.append(dict(anatomy=anat, fold=fold, n=len(g), constrained_nme=round(c.mean(), 4),
                                 free_nme=round(f.mean(), 4),
                                 relative_reduction_pct=round(100 * (f.mean() - c.mean()) / f.mean(), 1), p=p))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = multipletests(t.p, method="holm")[1]
    return t


def table_sdr(SA):
    """Table 8: SDR over images with a mm scale."""
    rows = []
    for anat, d in SA.items():
        for f, g in d.groupby("fold"):
            h = g[g.mean_mm_err.notna()]
            rows.append(dict(anatomy=anat, fold=f, n_all=len(g), n_scale=len(h),
                             **{f"SDR@{t}": round(100 * (h.mean_mm_err <= t).mean(), 1) for t in SDR_THRESH_MM}))
    return pd.DataFrame(rows)


def table_failure(SA):
    """Table 12: failure = seed-averaged endpoint error > 10 mm."""
    rows = []
    for anat, d in SA.items():
        for f, g in d.groupby("fold"):
            h = g[g.mean_mm_err.notna()]
            fl, ok = h[h.mean_mm_err > FAIL_MM], h[h.mean_mm_err <= FAIL_MM]
            row = dict(anatomy=anat, fold=f, n_scale=len(h), n_fail=len(fl),
                       fail_pct=round(100 * len(fl) / len(h), 1),
                       err_fail=round(fl.mean_mm_err.mean(), 2), err_ok=round(ok.mean_mm_err.mean(), 2))
            if "circ_gt_mm" in h:
                row.update(circ_fail=round(fl.circ_gt_mm.mean(), 1), circ_ok=round(ok.circ_gt_mm.mean(), 1))
            rows.append(row)
    return pd.DataFrame(rows)


def table_observer_band(SA):
    """Table 10: fraction of circumference errors within the cited 4.9-11.1% band."""
    rows = []
    for anat in ("Head", "Abdomen"):
        for f, g in SA[anat].groupby("fold"):
            c = g.dropna(subset=["circ_err_pct"])
            rows.append(dict(anatomy=anat, fold=f, n=len(c),
                             le_4_9=round(100 * (c.circ_err_pct <= INTEROBS_LO).mean(), 1),
                             le_11_1=round(100 * (c.circ_err_pct <= INTEROBS_HI).mean(), 1),
                             mean_pct=round(c.circ_err_pct.mean(), 2)))
    return pd.DataFrame(rows)


def table_ga(SA):
    """Table 9: head-derived gestational-age error in days (Hadlock 1984, HC-only)."""
    rows = []
    for f, g in SA["Head"].groupby("fold"):
        c = g.dropna(subset=["circ_gt_mm", "circ_pr_mm"])
        e = 7 * (hadlock_ga_weeks(c.circ_pr_mm) - hadlock_ga_weeks(c.circ_gt_mm)).abs()
        rows.append(dict(fold=f, n=len(c), mean=round(e.mean(), 2), median=round(e.median(), 2),
                         sd=round(e.std(), 2)))
    return pd.DataFrame(rows)


def table_fallback(raw):
    """Table 11: mask path vs fallback, counted per image-seed prediction (head and abdomen)."""
    rows = []
    for anat in ("Head", "Abdomen"):
        for f, g in raw[anat].groupby("fold"):
            mk, fb = g[g.circumference_source == "mask"], g[g.circumference_source != "mask"]
            rows.append(dict(anatomy=anat, fold=f, n_mask=len(mk), n_fallback=len(fb),
                             nme_mask=round(mk.nme.mean(), 4),
                             nme_fallback=round(fb.nme.mean(), 4) if len(fb) else np.nan))
    return pd.DataFrame(rows)


def table_seed_variance(raw):
    """Table 14: mean and SD across the three seeds of the fold-level mean NME."""
    rows = []
    for anat, d in raw.items():
        for f, g in d.groupby("fold"):
            ps = g.groupby("seed").nme.mean()
            rows.append(dict(anatomy=anat, fold=f, n_seeds=len(ps), nme_mean=round(ps.mean(), 4),
                             nme_sd=round(ps.std(), 4), per_seed=list(ps.round(4).values)))
    return pd.DataFrame(rows)


def table_med_dice(SA, free_med_csv=None):
    """Table 4b: MED constrained / free (mm, mm-scale images) and Dice (decoder-independent mask overlap)."""
    rows = []
    fm = None
    if free_med_csv and os.path.exists(free_med_csv):
        fm = pd.read_csv(free_med_csv)
        fm["med_free"] = fm[[c for c in fm.columns if c.startswith("med_free_mm_")]].mean(axis=1)
        fm = fm.groupby(["anatomy", "fold", "image_name"]).med_free.mean().reset_index().dropna()
    for anat, d in SA.items():
        for f, g in d.groupby("fold"):
            h = g[g.mean_mm_err.notna()]
            row = dict(anatomy=anat, fold=f, n_scale=len(h), med_con=round(h.mean_mm_err.mean(), 2))
            if fm is not None:
                x = fm[(fm.anatomy == anat) & (fm.fold == f)].merge(h[["image_name"]], on="image_name")
                row["med_free"] = round(x.med_free.mean(), 2)
            row["dice"] = round(g.dice.mean(), 3) if g.dice.notna().any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- bootstrap
def cluster_bootstrap(SA, n_boot=2000, seed=0):
    """Image-level and subject-level (cluster) 95% percentile bootstrap CIs of the fold-mean NME.

    The cluster bootstrap resamples subjects with replacement and keeps all images of each drawn
    subject; the statistic is the image-weighted ratio estimator  sum(S_b) / sum(N_b)  (paper Eq. 25),
    so the point estimate lies inside the bootstrap distribution. Subject id = image-name prefix.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for anat, d in SA.items():
        d = d.copy()
        d["subj"] = d.image_name.astype(str).str.split("_").str[0]
        for f, g in d.groupby("fold"):
            e = g.nme.values
            s = g.groupby("subj").nme.agg(["sum", "count"])
            S_, N_, k = s["sum"].values, s["count"].values, len(s)
            idx = rng.integers(0, k, (n_boot, k))
            mu = S_[idx].sum(1) / N_[idx].sum(1)
            im = e[rng.integers(0, len(e), (n_boot, len(e)))].mean(1)
            lo, hi = np.percentile(mu, [2.5, 97.5])
            ilo, ihi = np.percentile(im, [2.5, 97.5])
            m = e.mean()
            wi, ws = ihi - ilo, hi - lo
            rows.append(dict(anatomy=anat, fold=f, n_img=len(e), n_subj=k, nme_mean=round(m, 4),
                             img_lo=round(ilo, 4), img_hi=round(ihi, 4), subj_lo=round(lo, 4), subj_hi=round(hi, 4),
                             subject_ci="wider" if ws > 1.02 * wi else ("narrower" if ws < 0.98 * wi else "equal"),
                             mean_inside_subject_ci=bool(lo <= m <= hi)))
    return pd.DataFrame(rows)


def latex_rows(df, cols, fmts=None):
    """Helper that prints a dataframe as LaTeX table rows (`a & b & c \\\\`)."""
    out = []
    for _, r in df.iterrows():
        out.append(" & ".join(str(r[c]) for c in cols) + r" \\")
    return "\n".join(out)
