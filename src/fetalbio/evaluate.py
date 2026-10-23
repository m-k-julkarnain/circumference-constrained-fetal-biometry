"""Inference and per-image metrics."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import DEVICE, STRIDE
from .decoding import decode_free, guided_decode, robust_ellipse


def ram(a, b):
    """Ramanujan's first approximation to the circumference of an ellipse (semi-axes a, b)."""
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


@torch.no_grad()
def infer(model, imgs, tta=True):
    o = model(imgs).float()
    if tta:
        o = (o + model(imgs.flip(3)).float().flip(3)) / 2
    return o.cpu()


@torch.no_grad()
def evaluate(model, ds, tta=True, bs=32):
    """Per-image metrics for constrained (default) and free decoding.

    Columns: nme / nme_free, circumference_source, dice, validity flag, per-measurement NME,
    MED in mm (med_mm_*), length error, and (head/abdomen) circumference ground truth,
    prediction and errors.
    """
    model.eval()
    rows = []
    K = ds.K
    for b in DataLoader(ds, bs, shuffle=False, num_workers=2):
        o = infer(model, b["img"].to(DEVICE), tta)
        G = b["pts"].numpy()
        mm = (b["rate"] / b["sc"]).numpy()
        n = len(G)
        Pf = decode_free(o[:, :K])
        Pg = Pf.copy()
        src = ["free"] * n
        dice = np.full(n, np.nan)
        if ds.aux:
            pm = torch.sigmoid(o[:, K]).numpy()
            gm = b["hm"][:, K].numpy()
            hn = o[:, :K].numpy()
            for j in range(n):
                el, s_ = robust_ellipse(pm[j], Pf[j])
                src[j] = s_
                if el is not None:
                    Pg[j] = guided_decode(hn[j], K, el)
                p_, g_ = pm[j] > .5, gm[j] > .5
                dice[j] = 2 * (p_ & g_).sum() / (p_.sum() + g_.sum() + 1e-6)
        gl = np.linalg.norm(G[:, :, 0] - G[:, :, 1], axis=-1)

        def met(P):   # min over endpoint order, summed over the two endpoints
            d = np.minimum(np.linalg.norm(P - G, axis=-1).sum(-1),
                           np.linalg.norm(P - G[:, :, ::-1], axis=-1).sum(-1))
            return d, np.linalg.norm(P[:, :, 0] - P[:, :, 1], axis=-1)

        d, pl = met(Pg)
        d_f, plf = met(Pf)
        nme = d / (2 * gl + 1e-6)
        nmef = d_f / (2 * gl + 1e-6)
        for j in range(n):
            lo, hi = gl[j].min(), gl[j].max()
            r = {"i": int(b["i"][j]), "nme": float(nme[j].mean()), "nme_free": float(nmef[j].mean()),
                 "circumference_source": src[j], "dice": dice[j],
                 "gt_min_len": lo, "gt_ratio": hi / (lo + 1e-6),
                 "valid": int(lo >= (20 if K == 2 else 5) and hi / (lo + 1e-6) <= 2.8)}
            for k, m in enumerate(ds.meas):
                r[f"nme_{m}"] = nme[j, k]
                r[f"med_mm_{m}"] = d[j, k] / 2 * mm[j]
                r[f"len_err_mm_{m}"] = abs(pl[j, k] - gl[j, k]) * mm[j]
            if K == 2:
                cg = ram(gl[j, 0] / 2, gl[j, 1] / 2) * mm[j]
                cp = ram(pl[j, 0] / 2, pl[j, 1] / 2) * mm[j]
                cf = ram(plf[j, 0] / 2, plf[j, 1] / 2) * mm[j]
                r.update(circ_gt_mm=cg, circ_pr_mm=cp, circ_err_mm=abs(cg - cp),
                         circ_err_pct=abs(cg - cp) / cg * 100,
                         circ_free_err_pct=abs(cg - cf) / cg * 100)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["image_name"] = ds.df.image_name.iloc[df.i].values
    return df


@torch.no_grad()
def free_med(model, ds, tta=True, bs=32):
    """Free-decoding MED in mm per image and measurement (used for Table 4b)."""
    model.eval()
    rows = []
    K = ds.K
    for b in DataLoader(ds, bs, shuffle=False, num_workers=2):
        o = infer(model, b["img"].to(DEVICE), tta)
        Pf = decode_free(o[:, :K])
        G = b["pts"].numpy()
        mm = (b["rate"] / b["sc"]).numpy()
        d_f = np.minimum(np.linalg.norm(Pf - G, axis=-1).sum(-1),
                         np.linalg.norm(Pf - G[:, :, ::-1], axis=-1).sum(-1))
        for j in range(len(Pf)):
            r = {"i": int(b["i"][j])}
            for k, m in enumerate(ds.meas):
                r[f"med_free_mm_{m}"] = d_f[j, k] / 2 * mm[j]
            rows.append(r)
    df = pd.DataFrame(rows)
    df["image_name"] = ds.df.image_name.iloc[df.i].values
    return df


def _score(P, G, mm, K, pref):
    gl = np.linalg.norm(G[:, :, 0] - G[:, :, 1], axis=-1)
    d = np.minimum(np.linalg.norm(P - G, axis=-1).sum(-1), np.linalg.norm(P - G[:, :, ::-1], axis=-1).sum(-1))
    o = {f"{pref}nme": (d / (2 * gl + 1e-6)).mean(1), f"{pref}med": (d / 2).mean(1) * mm}
    if K == 2:
        pl = np.linalg.norm(P[:, :, 0] - P[:, :, 1], axis=-1)
        cg = ram(gl[:, 0] / 2, gl[:, 1] / 2) * mm
        cp = ram(pl[:, 0] / 2, pl[:, 1] / 2) * mm
        o["circ_gt_mm"] = cg
        o[f"{pref}circ_mm"] = cp
        o[f"{pref}circ_pct"] = np.abs(cg - cp) / cg * 100
    return o


@torch.no_grad()
def evaluate_all(model, ds, mask_head=True, gates=None, tta=True, bs=32):
    """Per-image table with every decoder on the same network output.

    Prefixes: free_ | strong_ | consist_ (K=2) | cons_ (constrained, with landmark-ellipse fallback)
    | cons_strict_ (gate fails -> free decoding, the manuscript wording). `gates` is an optional
    {name: gate dict} (see gate_variants); adds gv_<name>_* columns.
    """
    from .data import ellipse_from_pts
    from .decoding import _mask_ellipse_canvas, guided_decode
    from .reference_decoders import decode_consistent, decode_strong, gate_ok

    model.eval()
    K = ds.K
    rows = []
    for b in DataLoader(ds, bs, shuffle=False, num_workers=2):
        o = infer(model, b["img"].to(DEVICE), tta)
        hn = o[:, :K].numpy()
        G = b["pts"].numpy()
        mm = (b["rate"] / b["sc"]).numpy()
        n = len(G)
        D = {"free": decode_free(o[:, :K]), "strong": decode_strong(hn)}
        if K == 2:
            D["consist"] = decode_consistent(hn)
        cols, src = {}, ["none"] * n
        if ds.aux and mask_head:
            pm = torch.sigmoid(o[:, K]).numpy()
            cons, strict = D["free"].copy(), D["free"].copy()
            circ_ell = np.full(n, np.nan)
            raws = [_mask_ellipse_canvas(pm[j]) for j in range(n)]
            for j in range(n):
                if gate_ok(raws[j]):
                    el, src[j] = raws[j], "mask"
                    strict[j] = guided_decode(hn[j], K, el)
                    circ_ell[j] = ram(el[2], el[3]) * mm[j]
                else:
                    e2 = ellipse_from_pts(D["free"][j])
                    el = e2 if gate_ok(e2) else None
                    src[j] = "fallback" if el else "free"
                if el is not None:
                    cons[j] = guided_decode(hn[j], K, el)
            D["cons"], D["cons_strict"] = cons, strict
            cols["circ_ellipse_mm"] = circ_ell
            for name, g in (gates or {}).items():
                Pg = D["free"].copy()
                used_mask = np.zeros(n)
                for j in range(n):
                    el = raws[j] if gate_ok(raws[j], g) else None
                    used_mask[j] = el is not None
                    if el is None:
                        e2 = ellipse_from_pts(D["free"][j])
                        el = e2 if gate_ok(e2, g) else None
                    if el is not None:
                        Pg[j] = guided_decode(hn[j], K, el)
                for kk, v in _score(Pg, G, mm, K, f"gv_{name}_").items():
                    if "gt" not in kk:
                        cols[kk] = v
                cols[f"gv_{name}_mask"] = used_mask
        for nm, P in D.items():
            cols.update(_score(P, G, mm, K, f"{nm}_"))
        gl = np.linalg.norm(G[:, :, 0] - G[:, :, 1], axis=-1)
        lo, hi = gl.min(1), gl.max(1)
        cols.update(valid=((lo >= (20 if K == 2 else 5)) & (hi / (lo + 1e-6) <= 2.8)).astype(int),
                    src=src, i=b["i"].numpy())
        rows.append(pd.DataFrame(cols))
    df = pd.concat(rows, ignore_index=True)
    df["image_name"] = ds.df.image_name.iloc[df.i].values
    df["domain"] = ds.df.domain.iloc[df.i].values
    return df
