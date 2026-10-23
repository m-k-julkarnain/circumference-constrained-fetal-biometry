"""Dataset loading, ellipse geometry, augmentation and target construction."""
import math
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C
from .config import CFG, HS, S, SIGMA, STRIDE

cv2.setNumThreads(0)

_IDX = {}


def resolve(d, n):
    """Resolve an image file name inside folder `d`, ignoring the extension if needed."""
    p = os.path.join(d, n)
    if os.path.exists(p):
        return p
    if d not in _IDX:
        _IDX[d] = {os.path.splitext(f)[0]: os.path.join(d, f) for f in os.listdir(d)}
    return _IDX[d].get(os.path.splitext(n)[0])


def load_df(dom, anat, split=None):
    """Load annotations for one domain (FP / HC18 / UCL / MC) and anatomy."""
    base = C.DATA_ROOT
    meas = CFG[anat]["meas"]
    cols = [f"{m}_{i}_{c}" for m in meas for i in (1, 2) for c in "xy"]
    if dom == "MC":
        ad, idr = f"{base}/annotations/MULTICENTRE", f"{base}/images/MULTICENTRE/{anat}"
    else:
        ad, idr = f"{base}/annotations/{dom}", f"{base}/images/{dom}/{anat}"
    fn = anat + (f"_{split}" if split else "")
    df = pd.read_csv(f"{ad}/{fn}.csv")
    if "px_to_mm_rate" not in df:
        df["px_to_mm_rate"] = np.nan
    if dom == "UCL":
        p = f"{ad}/px_to_mm/{fn}.csv"
        if os.path.exists(p):
            r = pd.read_csv(p).dropna(subset=["px_to_mm_rate"]).drop_duplicates("image_name")
            df["px_to_mm_rate"] = df["px_to_mm_rate"].fillna(
                df["image_name"].map(dict(zip(r.image_name, r.px_to_mm_rate))))
    df["path"] = df.image_name.map(lambda n: resolve(idr, n))
    df = df.dropna(subset=cols + ["path"]).reset_index(drop=True)
    sid = df["SubjectID"] if "SubjectID" in df else pd.Series(np.nan, index=df.index)
    sid = sid.where(sid.notna() & (sid.astype(str) != ""), df.image_name.str.split("_").str[0])
    df["group"] = dom + "_" + sid.astype(str)
    df["domain"] = dom
    return df[["image_name", "path", "px_to_mm_rate", "group", "domain"] + cols]


def ellipse_from_pts(P):
    """Ellipse (cx, cy, a, b, angle_deg) from the two measurement segments.

    The first diameter defines the orientation and semi-axis a; the second defines b.
    The two diameters are treated as perpendicular principal axes.
    """
    c = P.reshape(-1, 2).mean(0)
    v = P[0, 1] - P[0, 0]
    a = np.linalg.norm(P[0, 0] - P[0, 1]) / 2
    b = np.linalg.norm(P[1, 0] - P[1, 1]) / 2 if len(P) > 1 else a
    ang = math.degrees(math.atan2(v[1], v[0]))
    return (c[0], c[1], max(a, 1e-3), max(b, 1e-3), ang)


def ellipse_boundary(cx, cy, a, b, angle_deg, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    r = math.radians(angle_deg)
    ex, ey = a * np.cos(t), b * np.sin(t)
    return np.stack([cx + ex * math.cos(r) - ey * math.sin(r),
                     cy + ex * math.sin(r) + ey * math.cos(r)], 1)


class DS(Dataset):
    """Landmark dataset. Set `.train=False` on a built instance to disable augmentation while
    keeping the same subject split (used by the augmentation ablation)."""

    def __init__(s, df, anat, train):
        c = CFG[anat]
        s.df = df.reset_index(drop=True)
        s.meas = c["meas"]
        s.K = len(s.meas)
        s.aux = c["aux"]
        s.train = train
        s.cols = [f"{m}_{i}_{a}" for m in s.meas for i in (1, 2) for a in "xy"]

    def __len__(s):
        return len(s.df)

    def __getitem__(s, i):
        r = s.df.iloc[i]
        img = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        sc = S / max(h, w)
        M = np.array([[sc, 0, (S - w * sc) / 2], [0, sc, (S - h * sc) / 2], [0, 0, 1.]])  # letterbox
        if s.train:   # geometric augmentation
            A = np.vstack([cv2.getRotationMatrix2D((S / 2, S / 2), random.uniform(-25, 25),
                                                   random.uniform(.85, 1.15)), [0, 0, 1]])
            A[0, 2] += random.uniform(-.08, .08) * S
            A[1, 2] += random.uniform(-.08, .08) * S
            if random.random() < .5:
                A = np.array([[-1, 0, S - 1], [0, 1, 0], [0, 0, 1.]]) @ A
            M = A @ M
        im = cv2.warpAffine(img, M[:2], (S, S), flags=cv2.INTER_LINEAR)
        if s.train:   # photometric + ultrasound-specific augmentation
            f = (im / 255.).astype(np.float32) ** random.uniform(.7, 1.4)
            f = f * random.uniform(.75, 1.25) + random.uniform(-.1, .1)
            if random.random() < .3:
                f = cv2.GaussianBlur(f, (0, 0), random.uniform(.5, 1.5))
            if random.random() < .3:
                f = f * np.random.gamma(25, 1 / 25, f.shape).astype(np.float32)   # speckle
            im = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        im = cv2.createCLAHE(3.0, (8, 8)).apply(im)
        P = np.asarray(r[s.cols], dtype=np.float64).reshape(-1, 2)
        P = (np.c_[P, np.ones(len(P))] @ M.T)[:, :2].reshape(s.K, 2, 2)
        Q = (P + .5) / STRIDE - .5
        yy, xx = np.mgrid[0:HS, 0:HS]
        hm = np.zeros((s.K + int(s.aux), HS, HS), np.float32)
        for k in range(s.K):
            for j in range(2):
                hm[k] = np.maximum(hm[k], np.exp(-((xx - Q[k, j, 0]) ** 2 + (yy - Q[k, j, 1]) ** 2)
                                                 / (2 * SIGMA ** 2)))
        if s.aux:   # auxiliary filled-ellipse mask built from the annotated diameters
            cx, cy, a, b, ang = ellipse_from_pts(Q)
            m = np.zeros((HS, HS), np.uint8)
            cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                        (max(int(round(a)), 1), max(int(round(b)), 1)), ang, 0, 360, 1, -1)
            hm[s.K] = m
        rate = float(r.px_to_mm_rate) if pd.notna(r.px_to_mm_rate) else float("nan")
        return dict(img=torch.from_numpy(np.repeat(im[None], 3, 0) / 255.).float(),
                    hm=torch.from_numpy(hm), pts=torch.from_numpy(P).float(),
                    sc=torch.tensor(sc, dtype=torch.float32),
                    rate=torch.tensor(rate, dtype=torch.float32), i=torch.tensor(i))


def get_fold(anat, fold):
    """Return (train_df, test_df) for LODO_<d>, MC, or IND_<d> (in-domain: train and test from the
    same source, 80/20 subject-grouped split, random_state=123). Used by the supplementary studies."""
    if fold.startswith("IND_"):
        from sklearn.model_selection import GroupShuffleSplit
        d = load_df(fold[4:], anat)
        a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=123).split(d, groups=d.group))
        return d.iloc[a].reset_index(drop=True), d.iloc[b].reset_index(drop=True)
    return {f: (t, e) for f, t, e in get_folds(anat)[1]}[fold]


def get_folds(anat):
    """Leave-one-domain-out folds plus the official multicentre (MC) split.

    Returns (per-domain dataframes, [(fold_name, train_df, test_df), ...]).
    """
    c = CFG[anat]
    dd = {d: load_df(d, anat) for d in c["doms"]}
    folds = [(f"LODO_{d}", pd.concat([dd[x] for x in c["doms"] if x != d], ignore_index=True), dd[d])
             for d in c["doms"]]
    folds.append(("MC", load_df("MC", anat, "Train"), load_df("MC", anat, "Test")))
    return dd, folds
