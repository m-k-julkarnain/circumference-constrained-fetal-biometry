"""Reference decoders and the parameterised plausibility gate used for the decoder comparison.

decode_strong      best pair among the top-k heat peaks (max summed heat, min separation 0.10*S).
                   A stronger *free* baseline: no mask, no ellipse.
decode_consistent  mask-free geometric baseline (head/abdomen): one pair per channel from the top-k
                   peaks maximising summed heat - lam*(centre offset / mean length) - lam*|cos angle|.
gate_ok / gate_variants   the plausibility gate with adjustable thresholds (+-25 % sensitivity study).
"""
import itertools

import numpy as np

from .config import S, STRIDE

GATE0 = dict(amin=.06, amax=.65, ratio=2.6, margin=.15)   # thresholds as in the manuscript (S = 256)
DMIN = .10 * S


def gate_ok(el, g=GATE0):
    if el is None:
        return False
    cx, cy, a, b, _ = el
    m = g["margin"] * S
    if not (g["amin"] * S <= a <= g["amax"] * S and g["amin"] * S <= b <= g["amax"] * S):
        return False
    if max(a, b) / max(min(a, b), 1e-6) > g["ratio"]:
        return False
    return (-m <= cx <= S + m) and (-m <= cy <= S + m)


def gate_variants():
    """Base gate, each threshold scaled by 0.75 and 1.25, and amax = 0.45."""
    g = {"base": GATE0}
    for key, val in GATE0.items():
        for tag, f in (("m25", .75), ("p25", 1.25)):
            g[f"{key}_{tag}"] = dict(GATE0, **{key: val * f})
    g["amax_045"] = dict(GATE0, amax=.45)
    return g


def peaks(h, k=6, excl=6):
    """Top-k heat peaks as (value, (x, y) canvas coordinates) with disc suppression and 5x5 centroid."""
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    h = h.astype(np.float64).copy()
    out = []
    for _ in range(k):
        y, x = np.unravel_index(h.argmax(), h.shape)
        v = h[y, x]
        if v <= -1e8:
            break
        y0, y1, x0, x1 = max(y - 2, 0), min(y + 3, H), max(x - 2, 0), min(x + 3, W)
        p = np.clip(h[y0:y1, x0:x1], 0, None) + 1e-6
        cy = (p * yy[y0:y1, x0:x1]).sum() / p.sum()
        cx = (p * xx[y0:y1, x0:x1]).sum() / p.sum()
        out.append((float(v), np.array([(cx + .5) * STRIDE - .5, (cy + .5) * STRIDE - .5])))
        h[(yy - y) ** 2 + (xx - x) ** 2 <= excl ** 2] = -1e9
    return out


def _pairs(pk, dmin=DMIN):
    return [(pk[i][0] + pk[j][0], pk[i][1], pk[j][1]) for i, j in itertools.combinations(range(len(pk)), 2)
            if np.linalg.norm(pk[i][1] - pk[j][1]) >= dmin]


def decode_strong(hm, k=5):
    """hm: numpy (B, K, H, W) -> (B, K, 2, 2) canvas coordinates."""
    B, K = hm.shape[:2]
    out = np.zeros((B, K, 2, 2), np.float32)
    for b in range(B):
        for c in range(K):
            pk = peaks(hm[b, c], k)
            pr = _pairs(pk)
            if pr:
                _, p, q = max(pr, key=lambda t: t[0])
            else:
                p, q = pk[0][1], pk[min(1, len(pk) - 1)][1]
            out[b, c, 0], out[b, c, 1] = p, q
    return out


def decode_consistent(hm, lam=2.0, k=6):
    """Two-channel (head/abdomen) mask-free decoder; lam is fixed a priori (not tuned)."""
    B = hm.shape[0]
    out = np.zeros((B, 2, 2, 2), np.float32)
    for b in range(B):
        c0, c1 = _pairs(peaks(hm[b, 0], k)), _pairs(peaks(hm[b, 1], k))
        if not c0 or not c1:
            out[b] = decode_strong(hm[b:b + 1])[0]
            continue
        best = None
        for s0, p0, q0 in c0:
            v0 = q0 - p0
            l0 = np.linalg.norm(v0)
            for s1, p1, q1 in c1:
                v1 = q1 - p1
                l1 = np.linalg.norm(v1)
                off = np.linalg.norm((p0 + q0) / 2 - (p1 + q1) / 2) / ((l0 + l1) / 2)
                cs = abs(v0 @ v1) / (l0 * l1 + 1e-9)
                sc = s0 + s1 - lam * off - lam * cs
                if best is None or sc > best[0]:
                    best = (sc, p0, q0, p1, q1)
        out[b, 0, 0], out[b, 0, 1], out[b, 1, 0], out[b, 1, 1] = best[1:]
    return out
