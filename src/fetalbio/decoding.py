"""Free decoding and circumference-constrained decoding.

Constrained decoding (paper, Section 4.4):
  1. threshold the predicted mask at 0.5, keep the largest connected component, fit an ellipse;
  2. plausibility gate on size, axis ratio and centre position;
  3. snap each measurement's endpoints to the ellipse boundary: 180 angle samples in [0, pi),
     diametrically opposite pairs scored by the heatmap at the two points;
  4. if the mask ellipse is implausible, fall back to an ellipse built from the free-decoded
     landmarks; if that is also implausible, return the free-decoded landmarks.
"""
import math

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from .config import S, STRIDE
from .data import ellipse_from_pts


def decode_free(hm, excl=6):
    """Free decoding: 5x5-window centroid at the peak, then suppress radius `excl` and repeat."""
    hm = hm.detach().float().cpu().numpy()
    B, K, H, W = hm.shape
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((B, K, 2, 2), np.float32)
    for b in range(B):
        for k in range(K):
            h = hm[b, k].copy()
            for j in range(2):
                y, x = np.unravel_index(h.argmax(), h.shape)
                y0, y1, x0, x1 = max(y - 2, 0), min(y + 3, H), max(x - 2, 0), min(x + 3, W)
                p = np.clip(h[y0:y1, x0:x1], 0, None) + 1e-6
                cy = (p * yy[y0:y1, x0:x1]).sum() / p.sum()
                cx = (p * xx[y0:y1, x0:x1]).sum() / p.sum()
                out[b, k, j] = ((cx + .5) * STRIDE - .5, (cy + .5) * STRIDE - .5)
                h[(yy - y) ** 2 + (xx - x) ** 2 <= excl ** 2] = -1e9
    return out


def _mask_ellipse_canvas(prob):
    n, lab, st, _ = cv2.connectedComponentsWithStats((prob > .5).astype(np.uint8))
    if n <= 1:
        return None
    b = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (x, y), (w, h), ang = cv2.fitEllipse(c)
    return ((x + .5) * STRIDE - .5, (y + .5) * STRIDE - .5, w / 2 * STRIDE, h / 2 * STRIDE, ang)


def _plausible(ell):
    """Plausibility gate: semi-axes in [0.06S, 0.65S], ratio <= 2.6, centre within [-0.15S, 1.15S]."""
    if ell is None:
        return False
    cx, cy, a, b, _ = ell
    if not (0.06 * S <= a <= 0.65 * S and 0.06 * S <= b <= 0.65 * S):
        return False
    if max(a, b) / max(min(a, b), 1e-6) > 2.6:
        return False
    if not (-0.15 * S <= cx <= 1.15 * S and -0.15 * S <= cy <= 1.15 * S):
        return False
    return True


def robust_ellipse(prob, free_pts):
    """Return (ellipse, source) with source in {"mask", "fallback_from_landmarks", "free"}."""
    el = _mask_ellipse_canvas(prob)
    if _plausible(el):
        return el, "mask"
    el2 = ellipse_from_pts(free_pts)
    return (el2, "fallback_from_landmarks") if _plausible(el2) else (None, "free")


def guided_decode(hm_np, K, ell):
    """Snap endpoints of each measurement channel to the ellipse boundary."""
    cx, cy, a, b, ang = ell
    r = math.radians(ang)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    t = np.linspace(0, np.pi, 180, endpoint=False)
    E = (R @ np.stack([a * np.cos(t), b * np.sin(t)])).T
    c = np.array([cx, cy])
    out = np.zeros((K, 2, 2), np.float32)
    for k in range(K):
        s = 0
        for sg in (1, -1):
            q = (c + sg * E + .5) / STRIDE - .5
            s = s + map_coordinates(hm_np[k], [q[:, 1], q[:, 0]], order=1, mode="nearest")
        j = int(s.argmax())
        out[k, 0] = c + E[j]
        out[k, 1] = c - E[j]
    return out
