"""Annotation audits (no GPU): (i) orthogonality and shared centre of the annotated diameters,
(ii) subject-ID audit (rows that rely on the image-name split rule).
Writes <out>/results/geometry_audit.csv and subject_id_audit.csv
"""
import numpy as np
import pandas as pd
from _common import base_parser, parse

from fetalbio import config as C
from fetalbio.data import load_df
from fetalbio.utils import results_dir

p = base_parser(__doc__)
a = parse(p)
R = results_dir(a.out)
rows = []
for anat in C.ANATOMIES:
    if not C.CFG[anat]["aux"]:
        continue
    m = C.CFG[anat]["meas"]
    for dom in C.CFG[anat]["doms"]:
        d = load_df(dom, anat)
        P = d[[f"{x}_{i}_{c}" for x in m for i in (1, 2) for c in "xy"]].values.reshape(-1, 2, 2, 2)
        v1, v2 = P[:, 0, 1] - P[:, 0, 0], P[:, 1, 1] - P[:, 1, 0]
        c1, c2 = P[:, 0].mean(1), P[:, 1].mean(1)
        l1, l2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
        ang = np.degrees(np.arccos(np.clip(np.abs((v1 * v2).sum(1)) / (l1 * l2 + 1e-9), 0, 1)))
        off = np.linalg.norm(c1 - c2, axis=1) / ((l1 + l2) / 2 + 1e-9) * 100
        rows.append(dict(anatomy=anat, domain=dom, n=len(d), angle_median=np.median(ang),
                         pct_angle_ge80=100 * (ang >= 80).mean(), centre_offset_pct_median=np.median(off),
                         pct_offset_le5=100 * (off <= 5).mean(), pct_offset_le10=100 * (off <= 10).mean()))
G = pd.DataFrame(rows).round(2)
print(G.to_string(index=False))
G.to_csv(f"{R}/geometry_audit.csv", index=False)

rows = []
for anat in C.ANATOMIES:
    for dom in C.CFG[anat]["doms"]:
        raw = pd.read_csv(f"{C.DATA_ROOT}/annotations/{dom}/{anat}.csv")
        rec = raw.image_name.astype(str).str.split("_").str[0]
        has = (raw.SubjectID.notna() & (raw.SubjectID.astype(str) != "")) if "SubjectID" in raw else pd.Series(False, index=raw.index)
        agree = (raw.SubjectID.astype(str)[has] == rec[has]).mean() * 100 if has.any() else np.nan
        rows.append(dict(anatomy=anat, domain=dom, rows=len(raw), subject_id_missing=int((~has).sum()),
                         reconstructed_unique=int(rec[~has].nunique()),
                         provided_vs_string_split_agree_pct=round(agree, 1) if agree == agree else np.nan))
S_ = pd.DataFrame(rows)
print(S_.to_string(index=False))
S_.to_csv(f"{R}/subject_id_audit.csv", index=False)
