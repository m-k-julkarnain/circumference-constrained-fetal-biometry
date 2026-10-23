"""Merge <Anatomy>_per_image_seed{0,1,2}.csv into <Anatomy>_per_image.csv (one row per fold/variant/seed/image)."""
import os

import pandas as pd
from _common import base_parser, parse

from fetalbio import config as C
from fetalbio.utils import results_dir

p = base_parser(__doc__)
p.add_argument("--anatomies", nargs="+", default=list(C.ANATOMIES))
a = parse(p)
R = results_dir(a.out)
for anat in a.anatomies:
    parts = []
    for seed in a.seeds:
        f = f"{R}/{anat}_per_image_seed{seed}.csv"
        if os.path.exists(f):
            parts.append(pd.read_csv(f))
            print(f"{anat} seed{seed}: {len(parts[-1])} rows")
        else:
            print(f"{anat} seed{seed}: MISSING {f}")
    if not parts:
        continue
    full = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["fold", "variant", "seed", "image_name"], keep="last")
    full.to_csv(f"{R}/{anat}_per_image.csv", index=False)
    b = full[full.variant == "baseline"]
    print(f"{anat}: merged {len(full)} rows; seeds per fold:",
          {f: sorted(g.seed.unique().tolist()) for f, g in b.groupby("fold")})
