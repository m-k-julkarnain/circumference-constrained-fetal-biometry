"""All paper tables from the merged per-image CSVs (no GPU needed).

Reads  <out>/results/<Anatomy>_per_image.csv  (run merge_per_seed.py first)
Writes one CSV per table into <out>/results/tables/ and prints LaTeX-ready rows.
"""
import os

from _common import base_parser, parse

from fetalbio import analysis as A
from fetalbio.utils import results_dir

p = base_parser(__doc__)
p.add_argument("--n-boot", type=int, default=2000)
a = parse(p)
R = results_dir(a.out)
T = os.path.join(R, "tables")
os.makedirs(T, exist_ok=True)

raw, SA = A.load_all(R)
for a_, d in SA.items():
    print(f"{a_}: {len(raw[a_])} raw rows -> {len(d)} seed-averaged images")

tables = {
    "table4_constrained_vs_free": A.constrained_vs_free(SA),
    "table4b_med_dice": A.table_med_dice(SA, os.path.join(R, "free_med_per_image.csv")),
    "table8_sdr": A.table_sdr(SA),
    "table9_ga_days": A.table_ga(SA),
    "table10_observer_band": A.table_observer_band(SA),
    "table11_fallback": A.table_fallback(raw),
    "table12_failure": A.table_failure(SA),
    "table13_bootstrap": A.cluster_bootstrap(SA, n_boot=a.n_boot),
    "table14_seed_variance": A.table_seed_variance(raw),
}
for name, t in tables.items():
    t.to_csv(os.path.join(T, name + ".csv"), index=False)
    print(f"\n=========== {name} ===========")
    print(t.to_string(index=False))
print(f"\nTables written to {T}")
