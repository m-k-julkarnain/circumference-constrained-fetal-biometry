# results/

Empty by default. Scripts write generated files to `<out>/results/` (default `outputs/results/`),
which is not tracked by git: per-image CSVs, `decoders/`, `runs/`, `tables/` and `tables_extended/`.
To publish final per-image CSVs, copy them into this folder. Model checkpoints (`<out>/checkpoints/`)
are not distributed; retrain with `scripts/train_lodo_mc.py`.
