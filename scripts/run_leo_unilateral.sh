#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
for G in 2 3 4; do
  [ -f results/swLU2c_g$G/done.txt ] || $PY scripts/optimize_gait.py --grade $G --mode functional --init results/swLF2s1_g$G/best.npz --leo --sigma0 0.05 --gens 60 --speed 0.73 --gait-set eval_duration=14 target_speed=0.73 --cost-set $COST --w-meta 10 --style-ref leoH3 --style-w 2.0 --free-legs LH --affected LH --seed 1 --tag swLU2c_g$G > results_swLU2c_g$G.log 2>&1
done
$PY - <<'PY'
import sys; sys.path.insert(0, ".")
import pandas as pd, shutil
from pathlib import Path
from scripts.evaluate_gait import evaluate
Path("results/swLU2c_g0").mkdir(exist_ok=True)
for f in Path("results/swLF2s1_g0").iterdir(): shutil.copy(f, Path("results/swLU2c_g0") / f.name)
rows = [evaluate(f"swLU2c_g{g}", quiet=True) for g in (0, 2, 3, 4)]
pd.DataFrame(rows).to_csv("results/swLU2c_summary.csv", index=False)
PY
echo UNI_DONE >> results_leo_sweeps_progress.txt
