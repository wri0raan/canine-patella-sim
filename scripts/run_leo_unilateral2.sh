#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
prev=swLU2c_g2
step() {  # tag grade ext(or "")
  tag=$1; g=$2; ext=$3
  if [ ! -f results/$tag/done.txt ]; then
    EXT=""; [ -n "$ext" ] && EXT="--ext-override $ext"
    $PY scripts/optimize_gait.py --grade $g --mode functional --init results/$prev/best.npz --leo --sigma0 0.04 --gens 40 --speed 0.73 --gait-set eval_duration=14 target_speed=0.73 --cost-set $COST --w-meta 10 --style-ref leoH3 --style-w 2.0 --free-legs LH --affected LH --seed 1 $EXT --tag $tag > results_$tag.log 2>&1
  fi
  prev=$tag
}
step swLU2d_a 3 138
step swLU2d_b 3 132
step swLU2d_c 3 ""
step swLU2d_d 4 120
step swLU2d_e 4 ""
echo UNI2_DONE >> results_leo_sweeps_progress.txt
