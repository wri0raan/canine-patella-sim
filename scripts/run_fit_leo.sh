#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03 stride_target=0 w_freq=3 w_speed=40"
step() {  # tag init speed grade gens freq style-ref sigma
  tag=$1; init=$2; spd=$3; gr=$4; gens=$5; fq=$6; sty=$7; sg=$8
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --grade $gr --mode functional --init results/$init/best.npz --leo --sigma0 $sg --gens $gens --speed $spd --gait-set eval_duration=14 target_speed=$spd freq_lo=$(echo "$fq" | awk '{printf "%.2f", $1-0.25}') --cost-set $COST freq_target=$fq --w-meta 10 --style-ref $sty --style-w 2.0 --tag $tag > results_$tag.log 2>&1
  echo "$tag done" >> results_fit_progress.txt
}
if [ "$1" = "phase1" ]; then
  step fitA1 leoH3 0.73 0 40 2.50 leoH3 0.02
  step fitA  fitA1 0.73 0 40 2.58 leoH3 0.02
  echo PHASE1DONE >> results_fit_progress.txt; exit 0
fi
step fitA_G2 fitA 0.73 2 50 2.58 fitA 0.015
step fitA_G3 fitA_G2 0.73 3 50 2.58 fitA 0.015
step fitB1  fitA  0.55 0 40 2.60 fitA 0.02
step fitB   fitB1 0.55 0 40 2.67 fitA 0.02
step fitB_G2 fitB 0.55 2 50 2.67 fitB 0.015
step fitB_G3 fitB_G2 0.55 3 50 2.67 fitB 0.015
echo FITDONE >> results_fit_progress.txt
