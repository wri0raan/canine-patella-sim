#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
until [ -f results/fitR_G3/done.txt ]; do sleep 30; done
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03 stride_target=0 w_freq=3 w_speed=40 w_calib=8"
step() {  # tag init speed grade gens freq sigma extra...
  tag=$1; init=$2; spd=$3; gr=$4; gens=$5; fq=$6; sg=$7; shift 7
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --grade $gr --mode functional --init results/$init/best.npz --leo --sigma0 $sg --gens $gens --speed $spd \
      --gait-set eval_duration=14 target_speed=$spd freq_lo=$(echo "$fq" | awk '{printf "%.2f", $1-0.25}') --cost-set $COST freq_target=$fq --w-meta 10 --tag $tag "$@" > results_$tag.log 2>&1
  echo "$tag done $(date +%H:%M)" >> results_refit2_progress.txt
}
step fitR3 fitR2 0.80 0 50 2.20 0.02 --calibrate
step fitR3_G2 fitR3 0.80 2 50 2.20 0.015 --style-ref fitR3
step fitR3_G3 fitR3_G2 0.80 3 50 2.20 0.015 --style-ref fitR3
echo REFIT2DONE >> results_refit2_progress.txt
