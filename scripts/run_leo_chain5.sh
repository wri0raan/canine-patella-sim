#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
while [ ! -f results/leoF2/done.txt ]; do sleep 20; done
run() {
  tag=$1; init=$2; gens=$3; sig=$4; ev=$5; shift 5
  [ -f results/$tag/done.txt ] || PYTHONUNBUFFERED=1 $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/$init/best.npz --calibrate --leo --sigma0 $sig --gens $gens --speed 0.85 --gait-set eval_duration=$ev --cost-set $COST "$@" --tag $tag > results_$tag.log 2>&1
}
COST="fall_const=120 w_yawrate=4 w_track=4 calib_scale=6 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1"; run leoG1 leoF2 250 0.05 14 --w-meta 10
echo CHAIN5_DONE > results_leo_chain5.txt
