#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run() {  # tag init gens sigma eval flo
  tag=$1; init=$2; gens=$3; sig=$4; ev=$5; flo=$6
  [ -f results/$tag/done.txt ] && return
  $PY scripts/clip_freq_init.py $init ${tag}_init $flo
  PYTHONUNBUFFERED=1 $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/${tag}_init/best.npz --calibrate --leo --sigma0 $sig --gens $gens --speed 0.85 --gait-set eval_duration=$ev freq_lo=$flo --cost-set $COST --w-meta 10 --tag $tag > results_$tag.log 2>&1
}
COST="fall_const=120 w_yawrate=4 w_track=4 calib_scale=6 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
run leoH1 leoG1 200 0.05 14 1.95
run leoH2 leoH1 200 0.05 14 2.15
run leoH3 leoH2 200 0.05 14 2.35
echo CHAIN6_DONE > results_leo_chain6.txt
