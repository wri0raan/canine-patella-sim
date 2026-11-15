#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run() {  # tag init gens speed f0 flo stride_target
  tag=$1; init=$2; gens=$3; spd=$4; f0=$5; flo=$6; st=$7
  [ -f results/$tag/done.txt ] && return
  $PY scripts/clip_freq_init.py $init ${tag}_init $flo $f0
  PYTHONUNBUFFERED=1 $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/${tag}_init/best.npz --calibrate --leo --sigma0 0.04 --gens $gens --speed $spd --gait-set eval_duration=14 freq_lo=$flo target_speed=$spd --cost-set $COST stride_target=$st --w-meta 10 --tag $tag > results_$tag.log 2>&1
}
COST="fall_const=120 w_yawrate=4 w_track=4 calib_scale=6 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
run leoS1 leoH1 100 0.52 1.80 1.60 0.8
run leoS2 leoS1 100 0.46 1.65 1.45 0.75
run leoS3 leoS2 100 0.41 1.50 1.30 0.7
run leoS4 leoS3 120 0.37 1.40 1.15 0.65
echo CHAIN8_DONE > results_leo_chain8.txt
