#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
step() {  # tag init gens sigma body-set...
  tag=$1; init=$2; gens=$3; sig=$4; shift 4
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/$init/best.npz --leo --sigma0 $sig --gens $gens --speed 0.73 --gait-set eval_duration=14 target_speed=0.73 --cost-set $COST --w-meta 10 --style-ref leoH3 --style-w 2.0 --body-set "$@" --tag $tag > results_$tag.log 2>&1
  echo "$tag done" >> results_composition_progress.txt
}
step cmp_M90 leoH3 50 0.015 muscle_hind=0.9
step cmp_M80 cmp_M90 50 0.015 muscle_hind=0.8
step cmp_M70 cmp_M80 50 0.015 muscle_hind=0.7
step cmp_M60 cmp_M70 60 0.015 muscle_hind=0.6
step cmp_M50 cmp_M60 60 0.015 muscle_hind=0.5
step cmp_F10 leoH3 50 0.015 fat=0.1
step cmp_F20 cmp_F10 50 0.015 fat=0.2
step cmp_F30 cmp_F20 50 0.015 fat=0.3
step cmp_F40 cmp_F30 60 0.015 fat=0.4
step cmp_M60F20 cmp_M60 60 0.015 muscle_hind=0.6 fat=0.2
echo ALLDONE >> results_composition_progress.txt
