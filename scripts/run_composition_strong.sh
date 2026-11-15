#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
until grep -q "^ALLDONE" results_composition_progress.txt 2>/dev/null; do sleep 60; done
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
step() {
  tag=$1; init=$2; gens=$3; sig=$4; shift 4
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/$init/best.npz --leo --sigma0 $sig --gens $gens --speed 0.73 --gait-set eval_duration=14 target_speed=0.73 --cost-set $COST --w-meta 10 --style-ref leoH3 --style-w 2.0 --body-set "$@" --tag $tag > results_$tag.log 2>&1
  echo "$tag done" >> results_composition_progress.txt
}
step cmp_P120 leoH3 50 0.015 muscle_hind=1.2
step cmp_P140 cmp_P120 50 0.015 muscle_hind=1.4
echo STRONGDONE >> results_composition_progress.txt
