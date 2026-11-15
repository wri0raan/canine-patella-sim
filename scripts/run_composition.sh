#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
run() {  # tag style(h1|h2) body-set...
  tag=$1; st=$2; shift 2
  [ -f results/$tag/done.txt ] && return
  EXTRA=""; [ "$st" = h2 ] && EXTRA="--style-ref leoH3 --style-w 2.0"
  $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/leoH3/best.npz --leo --sigma0 0.05 --gens 80 --speed 0.65 --gait-set eval_duration=14 target_speed=0.65 --cost-set $COST --w-meta 10 $EXTRA --body-set "$@" --tag $tag > results_$tag.log 2>&1
  echo "$tag done" >> results_composition_progress.txt
}
run cmpH2_C0   h2 muscle_hind=1.0
run cmpH2_M70  h2 muscle_hind=0.7
run cmpH2_M50  h2 muscle_hind=0.5
run cmpH2_F20  h2 fat=0.2
run cmpH2_F40  h2 fat=0.4
run cmpH2_M60F20 h2 muscle_hind=0.6 fat=0.2
run cmpH1_C0   h1 muscle_hind=1.0
run cmpH1_M50  h1 muscle_hind=0.5
run cmpH1_F40  h1 fat=0.4
echo ALLDONE >> results_composition_progress.txt
