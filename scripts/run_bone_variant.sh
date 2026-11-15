#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
until grep -q "STRONGDONE" results_composition_progress.txt 2>/dev/null; do sleep 60; done
COST="fall_const=120 w_yawrate=4 w_track=4 calib_scale=6 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
SC="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
step() {  # tag init hind_upper
  tag=$1; init=$2; hu=$3
  [ -f results/$tag/done.txt ] || {
    sed "s/^hind_upper:.*/hind_upper: $hu/" configs/model/leo_bone.yaml > configs/model/leo_bone_$tag.yaml
    $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/$init/best.npz --calibrate --leo --leo-config leo_bone_$tag --sigma0 0.02 --gens 60 --speed 0.73 --gait-set eval_duration=14 target_speed=0.73 --cost-set $COST --w-meta 10 --tag $tag > results_$tag.log 2>&1; }
  echo "$tag done" >> results_bone_progress.txt
}
step bone_U58 leoH3 0.58
step bone_U61 bone_U58 0.61
step bone_U64 bone_U61 0.64
step bone_U66 bone_U64 0.66
BASE="--base bone_U66 --leo --leo-config leo_bone_bone_U66 --speed 0.73 --gens 60 --grades 2 3 4 --w-meta 10 --gait-set eval_duration=14 --cost-set $SC"
run() { tag=$1; shift; [ -f results/${tag}_summary.csv ] || $PY scripts/sweep_grades.py --tag $tag $BASE "$@" >> results_$tag.log 2>&1; echo "$tag done" >> results_bone_progress.txt; }
run swBF2s1 --mode functional --style-ref bone_U66 --style-w 2.0 --seed 1
run swBM1s1 --mode mechanical --seed 1
run swBF1s1 --mode functional --seed 1
cstep() {  # tag init body-set...
  tag=$1; init=$2; shift 2
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --grade 0 --mode functional --init results/$init/best.npz --leo --leo-config leo_bone_bone_U66 --sigma0 0.015 --gens 50 --speed 0.73 --gait-set eval_duration=14 target_speed=0.73 --cost-set $SC --w-meta 10 --style-ref bone_U66 --style-w 2.0 --body-set "$@" --tag $tag > results_$tag.log 2>&1
  echo "$tag done" >> results_bone_progress.txt
}
cstep cmpB_M80 bone_U66 muscle_hind=0.8
cstep cmpB_M60 cmpB_M80 muscle_hind=0.6
cstep cmpB_F20 bone_U66 fat=0.2
cstep cmpB_F40 cmpB_F20 fat=0.4
cstep cmpB_P120 bone_U66 muscle_hind=1.2
echo ALLDONE >> results_bone_progress.txt
