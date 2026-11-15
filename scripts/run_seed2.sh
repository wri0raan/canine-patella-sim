#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
CC="fall_const=120 w_yawrate=4 w_track=4 calib_scale=6 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
step() {  # tag init speed grade gens body-set...
  tag=$1; init=$2; spd=$3; gr=$4; gens=$5; shift 5
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --seed 2 --grade $gr --mode functional --init results/$init/best.npz --leo --sigma0 0.015 --gens $gens --speed $spd --gait-set eval_duration=14 target_speed=$spd --cost-set $COST --w-meta 10 --style-ref leoH3 --style-w 2.0 --body-set "$@" --tag $tag > results_$tag.log 2>&1
  echo "$tag done" >> results_seed2_progress.txt
}
for SP in 73 55; do
  if [ $SP = 73 ]; then ROOT=leoH3; V=0.73; else ROOT=leoH1; V=0.55; fi
  step idg2_M10F00_S$SP $ROOT $V 0 40 muscle_hind=1.0
  step idg2_M12F00_S$SP idg2_M10F00_S$SP $V 0 40 muscle_hind=1.2
  step idg2_M08F00_S$SP idg2_M10F00_S$SP $V 0 40 muscle_hind=0.8
  step idg2_M06F00_S$SP idg2_M08F00_S$SP $V 0 40 muscle_hind=0.6
  step idg2_M10F20_S$SP idg2_M10F00_S$SP $V 0 40 muscle_hind=1.0 fat=0.2
  step idg2_M10F40_S$SP idg2_M10F20_S$SP $V 0 40 muscle_hind=1.0 fat=0.4
  step idg2_M12F20_S$SP idg2_M12F00_S$SP $V 0 40 muscle_hind=1.2 fat=0.2
  step idg2_M12F40_S$SP idg2_M12F20_S$SP $V 0 40 muscle_hind=1.2 fat=0.4
  step idg2_M08F20_S$SP idg2_M08F00_S$SP $V 0 40 muscle_hind=0.8 fat=0.2
  step idg2_M08F40_S$SP idg2_M08F20_S$SP $V 0 40 muscle_hind=0.8 fat=0.4
  step idg2_M06F20_S$SP idg2_M06F00_S$SP $V 0 40 muscle_hind=0.6 fat=0.2
  step idg2_M06F40_S$SP idg2_M06F20_S$SP $V 0 40 muscle_hind=0.6 fat=0.4
  step idg2_G2M10_S$SP idg2_M10F00_S$SP $V 2 50 muscle_hind=1.0
  step idg2_G3M10_S$SP idg2_G2M10_S$SP $V 3 50 muscle_hind=1.0
  step idg2_G2M08_S$SP idg2_M08F00_S$SP $V 2 50 muscle_hind=0.8
  step idg2_G3M08_S$SP idg2_G2M08_S$SP $V 3 50 muscle_hind=0.8
done
BASE="--base bone_U66 --leo --leo-config leo_bone_bone_U66 --speed 0.73 --gens 60 --grades 2 3 4 --w-meta 10 --gait-set eval_duration=14 --cost-set $COST"
run() { tag=$1; shift; [ -f results/${tag}_summary.csv ] || $PY scripts/sweep_grades.py --tag $tag $BASE "$@" >> results_$tag.log 2>&1; echo "$tag done" >> results_seed2_progress.txt; }
run swBF2s2 --mode functional --style-ref bone_U66 --style-w 2.0 --seed 2
run swBM1s2 --mode mechanical --seed 2
run swBF1s2 --mode functional --seed 2
cstep() {  # tag init body-set...
  tag=$1; init=$2; shift 2
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --seed 2 --grade 0 --mode functional --init results/$init/best.npz --leo --leo-config leo_bone_bone_U66 --sigma0 0.015 --gens 50 --speed 0.73 --gait-set eval_duration=14 target_speed=0.73 --cost-set $COST --w-meta 10 --style-ref bone_U66 --style-w 2.0 --body-set "$@" --tag $tag > results_$tag.log 2>&1
  echo "$tag done" >> results_seed2_progress.txt
}
cstep cmpB2_M80 bone_U66 muscle_hind=0.8
cstep cmpB2_M60 cmpB2_M80 muscle_hind=0.6
cstep cmpB2_F20 bone_U66 fat=0.2
cstep cmpB2_F40 cmpB2_F20 fat=0.4
cstep cmpB2_P120 bone_U66 muscle_hind=1.2
cs() { tag=$1; init=$2; gens=$3; shift 3; step $tag $init 0.73 0 $gens "$@"; }
cs cmp2_M90 leoH3 50 muscle_hind=0.9
cs cmp2_M80 cmp2_M90 50 muscle_hind=0.8
cs cmp2_M70 cmp2_M80 50 muscle_hind=0.7
cs cmp2_M60 cmp2_M70 60 muscle_hind=0.6
cs cmp2_M50 cmp2_M60 60 muscle_hind=0.5
cs cmp2_F10 leoH3 50 fat=0.1
cs cmp2_F20 cmp2_F10 50 fat=0.2
cs cmp2_F30 cmp2_F20 50 fat=0.3
cs cmp2_F40 cmp2_F30 60 fat=0.4
cs cmp2_M60F20 cmp2_M60 60 muscle_hind=0.6 fat=0.2
echo SEED2DONE >> results_seed2_progress.txt
