#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
until grep -q "^ALLDONE" results_bone_progress.txt 2>/dev/null; do sleep 60; done
COST="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03"
step() {  # tag init speed grade gens body-set...
  tag=$1; init=$2; spd=$3; gr=$4; gens=$5; shift 5
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --grade $gr --mode functional --init results/$init/best.npz --leo --sigma0 0.015 --gens $gens --speed $spd --gait-set eval_duration=14 target_speed=$spd --cost-set $COST --w-meta 10 --style-ref leoH3 --style-w 2.0 --body-set "$@" --tag $tag > results_$tag.log 2>&1
  echo "$tag done" >> results_ident_progress.txt
}
for SP in 73 55; do
  if [ $SP = 73 ]; then ROOT=leoH3; V=0.73; else ROOT=leoH1; V=0.55; fi
  step idg_M10F00_S$SP $ROOT $V 0 40 muscle_hind=1.0
  step idg_M12F00_S$SP idg_M10F00_S$SP $V 0 40 muscle_hind=1.2
  step idg_M08F00_S$SP idg_M10F00_S$SP $V 0 40 muscle_hind=0.8
  step idg_M06F00_S$SP idg_M08F00_S$SP $V 0 40 muscle_hind=0.6
  step idg_M10F20_S$SP idg_M10F00_S$SP $V 0 40 muscle_hind=1.0 fat=0.2
  step idg_M10F40_S$SP idg_M10F20_S$SP $V 0 40 muscle_hind=1.0 fat=0.4
  step idg_M12F20_S$SP idg_M12F00_S$SP $V 0 40 muscle_hind=1.2 fat=0.2
  step idg_M12F40_S$SP idg_M12F20_S$SP $V 0 40 muscle_hind=1.2 fat=0.4
  step idg_M08F20_S$SP idg_M08F00_S$SP $V 0 40 muscle_hind=0.8 fat=0.2
  step idg_M08F40_S$SP idg_M08F20_S$SP $V 0 40 muscle_hind=0.8 fat=0.4
  step idg_M06F20_S$SP idg_M06F00_S$SP $V 0 40 muscle_hind=0.6 fat=0.2
  step idg_M06F40_S$SP idg_M06F20_S$SP $V 0 40 muscle_hind=0.6 fat=0.4
  step idg_G2M10_S$SP idg_M10F00_S$SP $V 2 50 muscle_hind=1.0
  step idg_G3M10_S$SP idg_G2M10_S$SP $V 3 50 muscle_hind=1.0
  step idg_G2M08_S$SP idg_M08F00_S$SP $V 2 50 muscle_hind=0.8
  step idg_G3M08_S$SP idg_G2M08_S$SP $V 3 50 muscle_hind=0.8
done
echo IDENTDONE >> results_ident_progress.txt
