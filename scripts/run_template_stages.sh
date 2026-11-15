#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SH=$(cat results/template_shift.txt)
CS="w_toeoff=12 w_tdlag=4 tdlag_target=0.02 w_retract=3 w_peak=2 w_paw=4 fall_const=120 w_ref=6 w_carpus_rom=1"
prev=results/h24/best.npz
for a in 0.3 0.6 1.0; do
  tag=h26_a$a
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --grade 0 --mode functional --init $prev --calibrate --template --template-mix $a --template-shift $SH --sigma0 0.04 --gens 100 --speed 0.9 --w-meta 10 --cost-set $CS --tag $tag > results_$tag.log 2>&1
  prev=results/$tag/best.npz
done
echo ALLDONE > results_template_stages.txt
