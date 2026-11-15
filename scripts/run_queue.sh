#!/bin/sh
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
run() {  # run <summary-tag> <args...>
  tag=$1; shift
  if [ -f results/${tag}_summary.csv ]; then echo "skip $tag"; return; fi
  $PY scripts/sweep_grades.py --tag $tag "$@" >> results_$tag.log 2>&1
}
for SEED in 2 3; do
  run swF2d$SEED --base h11 --mode functional --style-ref h11 --style-w 2.0 --grades 2 3 4 --gens 40 --seed $SEED
  run swM1d$SEED --base h11 --mode mechanical --grades 2 3 4 --gens 40 --seed $SEED
  run swF1d$SEED --base h11 --mode functional --grades 2 3 4 --gens 40 --seed $SEED
  echo "seed $SEED done" >> results_queue_progress.txt
done
run swH3d1 --base h11 --mode functional --style-ref h11 --style-w 2.0 --load-avoid 0.10 --grades 2 3 4 --gens 40 --seed 1
echo "H3 done" >> results_queue_progress.txt
run swU2d1 --base h11 --mode functional --style-ref h11 --style-w 2.0 --free-legs LH --affected LH --grades 2 3 4 --gens 40 --seed 1
echo "unilateral done" >> results_queue_progress.txt
[ -f results/sensH2_summary.csv ] || $PY scripts/sensitivity.py --base h11 --tag sensH2 --style-ref h11 --style-w 2.0 >> results_sensH2.log 2>&1
[ -f results/sensH1_summary.csv ] || $PY scripts/sensitivity.py --base h11 --tag sensH1 >> results_sensH1.log 2>&1
echo ALLDONE >> results_queue_progress.txt
