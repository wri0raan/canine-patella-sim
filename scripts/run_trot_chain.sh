#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
BASE="fall_const=120 w_yawrate=4 w_track=4 w_clear=8 w_spine=2 w_sole=3 w_paw=3 w_ref=3 w_carpus_rom=1 w_toeoff=6 w_retract=2 w_peak=1 w_tdlag=4 tdlag_target=0.03 stride_target=0 w_freq=3 w_speed=40"
step() {  # tag init speed grade gens freq duty sigma style extra
  tag=$1; init=$2; spd=$3; gr=$4; gens=$5; fq=$6; duty=$7; sg=$8; sty=$9; ex=${10}
  [ -f results/$tag/done.txt ] || $PY scripts/optimize_gait.py --grade $gr --mode functional --init results/$init/best.npz --leo --sigma0 $sg --gens $gens --speed $spd \
      --gait-set eval_duration=8 target_speed=$spd freq_lo=$(echo "$fq" | awk '{printf "%.2f", $1-0.3}') amp_max=1.2 --cost-set $BASE freq_target=$fq duty_target=$duty \
      --w-meta 10 --style-ref $sty --style-w 0.3 $ex --tag $tag > results_$tag.log 2>&1
  echo "$tag done $(date +%H:%M)" >> results_trot_progress.txt
}
chain() {  # prefix extra-args
  p=$1; ex=$2
  step ${p}1 fitA   1.1 0 60 2.70 0.50 0.04 fitA "$ex"
  step ${p}2 ${p}1  1.5 0 60 2.85 0.46 0.04 fitA "$ex"
  step ${p}3 ${p}2  1.9 0 60 3.00 0.42 0.04 fitA "$ex"
  step ${p}4 ${p}3  2.3 0 60 3.10 0.40 0.04 fitA "$ex"
  step ${p}5 ${p}4  2.6 0 60 3.20 0.38 0.04 fitA "$ex"
}
chain trotA ""
chain trotS "--spine"
B=$($PY scripts/pick_trot.py trotA1 trotA2 trotA3 trotA4 trotA5 trotS1 trotS2 trotS3 trotS4 trotS5 2>> results_trot_progress.txt)
echo "best stage: $B" >> results_trot_progress.txt
if [ "$B" != "NONE" ]; then
  SPD=$($PY -c "import yaml;print(yaml.safe_load(open('results/$B/trot.yaml'))['gait']['target_speed'])")
  FQ=$($PY -c "import yaml;print(yaml.safe_load(open('results/$B/trot.yaml'))['cost']['freq_target'])")
  DT=$($PY -c "import yaml;print(yaml.safe_load(open('results/$B/trot.yaml'))['cost']['duty_target'])")
  EX=""; case "$B" in trotS*) EX="--spine";; esac
  step ${B}_G2 $B       $SPD 2 50 $FQ $DT 0.03 $B "$EX"
  step ${B}_G3 ${B}_G2  $SPD 3 50 $FQ $DT 0.03 $B "$EX"
fi
echo TROTDONE >> results_trot_progress.txt
