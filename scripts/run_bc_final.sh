#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; BPY=.venv_bpy/Scripts/python.exe
export PYTHONUNBUFFERED=1
for r in fitR3 fitR3_G2 fitR3_G3; do
  [ -f results/bc_motion_$r.npz ] || $PY scripts/export_motion_for_blender.py $r results/bc_motion_$r.npz --seconds 10 --fps 30 > results_bcexp_$r.log 2>&1
  echo "export $r $(date +%H:%M)" >> results_bc_final_progress.txt
done
for r in fitR3 fitR3_G2 fitR3_G3; do
  [ -f results/bcA_frames_$r/done.txt ] || { $BPY -I scripts/bc_retarget_render.py results/bc_motion_$r.npz results/bcA_frames_$r --mode hybrid --ears up --w 800 --h 450 > results_bcA_$r.log 2>&1 && echo ok > results/bcA_frames_$r/done.txt; }
  echo "A $r $(date +%H:%M)" >> results_bc_final_progress.txt
done
for r in fitR3 fitR3_G2 fitR3_G3; do
  [ -f results/bcB_frames_$r/done.txt ] || { $BPY -I scripts/bc_retarget_render.py results/bc_motion_$r.npz results/bcB_frames_$r --mode hybrid --ears fold --albedo results/bc_leo_albedo.png --w 800 --h 450 > results_bcB_$r.log 2>&1 && echo ok > results/bcB_frames_$r/done.txt; }
  echo "B $r $(date +%H:%M)" >> results_bc_final_progress.txt
done
echo BCDONE >> results_bc_final_progress.txt
