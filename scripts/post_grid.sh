#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
until grep -q "^IDENTDONE" results_ident_progress.txt 2>/dev/null; do sleep 60; done
$PY scripts/export_ident_landmarks.py > results_export2.log 2>&1
$PY analysis/identify.py > results/ident_results_log.txt 2>&1
$PY analysis/apply_leo.py > results_apply_leo.log 2>&1
echo POSTDONE > results_post_progress.txt
