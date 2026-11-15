#!/bin/sh
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
$PY scripts/leo_judge_pipeline.py data/real/leo2/leo_1080p120.mp4 l2a --start 33.0 --dur 4.8 --stages trim,crop,pose,convert,tracks > results_l2a.log 2>&1
$PY scripts/leo_judge_pipeline.py data/real/leo2/leo_1080p120.mp4 l2b --start 43.0 --dur 4.8 --stages trim,crop,pose,convert,tracks > results_l2b.log 2>&1
echo L2DONE >> results_l2a.log
