# Video -> physics model -> clinical-grade gait -> rendering: reproduction guide (2026-10-20)

Everything below runs from the repo root. `.venv` = simulation, `.venv_dlc` = pose estimation (torch), `.venv_bpy` = Blender 5.0.1 (run with `-I`).
Windows App Control sometimes blocks `.venv*/Scripts/python.exe`; see docs/STATE.md for the base-interpreter workaround.

| # | Stage | Command / file | Output | Manual? |
|---|---|---|---|---|
| 1 | Record video (4K, 120 fps) | - | data/real/leo2/*.mp4 | yes |
| 2 | Keypoints (SuperAnimal-Quadruped) | scripts/leo_pose.py (GPU, .venv_dlc); `analysis/leo_segments.py convert`, `analyse` | data/real/leo2/work/leo_kp.npz, results/leo_segments.csv | picking passes from contact sheets is manual |
| 3 | Body proportions | measure lengths + standing side photo; `analysis/auto_body_fit.py` (prototype), `scripts/overlay_leo_photo.py` (check) | scale factors -> configs/model/leo.yaml | measuring is manual; auto-fit gives a starting point (mean 13 % off the hand-tuned values) |
| 4 | Build model | `sim/dog_scale.build_leo_model` via `apply_leo` | MuJoCo model with skin | automatic |
| 5 | Healthy gait (staged) | `sh scripts/run_leo_chain4.sh`, `run_leo_chain5.sh`, `run_leo_chain6.sh` (leoF1..F2, G1, H1..H3) | results/leoH3 (0.73 m/s) | order of cost terms chosen by hand |
| 6 | Independent validation | `python scripts/eval_leo_gait.py leoH3` (20 s report) | metrics vs video | automatic |
| 7 | Clinical grades (limits) | `sh scripts/run_leo_sweeps.sh` (H1/H2/H3, 3 seeds), `run_leo_unilateral2.sh` | results/swL*_g{2,3,4}, *_summary.csv | automatic |
| 8 | Tables and figures | `python analysis/final_results.py --leo --prefix s`; `python analysis/pilot_classifier.py --leo --cv seed` | results/final_table_leo.csv, figures/final_grades_leo.png, pilot_*_leo | automatic |
| 9 | Skeleton / skin videos | `scripts/render_two_skeletons.py`, `scripts/render_skin_skeleton.py` | results/videos | automatic |
| 10 | Border Collie re-skin | `scripts/export_motion_for_blender.py <run> results/bc_motion_<run>.npz`; `.venv_bpy/Scripts/python.exe -I scripts/bc_retarget_render.py <npz> <frames_dir> [--ears up\|fold] [--albedo results/bc_leo_albedo.png]` | frames -> mp4 (imageio) | automatic |
| 11 | Coat repaint (optional) | `.venv_bpy ... scripts/bc_export_mesh_uv.py`; `python scripts/bc_leo_coat.py` | results/bc_leo_albedo.png | rule values by hand |

Gotchas: apply_leo overrides gait.target_speed (use `--gait-set target_speed=...`); evaluate >= 14 s (short runs hide late falls); kill orphan CMA workers
(PowerShell: Get-CimInstance Win32_Process, Stop-Process); Notion file upload content type must be `application/mp4`.
