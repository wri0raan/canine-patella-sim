import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_gait import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="run dir name of the healthy (Grade 0) gait")
ap.add_argument("--mode", default="functional", choices=["mechanical", "functional"])
ap.add_argument("--tag", required=True)
ap.add_argument("--gens", type=int, default=50)
ap.add_argument("--speed", type=float, default=None, help="common target speed; default = speed the base gait actually reaches")
ap.add_argument("--grades", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--free-legs", nargs="*", default=[])
ap.add_argument("--affected", nargs="*", default=None)
ap.add_argument("--style-ref", default=None, help="H2: healthy run whose joint excursions are kept")
ap.add_argument("--style-w", type=float, default=2.0)
ap.add_argument("--sigma0", type=float, default=0.06)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--load-avoid", type=float, default=0.0)
ap.add_argument("--w-meta", type=float, default=None)
ap.add_argument("--leo", action="store_true", help="Leo-proportioned model (target speed is then passed through --gait-set because apply_leo overrides it)")
ap.add_argument("--leo-config", default="leo")
ap.add_argument("--gait-set", nargs="*", default=[])
ap.add_argument("--cost-set", nargs="*", default=[])
a = ap.parse_args()

rows = []
base0 = f"{a.tag}_g0"
(ROOT / "results" / base0).mkdir(parents=True, exist_ok=True)
import numpy as np, shutil
z = np.load(ROOT / "results" / a.base / "best.npz", allow_pickle=True)
np.savez(ROOT / "results" / base0 / "best.npz", x=z["x"], cost=z["cost"], free_legs=z["free_legs"], grade=0,
         mode=a.mode, affected=z["affected"], summary="base")
shutil.copy(ROOT / "results" / a.base / "trot.yaml", ROOT / "results" / base0 / "trot.yaml")
rows.append(evaluate(base0, quiet=True))
speed = a.speed or round(rows[0]["speed"], 2)
print("common target speed", speed, "m/s", flush=True)
prev = f"results/{base0}/best.npz"
for g in a.grades:
    tag = f"{a.tag}_g{g}"
    cmd = [sys.executable, str(ROOT / "scripts/optimize_gait.py"), "--grade", str(g), "--mode", a.mode,
           "--speed", str(speed), "--gens", str(a.gens), "--tag", tag, "--init", prev, "--sigma0", str(a.sigma0)]
    if a.free_legs:
        cmd += ["--free-legs", *a.free_legs]
    if a.affected:
        cmd += ["--affected", *a.affected]
    if a.seed is not None:
        cmd += ["--seed", str(a.seed)]
    if a.load_avoid:
        cmd += ["--load-avoid", str(a.load_avoid)]
    if a.w_meta is not None:
        cmd += ["--w-meta", str(a.w_meta)]
    if a.leo:
        cmd += ["--leo", "--leo-config", a.leo_config]
    gs = list(a.gait_set) + ([f"target_speed={speed}"] if a.leo else [])
    if gs:
        cmd += ["--gait-set", *gs]
    if a.cost_set:
        cmd += ["--cost-set", *a.cost_set]
    if a.style_ref:
        cmd += ["--style-ref", a.style_ref, "--style-w", str(a.style_w)]
    print(f"=== grade {g} ({a.mode}) ===", flush=True)
    if (ROOT / "results" / tag / "done.txt").exists():
        print("already done, reusing", flush=True)
    else:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print("\n".join(r.stdout.strip().splitlines()[-2:]), flush=True)
        if r.returncode != 0:
            print(r.stderr[-1500:]); sys.exit(1)
    rows.append(evaluate(tag, quiet=True))
    prev = f"results/{tag}/best.npz"
pd.DataFrame(rows).to_csv(ROOT / "results" / f"{a.tag}_summary.csv", index=False)
print("wrote", ROOT / "results" / f"{a.tag}_summary.csv")
