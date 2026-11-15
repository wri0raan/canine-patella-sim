import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--grade", type=int, default=0)
ap.add_argument("--mode", default="mechanical")
ap.add_argument("--speeds", type=float, nargs="+", default=[0.4, 0.7, 1.0, 1.35])
ap.add_argument("--digit-kps", type=float, nargs="+", default=None, help="digit stiffness per stage (same length as --speeds)")
ap.add_argument("--gens", type=int, default=60)
ap.add_argument("--init", default=None)
ap.add_argument("--free-legs", nargs="*", default=[])
ap.add_argument("--affected", nargs="*", default=None)
ap.add_argument("--sigma0", type=float, default=None)
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--extra", default="", help="extra optimize_gait arguments for every stage, e.g. \"--leo --gait-set eval_duration=8\"")
a = ap.parse_args()

prev = a.init
for k, v in enumerate(a.speeds):
    stage = f"{a.tag}_s{v:g}"
    cmd = [sys.executable, str(ROOT / "scripts/optimize_gait.py"), "--grade", str(a.grade), "--mode", a.mode,
           "--speed", str(v), "--gens", str(a.gens), "--tag", stage]
    if prev:
        cmd += ["--init", prev]
    if a.free_legs:
        cmd += ["--free-legs", *a.free_legs]
    if a.affected:
        cmd += ["--affected", *a.affected]
    if a.sigma0:
        cmd += ["--sigma0", str(a.sigma0)]
    if a.calibrate:
        cmd += ["--calibrate"]
    if a.digit_kps:
        cmd += ["--digit-kp", str(a.digit_kps[k])]
    if a.extra:
        cmd += shlex.split(a.extra)
    print(f"=== stage {k + 1}/{len(a.speeds)} target speed {v} m/s ===", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    print("\n".join(lines[-3:]), flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(1)
    prev = f"results/{stage}/best.npz"

final = ROOT / "results" / a.tag
final.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / prev, final / "best.npz")
shutil.copy(ROOT / f"results/{a.tag}_s{a.speeds[-1]:g}/trot.yaml", final / "trot.yaml")
shutil.copy(ROOT / f"results/{a.tag}_s{a.speeds[-1]:g}/patella.yaml", final / "patella.yaml")
print("done ->", final / "best.npz")
