import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_gait import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--limits", type=float, nargs="+", default=[158, 154, 150, 146, 142, 138, 134, 130, 126, 122, 118])
ap.add_argument("--gens", type=int, default=40)
ap.add_argument("--style-ref", default=None)
ap.add_argument("--style-w", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--sigma0", type=float, default=0.06)
ap.add_argument("--speed", type=float, default=None)
a = ap.parse_args()

import yaml
speed = a.speed or round(float(pd.read_csv(ROOT / "results" / "swF2d1_summary.csv").sort_values("grade").iloc[0]["speed"]), 2)
prev = f"results/{a.base}/best.npz"
rows = []
for lim in a.limits:
    tag = f"{a.tag}_L{int(lim)}"
    cmd = [sys.executable, str(ROOT / "scripts/optimize_gait.py"), "--grade", "3", "--mode", "functional", "--speed", str(speed),
           "--gens", str(a.gens), "--tag", tag, "--init", prev, "--sigma0", str(a.sigma0), "--seed", str(a.seed),
           "--ext-override", str(lim)]
    if a.style_ref:
        cmd += ["--style-ref", a.style_ref, "--style-w", str(a.style_w)]
    print(f"=== limit {lim} ===", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print("\n".join(r.stdout.strip().splitlines()[-2:]), flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:]); sys.exit(1)
    pj = ROOT / "results" / tag / "patella.yaml"
    d = yaml.safe_load(open(pj, encoding="utf-8"))
    d["grades"][3]["prom_ext"] = lim; d["grades"][3]["arom_ext"] = lim
    yaml.safe_dump(d, open(pj, "w", encoding="utf-8"), sort_keys=False)
    s = evaluate(tag, quiet=True); s["limit"] = lim; rows.append(s)
    prev = f"results/{tag}/best.npz"
pd.DataFrame(rows).to_csv(ROOT / "results" / f"{a.tag}_summary.csv", index=False)
print("wrote", ROOT / "results" / f"{a.tag}_summary.csv")
