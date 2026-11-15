import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec
from sim.rollout import Sim

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--every", type=float, default=0.25)
a = ap.parse_args()

import yaml

cfg = load_yaml("configs/gait/trot.yaml")
imp = load_yaml("configs/impairment/patella.yaml")
used = ROOT / "results" / a.run / "trot.yaml"
if used.exists():
    cfg = yaml.safe_load(open(used, encoding="utf-8"))
sim = Sim(cfg)
if a.run == "init":
    spec = GaitSpec(cfg["gait"])
    sim.limiter.set_grade(imp, 0, "mechanical")
    x = spec.initial_guess()
else:
    z = np.load(ROOT / "results" / a.run / "best.npz", allow_pickle=True)
    spec = GaitSpec(cfg["gait"], list(z["free_legs"]))
    sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    x = z["x"]
o = sim.run(spec, x, record=True)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if not isinstance(v, np.ndarray)})
t = o["time"]
step = int(a.every / (t[1] - t[0]))
print(" t     z     up    x      y     yaw   grf(LF RF LH RH)")
for i in range(0, len(t), step):
    print(f"{t[i]:4.2f} {o['torso'][i,2]:.3f} {o['up'][i]:.2f} {o['torso'][i,0]:6.2f} {o['torso'][i,1]:6.2f} "
          f"{np.degrees(o['yaw'][i]):6.1f}  {np.round(o['grf'][i], 0)}")
