import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec
from sim.rollout import Sim
imp = load_yaml("configs/impairment/patella.yaml")
out_png, runs = sys.argv[1], sys.argv[2:]
fig, axs = plt.subplots(len(runs), 4, figsize=(12, 2.4 * len(runs)), squeeze=False)
for r, run in enumerate(runs):
    d0 = ROOT / "results" / run
    cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
    sim = Sim(cfg); sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    o = sim.run(GaitSpec(cfg["gait"], list(z["free_legs"])), z["x"], record=True)
    t, q, grf, f0 = o["time"], o["q"], o["grf"], o["freq"]; ok = t >= 2.0
    ph = (t * f0) % 1.0; st = grf[:, 0] > 2.0
    for c, name in enumerate(("shoulder_L_extend", "elbow_L", "wrist_L")):
        k = sim.name_to_k[name]; a = np.degrees(q[:, sim.sv["qadr"].tolist().index(sim.sv["qadr"][k])] if False else q[:, k])
        axs[r, c].scatter(ph[ok], a[ok], s=1.5, c=np.where(st[ok], "tab:red", "tab:blue")); axs[r, c].set_title(f"{run}: {name} q (deg)", fontsize=8)
    axs[r, 3].scatter(ph[ok], grf[ok, 0], s=1.5, c="k"); axs[r, 3].set_title("LF vertical force (N)", fontsize=8)
    for ax in axs[r]: ax.set_xlabel("gait phase (red = stance)")
fig.tight_layout(); fig.savefig(out_png, dpi=140); print("saved", out_png)
