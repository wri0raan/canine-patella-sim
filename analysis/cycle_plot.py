import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metrics import hind_angles
from sim.config import ROOT
out_png, runs = sys.argv[1], sys.argv[2:]
fig, axs = plt.subplots(len(runs), 3, figsize=(10, 2.4 * len(runs)), squeeze=False)
for r, run in enumerate(runs):
    L = dict(np.load(ROOT / "results" / run / "log.npz")); L["freq"] = float(L["freq"]); L["warmup"] = float(L["warmup"])
    ang = hind_angles(L); t = L["time"]; f0 = L["freq"]; t0 = L["warmup"]
    ph = ((t - t0) * f0) % 1.0; ok = t >= t0
    stance = L["grf"][:, 2 + 0] > 2.0
    for c, (k, lab) in enumerate((("stifle", "Stifle (deg)"), ("hock", "Hock (deg)"))):
        ax = axs[r, c]; a = ang["LH"][k]
        ax.scatter(ph[ok], a[ok], s=1.5, c=np.where(stance[ok], "tab:red", "tab:blue")); ax.set_ylabel(lab); ax.set_title(f"{run}: {k}", fontsize=8)
        sp = np.zeros(50, bool)
    A = L["anchors"]; v = A[:, 3, 0] - A[:, 2, 0]; incl = np.degrees(np.arctan2(-v[:, 2], np.hypot(v[:, 0], v[:, 1])))
    ax = axs[r, 2]; ax.scatter(ph[ok], incl[ok], s=1.5, c=np.where(stance[ok], "tab:red", "tab:blue")); ax.set_title("metatarsus inclination (deg)", fontsize=8)
    for ax in axs[r]:
        ax.set_xlabel("gait phase (red = stance, blue = swing)")
fig.tight_layout(); fig.savefig(out_png, dpi=150)
print("saved", out_png)
