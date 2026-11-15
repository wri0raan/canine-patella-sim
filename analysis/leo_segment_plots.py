import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from leo_segments import presence, analyse_segment, smooth_series
z = np.load(sys.argv[1]); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
fps = float(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 120.0
segs, _, _ = presence(kp, ix, fps)
fig, axs = plt.subplots(len(segs), 1, figsize=(11, 2.1 * len(segs)), squeeze=False)
col = {"LF": "tab:red", "RF": "tab:orange", "LH": "tab:blue", "RH": "tab:cyan"}
for ax, (a, b) in zip(axs[:, 0], segs):
    r, td = analyse_segment(kp, ix, a, b, fps)
    seg = kp[a:b + 1]; P = lambda nm: smooth_series(seg[:, ix[nm], :2], seg[:, ix[nm], 2])
    nose, tail, bb, be = P("nose"), P("tail_base"), P("back_base"), P("back_end"); f = (nose - tail).mean(0); f /= np.linalg.norm(f)
    ctr = 0.5 * (bb + be); L = np.median(np.linalg.norm(bb - be, axis=1)); t = np.arange(len(seg)) / fps + a / fps
    for leg, pre in (("LF", "front_left"), ("RF", "front_right"), ("LH", "back_left"), ("RH", "back_right")):
        u = ((P(pre + "_paw") - ctr) @ f) / L; ax.plot(t, u, color=col[leg], lw=1, label=leg); ax.plot(t[td[leg]], u[td[leg]], "o", color=col[leg], ms=3)
    ax.set_title(f"{r['start']:.1f}-{r['end']:.1f}s  {r['gait']}  speed {r['speed_ms']:.2f} m/s  stride {r['stride_period']:.2f}s", fontsize=8); ax.set_ylabel("paw u (BL)", fontsize=7)
axs[0, 0].legend(ncol=4, fontsize=7, frameon=False); axs[-1, 0].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130); print("saved", sys.argv[2])
