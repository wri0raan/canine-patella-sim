import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.video_style_features import video_landmarks
from analysis.identify_angles import video_L, angle_series

SEG = {"n2": (2.8, 5.8), "n4": (1.6, 5.2)}
rows = []
for tag, (t0, t1) in SEG.items():
    z = np.load(f"data/real/leo4/work/{tag}_kp.npz"); kp, names = z["kp"], list(z["names"]); a, b = int(t0 * 120), int(t1 * 120) - 1
    A = angle_series(video_L(kp, names, a, b), 120)
    rows.append(dict(clip=tag, seconds=t1 - t0, **{f"{j}_{s}": f(A[j]) for j in ("hock", "stif", "carp", "elb")
                                                    for s, f in (("mean", np.mean), ("rom", lambda v: np.percentile(v, 97) - np.percentile(v, 3)))}))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/Q_leo4_walk.csv", index=False); pd.set_option("display.width", 220); print(R.round(1).to_string())
