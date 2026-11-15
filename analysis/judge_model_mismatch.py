import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.judge_knee import judge, load
M = pd.read_csv("results/ident_results/O_marker_sensitivity.csv")
G2 = sorted(set(np.round(pd.concat([M.G2_A, M.G2_B]), 1))); G3 = sorted(set(np.round(M.G3_B, 1)))
TRUE = {"abnormal": [min(G2), float(np.median(G2)), max(G2)], "severe": [min(G3), float(np.median(G3)), max(G3)]}
k2, n2 = load("data/real/leo2/work/leo_kp.npz"); P = {}
for _, r in pd.read_csv("results/leo_segments.csv").iterrows():
    if r.speed_ms >= 2.0: P[f"leo2@{r.start:.0f}s"] = (k2, n2, int(r.start * 120), int(r.end * 120) - 1, 120)
for t, (t0, t1) in {"n1": (0.0, 0.9), "n2": (1.1, 2.5), "n4": (0.4, 1.7), "n5": (0.6, 2.1)}.items():
    kp, nm = load(f"data/real/leo3/work/{t}_kp.npz"); P[f"leo3 {t}"] = (kp, nm, int(t0 * 120), int(t1 * 120) - 1, 120)
rows = []
for t in P:
    base = [P[o] for o in P if o != t]; kp, nm, a, b, fps = P[t]
    for truth, shifts in TRUE.items():
        for lab, sh in zip(("strongest", "median", "weakest"), shifts):
            r = judge(kp, nm, a, b, fps, 7.0, base, leo_model=True, shift=sh)
            rows.append(dict(clip=t, truth=truth, true_change=sh, which=lab, call=r["call"], pmax=max(r["p_normal"], r["p_abnormal"], r["p_severe"]), reliable=r["reliable"]))
R = pd.DataFrame(rows); R["ok"] = R.truth == R.call; R.to_csv("results/judge_model_mismatch.csv", index=False)
print("grade 2 true changes", TRUE["abnormal"], " grade 3", TRUE["severe"])
print(R.groupby(["truth", "which", "true_change"]).agg(acc=("ok", "mean"), pmax=("pmax", "mean"), reliable=("reliable", "mean"), n=("ok", "size")).round(2).to_string())
