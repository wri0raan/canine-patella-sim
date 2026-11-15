import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.judge_knee import judge, load, LEO_DELTA
SEG = {"n1": (0.0, 0.9), "n2": (1.1, 2.5), "n4": (0.4, 1.7), "n5": (0.6, 2.1)}
K3 = {t: load(f"data/real/leo3/work/{t}_kp.npz") for t in SEG}; rows = []
for t, (t0, t1) in SEG.items():
    kp, nm = K3[t]; a, b = int(t0 * 120), int(t1 * 120) - 1
    base = [(K3[o][0], K3[o][1], int(SEG[o][0] * 120), int(SEG[o][1] * 120) - 1, 120) for o in SEG if o != t]
    for truth in ("normal", "abnormal", "severe"):
        r = judge(kp, nm, a, b, 120, 7.0, base, leo_model=True, shift=LEO_DELTA[truth]); rows.append(dict(clip=t, truth=truth, call=r["call"], p_normal=r["p_normal"], p_abnormal=r["p_abnormal"], p_severe=r["p_severe"], reliable=r["reliable"]))
R = pd.DataFrame(rows); R.to_csv("results/judge_leo_eval.csv", index=False); pd.set_option("display.width", 200); print(R.round(2).to_string())
print("accuracy by truth:", R.assign(ok=R.truth == R.call).groupby("truth").ok.mean().round(2).to_dict(), "| overall %.2f" % (R.truth == R.call).mean())
R["pmax"] = R[["p_normal", "p_abnormal", "p_severe"]].max(axis=1)
print("mean max probability %.2f ; reliable (>=0.8): %d/%d ; accuracy among reliable %.2f" % (R.pmax.mean(), R.reliable.sum(), len(R), (R[R.reliable].truth == R[R.reliable].call).mean() if R.reliable.any() else float("nan")))
