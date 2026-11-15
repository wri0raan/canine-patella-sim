import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.judge_knee import judge, load, LEO_DELTA
k2, n2 = load("data/real/leo2/work/leo_kp.npz")
P = {"leo2@34s": (k2, n2, int(33.78 * 120), int(36.82 * 120) - 1, 120), "leo2@44s": (k2, n2, int(43.73 * 120), int(46.70 * 120) - 1, 120)}
for t, (t0, t1) in {"n2": (2.8, 5.8), "n4": (1.6, 5.2)}.items():
    kp, nm = load(f"data/real/leo4/work/{t}_kp.npz"); P[f"leo4 {t}"] = (kp, nm, int(t0 * 120), int(t1 * 120) - 1, 120)
rows = []
def run(test_names, base_names, scheme):
    base = [P[b] for b in base_names]
    for t in test_names:
        kp, nm, a, b, fps = P[t]
        for truth in ("normal", "abnormal", "severe"):
            r = judge(kp, nm, a, b, fps, 7.0, base, leo_model=True, shift=LEO_DELTA[truth])
            rows.append(dict(scheme=scheme, clip=t, truth=truth, call=r["call"], p_normal=r["p_normal"], p_abnormal=r["p_abnormal"], p_severe=r["p_severe"], reliable=r["reliable"], speed=r["speed"], stif_mean=r["stif_mean"]))
names = list(P); print("passes:", names)
for t in names: run([t], [o for o in names if o != t], "leave-one-pass-out (4 walking passes, 2 sessions)")
l2 = [n for n in names if n.startswith("leo2")]; l4 = [n for n in names if n.startswith("leo4")]
run(l2, l4, "cross-session: leo4 baseline -> leo2 passes"); run(l4, l2, "cross-session: leo2 baseline -> leo4 passes")
R = pd.DataFrame(rows); R.to_csv("results/judge_leo_walk_eval.csv", index=False)
R["ok"] = R.truth == R.call; R["pmax"] = R[["p_normal", "p_abnormal", "p_severe"]].max(axis=1)
for sc, g in R.groupby("scheme"):
    print(f"{sc}: acc normal {g[g.truth=='normal'].ok.mean():.2f}  abnormal {g[g.truth=='abnormal'].ok.mean():.2f}  severe {g[g.truth=='severe'].ok.mean():.2f}  overall {g.ok.mean():.2f} (n={len(g)}) | mean pmax {g.pmax.mean():.2f}")
pd.set_option("display.width", 220); print(R[R.scheme.str.startswith("leave")].round(2)[["clip", "truth", "call", "p_normal", "p_abnormal", "p_severe", "speed", "stif_mean"]].to_string())
