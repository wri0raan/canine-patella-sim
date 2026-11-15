import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.judge_knee import judge, load, LEO_DELTA
k2, n2 = load("data/real/leo2/work/leo_kp.npz")
P = {}
for _, r in pd.read_csv("results/leo_segments.csv").iterrows():
    if r.speed_ms >= 2.0: P[f"leo2@{r.start:.0f}s"] = (k2, n2, int(r.start * 120), int(r.end * 120) - 1, 120)
for t, (t0, t1) in {"n1": (0.0, 0.9), "n2": (1.1, 2.5), "n4": (0.4, 1.7), "n5": (0.6, 2.1)}.items():
    kp, nm = load(f"data/real/leo3/work/{t}_kp.npz"); P[f"leo3 {t}"] = (kp, nm, int(t0 * 120), int(t1 * 120) - 1, 120)
rows = []
def run(test_names, base_names, scheme):
    base = [P[b] for b in base_names]
    for t in test_names:
        kp, nm, a, b, fps = P[t]
        for truth in ("normal", "abnormal", "severe"):
            r = judge(kp, nm, a, b, fps, 7.0, base, leo_model=True, shift=LEO_DELTA[truth])
            rows.append(dict(scheme=scheme, clip=t, truth=truth, call=r["call"], p_normal=r["p_normal"], p_abnormal=r["p_abnormal"], p_severe=r["p_severe"], reliable=r["reliable"], speed=r["speed"], stif_mean=r["stif_mean"]))
names = list(P); print("passes:", names)
for t in names: run([t], [o for o in names if o != t], "leave-one-pass-out (8 passes, 2 sessions)")
l2 = [n for n in names if n.startswith("leo2")]; l3 = [n for n in names if n.startswith("leo3")]
run(l2, l3, "cross-session: leo3 baseline -> leo2 passes"); run(l3, l2, "cross-session: leo2 baseline -> leo3 clips")
R = pd.DataFrame(rows); R.to_csv("results/judge_leo_eval2.csv", index=False)
R["ok"] = R.truth == R.call; R["pmax"] = R[["p_normal", "p_abnormal", "p_severe"]].max(axis=1)
for sc, g in R.groupby("scheme"):
    print(f"{sc}: acc normal {g[g.truth=='normal'].ok.mean():.2f}  abnormal {g[g.truth=='abnormal'].ok.mean():.2f}  severe {g[g.truth=='severe'].ok.mean():.2f}  overall {g.ok.mean():.2f} (n={len(g)}) | mean pmax {g.pmax.mean():.2f} | reliable {int(g.reliable.sum())}/{len(g)} acc among reliable {g[g.reliable].ok.mean():.2f}")
pd.set_option("display.width", 220); print(R[R.scheme.str.startswith("leave")].round(2)[["clip", "truth", "call", "p_normal", "p_abnormal", "p_severe", "speed", "stif_mean"]].to_string())
