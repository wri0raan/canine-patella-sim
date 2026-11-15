import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.leo_judge_lib import constants, clip_stifle, posterior

import os; tags = sys.argv[1:] or ["a", "b", "l2a", "l2b", "t7725"]; C = constants(runset=os.environ.get("RUNSET", "R3")); X = {t: clip_stifle(t, C["c_rump"]) for t in tags}
for t, v in X.items(): print(f"clip {t}: stifle {v[0]:.1f} (legs {v[2][0]:.1f} / {v[2][1]:.1f}), window {v[1]:.1f} s")
sess = lambda t: "leo2" if t.startswith("l2") else "leo4"
mu_abs = {"normal": C["mu_normal"], "abnormal": C["mu_normal"] + C["d_abnormal"], "severe": C["mu_normal"] + C["d_severe"]}; sd_abs = float(np.sqrt(4.5 ** 2 + C["sd_sim"] ** 2 + C["sd_c"] ** 2 + C["sd_tracker"] ** 2))
DL = {"normal": 0.0, "abnormal": C["d_abnormal"], "severe": C["d_severe"]}
rows = []
def judge(x, mus, sd, scheme, clip, truth):
    p = posterior(x, mus, sd); call = max(p, key=p.get); rows.append(dict(scheme=scheme, clip=clip, truth=truth, x=x, call=call, ok=call == truth, pmax=p[call], reliable=p[call] >= 0.8))
for t in tags:
    for truth, d in DL.items(): judge(X[t][0] + d, mu_abs, sd_abs, "absolute (sim only)", t, truth)
def hyb(base, scheme):
    B = [X[b][0] for b in base]; mb = float(np.mean(B)); n = len(B); sdw = max(float(np.std(B, ddof=1)) if n > 1 else 4.5, 4.0); mus = {k: mb + d for k, d in DL.items()}; sd = float(np.sqrt(sdw ** 2 * (1 + 1 / n) + 2.0 ** 2)); return mus, sd
for t in tags:
    if len(tags) > 2:
        mus, sd = hyb([b for b in tags if b != t], "loo")
        for truth, d in DL.items(): judge(X[t][0] + d, mus, sd, f"hybrid leave-one-clip-out ({len(tags) - 1} baseline clips)", t, truth)
for tr in (("leo2", "leo4"), ("leo4", "leo2")):
    bs = [t for t in tags if sess(t) == tr[0]]; ts = [t for t in tags if sess(t) == tr[1]]
    if len(bs) >= 2 and ts:
        mus, sd = hyb(bs, "x")
        for t in ts:
            for truth, d in DL.items(): judge(X[t][0] + d, mus, sd, f"hybrid cross-session: baseline {tr[0]} -> test {tr[1]}", t, truth)
R = pd.DataFrame(rows); R.to_csv("results/judge_hybrid_eval" + ("" if os.environ.get("RUNSET", "R3") == "fit" else "_" + os.environ["RUNSET"]) + ".csv", index=False); pd.set_option("display.width", 200)
S = R.groupby(["scheme", "truth"]).agg(n=("ok", "size"), acc=("ok", "mean"), pmax=("pmax", "mean"), reliable=("reliable", "mean")).round(2); print("\n" + S.to_string())
print("\noverall accuracy per scheme:"); print(R.groupby("scheme").ok.mean().round(2).to_string())
