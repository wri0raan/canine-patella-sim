import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestClassifier
from analysis.identify import ALL, noisy
from analysis.video_style_features import sim_landmarks, windows, video_landmarks, features

D = pd.read_csv("results/ident_results/dataset.csv"); gr = D[(D.hyp != "body") | D.cond.str.match(r"idg_M(10|08)F00_S")].copy()
gr["knee"] = np.select([gr.grade == 0, gr.grade == 2], ["normal", "abnormal"], "severe"); gr["chain"] = "base"
LAB = {"": "normal", "_G2": "abnormal", "_G3": "severe"}
rows = []
for ch in "AB":
    for suf, lab in LAB.items():
        run = f"fit{ch}{suf}"
        try: L, fps = sim_landmarks(f"results/ident/{run}.npz")
        except Exception as e: print("missing", run, e); continue
        for w in windows(L, fps, 3.0, 1.0):
            if all(np.isfinite(w[k]) for k in ALL): rows.append(dict(run=run, chain=ch, knee=lab, **{k: w[k] for k in ALL}))
F = pd.DataFrame(rows); print(F.groupby(["chain", "knee"])[["speed", "stride_hz", "step", "hock_min", "hock_max"]].median().round(2))
SETS = {"speed+stride+step": ["speed", "stride_hz", "step"], "stride+step": ["stride_hz", "step"], "stride": ["stride_hz"]}
rng = np.random.default_rng(0); ORDER = ["normal", "abnormal", "severe"]
mk = lambda d, f: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, class_weight="balanced").fit(d[f], d.knee)
out1 = []
for nm, f in SETS.items():
    m = mk(pd.concat([gr, F[F.chain == "A"]]), f); te = F[F.chain == "B"]
    for sc in (0.0, 1.0):
        p = m.predict(noisy(te, f, sc, rng)); rec = {c: float((p[te.knee.values == c] == c).mean()) for c in ORDER if (te.knee == c).any()}
        out1.append(dict(test="T1 chain B (train existing+A)", feats=nm, noise=sc, bal_acc=float(np.mean(list(rec.values()))), **{"recall_" + c: v for c, v in rec.items()}))
T1 = pd.DataFrame(out1); T1.to_csv("results/ident_results/K_fit.csv", index=False); print(T1.round(2).to_string())
z = np.load("data/real/leo2/work/leo_kp.npz", allow_pickle=True); kp, names = z["kp"], list(z["names"])
LEO = {k: features(video_landmarks(kp, names, int(a * 120), int(b * 120) - 1), 120) for k, (a, b) in {"A": (43.7, 46.8), "B": (33.8, 36.8)}.items()}
out2 = []
for lab, tr in (("T0 existing only", gr), ("T2 existing + A", pd.concat([gr, F[F.chain == "A"]])), ("T3 existing + A + B", pd.concat([gr, F]))):
    for nm, f in SETS.items():
        m = mk(tr, f)
        for k, v in LEO.items():
            pr = dict(zip(m.classes_, m.predict_proba(pd.DataFrame([{c: v[c] for c in f}]))[0]))
            out2.append(dict(train=lab, feats=nm, leo=k, p_normal=pr.get("normal", 0), p_abnormal=pr.get("abnormal", 0), p_severe=pr.get("severe", 0), call=max(pr, key=pr.get)))
T2 = pd.DataFrame(out2); T2.to_csv("results/ident_results/K_fit_leo.csv", index=False); print(T2.round(2).to_string())
