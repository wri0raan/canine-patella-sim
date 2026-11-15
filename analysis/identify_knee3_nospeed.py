import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestClassifier
from analysis.identify import noisy

D = pd.read_csv("results/ident_results/dataset.csv"); rng = np.random.default_rng(0)
gr = D[(D.hyp != "body") | D.cond.str.match(r"idg_M(10|08)F00_S")].copy().reset_index(drop=True)
gr["knee"] = np.select([gr.grade == 0, gr.grade == 2], ["normal", "abnormal"], "severe")
ORDER = ["normal", "abnormal", "severe"]
SETS = {"speed+stride+step (기존)": ["speed", "stride_hz", "step"], "stride": ["stride_hz"], "stride+step": ["stride_hz", "step"], "shape": ["stride_hz", "bounce", "hock_min", "hock_max", "carpus_min", "carpus_max"]}
rows = []
for nm, f in SETS.items():
    for sc in (0.0, 1.0, 2.0):
        p = pd.Series("", index=gr.index, dtype=object)
        for g in gr.group.unique():
            tr, te = gr.group != g, gr.group == g
            m = RandomForestClassifier(300, min_samples_leaf=2, random_state=0, class_weight="balanced").fit(gr.loc[tr, f], gr.loc[tr, "knee"]); p[te] = m.predict(noisy(gr[te], f, sc, rng))
        rec = {c: float((p[gr.knee == c] == c).mean()) for c in ORDER}
        rows.append(dict(feats=nm, noise=sc, bal_acc=float(np.mean(list(rec.values()))), **{"recall_" + c: v for c, v in rec.items()}))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/J_knee3_nospeed.csv", index=False); print(R.round(2).to_string())
rows2 = []
gr["bin"] = np.where(gr.knee == "normal", "normal", "impaired")
for nm, f in SETS.items():
    for sc in (0.0, 1.0):
        p = pd.Series("", index=gr.index, dtype=object)
        for g in gr.group.unique():
            tr, te = gr.group != g, gr.group == g
            m = RandomForestClassifier(300, min_samples_leaf=2, random_state=0, class_weight="balanced").fit(gr.loc[tr, f], gr.loc[tr, "bin"]); p[te] = m.predict(noisy(gr[te], f, sc, rng))
        rec = {c: float((p[gr.bin == c] == c).mean()) for c in ("normal", "impaired")}
        rows2.append(dict(feats=nm, noise=sc, bal_acc2=float(np.mean(list(rec.values()))), **rec))
R2 = pd.DataFrame(rows2); R2.to_csv("results/ident_results/J_knee2_nospeed.csv", index=False); print(R2.round(2).to_string())
