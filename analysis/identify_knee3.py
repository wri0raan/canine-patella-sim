import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestClassifier
from analysis.identify import ALL, ROBUST, noisy

D = pd.read_csv("results/ident_results/dataset.csv"); rng = np.random.default_rng(0)
gr = D[(D.hyp != "body") | D.cond.str.match(r"idg_M(10|08)F00_S")].copy()
gr["knee"] = np.select([gr.grade == 0, gr.grade == 2], ["normal", "abnormal"], "severe")
print(gr.knee.value_counts().to_dict(), "sweeps", gr.group.nunique(), "runs", gr.run.nunique())
rows, conf = [], []
order = ["normal", "abnormal", "severe"]
for fn, f in (("ALL", ALL), ("ROBUST", ROBUST)):
    for sc in (0.0, 0.5, 1.0, 2.0):
        p = pd.Series("", index=gr.index, dtype=object)
        for g in gr.group.unique():
            tr, te = gr.group != g, gr.group == g
            m = RandomForestClassifier(300, min_samples_leaf=2, random_state=0, class_weight="balanced").fit(gr.loc[tr, f], gr.loc[tr, "knee"]); p[te] = m.predict(noisy(gr[te], f, sc, rng))
        rec = {c: float((p[gr.knee == c] == c).mean()) for c in order}
        rows.append(dict(feats=fn, noise=sc, bal_acc=float(np.mean(list(rec.values()))), chance=1 / 3, **{"recall_" + c: v for c, v in rec.items()}, acc_all=float((p == gr.knee).mean())))
        if sc == 1.0:
            ct = pd.crosstab(gr.knee, p).reindex(index=order, columns=order, fill_value=0); ct = ct.div(ct.sum(1), axis=0); ct.insert(0, "feats", fn); conf.append(ct.reset_index())
R = pd.DataFrame(rows); R.to_csv("results/ident_results/H_knee3.csv", index=False); pd.concat(conf).to_csv("results/ident_results/H_knee3_confusion.csv", index=False)
print(R.round(2).to_string()); print(pd.concat(conf).round(2).to_string())
