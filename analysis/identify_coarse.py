import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestClassifier
from analysis.identify import ALL, ROBUST, noisy

D = pd.read_csv("results/ident_results/dataset.csv"); rng = np.random.default_rng(0); rows = []

def bal_acc(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))

def loo(df, label, feats, gcol, sc):
    p = pd.Series("", index=df.index, dtype=object)
    for g in df[gcol].unique():
        tr, te = df[gcol] != g, df[gcol] == g
        m = RandomForestClassifier(300, min_samples_leaf=2, random_state=0, class_weight="balanced").fit(df.loc[tr, feats], df.loc[tr, label])
        p[te] = m.predict(noisy(df[te], feats, sc, rng))
    return p

body = D[(D.grade == 0) & (D.hyp == "body")].copy()
body["mcls"] = pd.cut(body.muscle, [0, 0.75, 1.05, 9], labels=["weak", "normal", "strong"]).astype(str)
body = body[~body.muscle.between(1.05, 1.15)]
gr = D[(D.hyp != "body") | D.cond.str.match(r"idg_M(10|08)F00_S")].copy()
tasks = [("muscle3_allfat", body, "mcls", "cond"), ("muscle3_fat0", body[body.fat == 0], "mcls", "cond")]
gr["abn2"] = np.where(gr.grade >= 2, "abnormal", "ok"); gr["sev3"] = np.where(gr.grade >= 3, "severe", "ok")
tasks += [("knee_ok_vs_abn(g>=2)", gr, "abn2", "group"), ("knee_ok_vs_severe(g>=3)", gr[gr.grade != 2], "sev3", "group")]
for name, df, lab, gcol in tasks:
    ncls = df[lab].nunique(); print(name, "classes", df[lab].value_counts().to_dict(), "groups", df[gcol].nunique())
    for fn, f in (("ALL", ALL), ("ROBUST", ROBUST)):
        for sc in (0.0, 0.5, 1.0, 2.0):
            p = loo(df, lab, f, gcol, sc); rows.append(dict(task=name, feats=fn, noise=sc, bal_acc=bal_acc(df[lab].values, p.values), chance=1 / ncls))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/G_coarse.csv", index=False)
print(R.pivot_table(index=["task", "feats"], columns="noise", values="bal_acc").round(2).to_string()); print(R.groupby("task").chance.first())
