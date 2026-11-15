import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestClassifier
from analysis.identify import ROBUST, noisy

D = pd.read_csv("results/ident_results/dataset.csv"); rng = np.random.default_rng(0)
gr = D[(D.hyp != "body") | D.cond.str.match(r"idg_M(10|08)F00_S")].copy().reset_index(drop=True)
gr["knee"] = np.select([gr.grade == 0, gr.grade == 2], ["normal", "abnormal"], "severe")
ORDER = ["normal", "abnormal", "severe"]


def weights(df, speed_balanced):
    if not speed_balanced:
        return df.knee.map(1 / df.knee.value_counts()).values * len(df) / 3
    b = (df.speed / 0.05).round().astype(int); w = np.zeros(len(df))
    for _, idx in df.groupby(b).groups.items():
        sub = df.loc[idx]; cnt = sub.knee.value_counts(); w[df.index.get_indexer(idx)] = sub.knee.map(len(sub) / (len(cnt) * cnt)).values
    return w


def cv_probs(speed_balanced, sc):
    P = np.zeros((len(gr), 3))
    for g in gr.group.unique():
        tr, te = gr.group != g, gr.group == g
        d = gr[tr]; m = RandomForestClassifier(300, min_samples_leaf=2, random_state=0).fit(d[ROBUST], d.knee, sample_weight=weights(d, speed_balanced))
        pr = m.predict_proba(noisy(gr[te], ROBUST, sc, rng)); P[np.where(te)[0][:, None], [list(m.classes_).index(c) if c in m.classes_ else 0 for c in ORDER]] = pr
    return P


def decide(P, tau=None):
    if tau is None: return np.array(ORDER)[P.argmax(1)]
    other = np.where(P[:, 1] >= P[:, 2], "abnormal", "severe")
    return np.where(P[:, 0] >= tau, "normal", other)


def report(name, P, sc, tau=None):
    p = decide(P, tau); rec = {c: float((p[gr.knee == c] == c).mean()) for c in ORDER}
    return dict(variant=name, noise=sc, tau=tau, bal_acc=float(np.mean(list(rec.values()))), **{"recall_" + c: v for c, v in rec.items()})


rows = []; probs = {}
for name, sb in (("base", False), ("spdbal", True)):
    for sc in (0.0, 1.0, 2.0):
        P = cv_probs(sb, sc); probs[(name, sc)] = P; rows.append(report(name, P, sc))
P1 = probs[("spdbal", 1.0)]; tau = next((t for t in np.arange(0.05, 0.96, 0.01) if report("x", P1, 1.0, t)["recall_normal"] >= 0.80), None)
tau = None if tau is None else round(float(tau), 2)
print("tau", tau)
if tau is not None:
    for sc in (0.0, 1.0, 2.0): rows.append(report("spdbal+tau", probs[("spdbal", sc)], sc, tau))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/I_knee3_variants.csv", index=False); json.dump(dict(tau=tau, speed_balanced=True), open("results/ident_results/I_knee3_choice.json", "w"))
print(R.round(2).to_string())
