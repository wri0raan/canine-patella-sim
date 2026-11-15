import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.identify import loo_regress, ALL
D = pd.read_csv("results/ident_results/dataset.csv"); D["mass"] = 7.0 * (1 + D.fat)
rows = []
for sp in ("S73", "S55"):
    d = D[D.cond.str.startswith("idg_M") & D.cond.str.endswith(sp)].copy().reset_index(drop=True)
    for name, feats in (("ALL", ALL), ("no_speed", [f for f in ALL if f != "speed"])):
        for tgt in ("muscle", "mass"):
            P = loo_regress(d, tgt, feats, "cond", [0.0, 1.0]); base = float(np.mean(np.abs(d[tgt] - d[tgt].mean())))
            for sc, p in P.items():
                rows.append(dict(speed=sp, runs=d.run.nunique(), conds=d.cond.nunique(), feats=name, target=tgt, noise=sc, mae=float(np.mean(np.abs(p - d[tgt]))), baseline_mae=base,
                                 r2=float(1 - ((p - d[tgt]) ** 2).sum() / ((d[tgt] - d[tgt].mean()) ** 2).sum())))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/D_speedmatched.csv", index=False); print(R.round(3).to_string())

from sklearn.ensemble import RandomForestRegressor
rows = []
for sp in ("S73", "S55"):
    d = D[D.cond.str.startswith("idg_M") & D.cond.str.endswith(sp)]; tr, te = d[d.seed == 1], d[d.seed == 2]
    if len(te) == 0: continue
    for name, feats in (("ALL", ALL), ("no_speed", [f for f in ALL if f != "speed"])):
        for tgt in ("muscle", "mass"):
            m = RandomForestRegressor(300, min_samples_leaf=2, random_state=0).fit(tr[feats], tr[tgt]); p = m.predict(te[feats])
            rows.append(dict(speed=sp, feats=name, target=tgt, n_train_runs=tr.run.nunique(), n_test_runs=te.run.nunique(), mae=float(np.mean(np.abs(p - te[tgt]))), baseline_mae=float(np.mean(np.abs(te[tgt] - tr[tgt].mean())))))
E = pd.DataFrame(rows); E.to_csv("results/ident_results/E_seed_transfer.csv", index=False); print(E.round(3).to_string())
