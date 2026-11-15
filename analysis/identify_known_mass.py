import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestRegressor
from analysis.identify import ALL, ROBUST, NOISE, MASS0, noisy

D = pd.read_csv("results/ident_results/dataset.csv")
body = D[(D.grade == 0) & (D.hyp == "body")].copy(); body["mass"] = MASS0 * (1 + body.fat)
rng = np.random.default_rng(0); rows = []
base = float(np.mean(np.abs(body.muscle - body.muscle.mean())))
for fs_name, feats in (("ALL", ALL), ("ROBUST", ROBUST)):
    for use_mass in (False, True):
        cols = feats + (["mass"] if use_mass else [])
        for sc in (0.0, 0.5, 1.0, 2.0):
            pred = pd.Series(np.nan, index=body.index)
            for g in body.cond.unique():
                tr, te = body.cond != g, body.cond == g
                m = RandomForestRegressor(300, min_samples_leaf=2, random_state=0).fit(body.loc[tr, cols], body.loc[tr, "muscle"])
                X = noisy(body[te], feats, sc, rng)
                if use_mass: X["mass"] = body.loc[te, "mass"].values
                pred[te] = m.predict(X[cols])
            mae = float(np.mean(np.abs(pred - body.muscle)))
            rows.append(dict(feats=fs_name, mass_input=use_mass, noise=sc, mae=mae, baseline_mae=base, skill=1 - mae / base))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/F_known_mass.csv", index=False)
print(R.round(3).to_string())
