from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, ".")
from analysis.video_style_features import sim_landmarks, windows
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ALL = ["speed", "stride_hz", "hock_min", "hock_max", "carpus_min", "carpus_max", "bounce", "step"]
ROBUST = ["speed", "stride_hz", "step"]
NOISE = dict(speed=0.08, stride_hz=0.06, step=0.10, bounce=0.20, hock_min=5.0, hock_max=5.0, carpus_min=5.0, carpus_max=5.0)
MASS0 = 7.0


def load_runs(root="results"):
    rows = []
    for f in sorted(Path(root, "ident").glob("*.npz")):
        run = f.stem; d = Path(root) / run
        base = re.sub(r"^(idg|cmp)2_", r"\g<1>_", run); seed = 1 if base == run else 2
        if not (d / "trot.yaml").exists(): continue
        cfg = yaml.safe_load(open(d / "trot.yaml", encoding="utf-8")); z = np.load(d / "best.npz", allow_pickle=True); body = cfg.get("body") or {}
        m = re.match(r"swL(F2|F1|M1|H3)s(\d)_g(\d)", base)
        group = (m.group(1) + "s" + m.group(2)) if m else run
        hyp = m.group(1) if m else "body"
        g = re.match(r"idg_G(\d)M(\d+)_S(\d+)", base); b = re.match(r"idg_M(10|08)F00_S(\d+)", base)
        if g: group, hyp = f"idgM{g.group(2)}_S{g.group(3)}", "H2"
        elif b: group = f"idgM{b.group(1)}_S{b.group(2)}"
        try:
            L, fps = sim_landmarks(str(f)); W = windows(L, fps, 3.0, 1.0)
        except Exception as e:
            print("skip", run, e); continue
        for w in W:
            if any(not np.isfinite(w[k]) for k in ALL): continue
            rows.append(dict(run=run, cond=base, seed=seed, group=group, hyp=hyp, grade=int(z["grade"]), muscle=float(body.get("muscle_hind", 1.0)), fat=float(body.get("fat", 0.0)), **{k: w[k] for k in ALL}))
    return pd.DataFrame(rows)


def noisy(X: pd.DataFrame, feats, scale, rng):
    X = X[feats].copy()
    for k in feats:
        s = NOISE[k] * scale
        X[k] = X[k] * (1 + rng.normal(0, s, len(X))) if k in ("speed", "stride_hz", "step", "bounce") else X[k] + rng.normal(0, s, len(X))
    return X


def loo_regress(df, target, feats, group_col, noise_scales, model="rf", seed=0):
    rng = np.random.default_rng(seed); res = {}
    mk = (lambda: RandomForestRegressor(300, min_samples_leaf=2, random_state=0)) if model == "rf" else (lambda: make_pipeline(StandardScaler(), Ridge(1.0)))
    for sc in noise_scales:
        pred = pd.Series(np.nan, index=df.index)
        for g in df[group_col].unique():
            tr, te = df[group_col] != g, df[group_col] == g
            mdl = mk().fit(df.loc[tr, feats], df.loc[tr, target]); pred[te] = mdl.predict(noisy(df[te], feats, sc, rng))
        res[sc] = pred
    return res


def loo_classify(df, target, feats, group_col, noise_scales, seed=0):
    rng = np.random.default_rng(seed); res = {}
    for sc in noise_scales:
        pred = pd.Series(-1, index=df.index)
        for g in df[group_col].unique():
            tr, te = df[group_col] != g, df[group_col] == g
            mdl = RandomForestClassifier(300, min_samples_leaf=2, random_state=0).fit(df.loc[tr, feats], df.loc[tr, target]); pred[te] = mdl.predict(noisy(df[te], feats, sc, rng))
        res[sc] = pred
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/ident_results"); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load_runs(); df.to_csv(out / "dataset.csv", index=False); print(len(df), "windows,", df.run.nunique(), "runs")
    scales = [0.0, 0.5, 1.0, 2.0]; summ = []
    body = df[(df.grade == 0) & (df.hyp == "body")].copy(); body["mass"] = MASS0 * (1 + body.fat)
    print("A: body-only runs", body.run.nunique())
    for fs_name, feats in (("ALL", ALL), ("ROBUST", ROBUST)):
        for tgt in ("muscle", "mass"):
            P = loo_regress(body, tgt, feats, "cond", scales); base = float(np.mean(np.abs(body[tgt] - body[tgt].mean())))
            for sc, p in P.items():
                summ.append(dict(study="A", feats=fs_name, target=tgt, noise=sc, mae=float(np.mean(np.abs(p - body[tgt]))), baseline_mae=base, r2=float(1 - ((p - body[tgt]) ** 2).sum() / ((body[tgt] - body[tgt].mean()) ** 2).sum())))
            if fs_name == "ALL": body[f"pred_{tgt}"] = P[0.0]; body[f"pred_{tgt}_n1"] = P[1.0]
    body.to_csv(out / "A_predictions.csv", index=False)
    gr = df[(df.hyp != "body") | df.cond.str.match(r"idg_M(10|08)F00_S")].copy()
    print("B: grade runs", gr.run.nunique(), "sweeps", gr.group.nunique())
    for fs_name, feats in (("ALL", ALL), ("ROBUST", ROBUST)):
        P = loo_classify(gr, "grade", feats, "group", scales); chance = float(gr.grade.value_counts(normalize=True).max())
        for sc, p in P.items(): summ.append(dict(study="B", feats=fs_name, target="grade", noise=sc, acc=float((p == gr.grade).mean()), baseline_acc=chance))
        if fs_name == "ALL": gr["pred_grade"] = P[0.0]; gr["pred_grade_n1"] = P[1.0]
    gr.to_csv(out / "B_predictions.csv", index=False)
    clf = RandomForestClassifier(300, min_samples_leaf=2, random_state=0).fit(gr[ALL], gr.grade); body["grade_from_body"] = clf.predict(body[ALL])
    reg = RandomForestRegressor(300, min_samples_leaf=2, random_state=0).fit(body[ALL], body.muscle); gr["muscle_from_grade"] = reg.predict(gr[ALL])
    c1 = body.groupby("run").agg(muscle=("muscle", "first"), fat=("fat", "first"), pred_grade=("grade_from_body", lambda s: s.mode().iloc[0])).reset_index()
    c2 = gr.groupby(["hyp", "grade"]).agg(pred_muscle=("muscle_from_grade", "mean")).reset_index()
    c1.to_csv(out / "C_body_as_grade.csv", index=False); c2.to_csv(out / "C_grade_as_muscle.csv", index=False)
    S = pd.DataFrame(summ); S.to_csv(out / "summary.csv", index=False); print(S.round(3).to_string())
    print("\nC1: weak/strong/fat body runs classified as grade:"); print(c1.round(2).to_string())
    print("\nC2: grade runs -> predicted muscle:"); print(c2.round(2).to_string())


if __name__ == "__main__":
    main()
