from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metrics import hind_angles
from sim.config import ROOT

SETTINGS = {"swM1d": "H1 mechanical", "swF1d": "H1 functional", "swF2d": "H2 functional", "swH3d": "H3 pain avoidance"}
LEO_SETTINGS = {"swLM1s": "H1 mechanical", "swLF1s": "H1 functional", "swLF2s": "H2 functional", "swLH3s": "H3 pain avoidance"}
SFX = ""
FEATS = ["stifle_min", "stifle_max", "hock_min", "hock_max", "hip_rom"]


LIMIT = {
    ("H1 mechanical", 0): 162.9, ("H1 mechanical", 2): 155.0, ("H1 mechanical", 3): 149.7, ("H1 mechanical", 4): 140.0,
    ("H1 functional", 2): 144.9, ("H1 functional", 3): 125.1, ("H1 functional", 4): 115.0,
    ("H2 functional", 2): 144.9, ("H2 functional", 3): 125.1, ("H2 functional", 4): 115.0,
    ("H3 pain avoidance", 2): 144.9, ("H3 pain avoidance", 3): 125.1, ("H3 pain avoidance", 4): 115.0,
    ("healthy", 0): 162.9,
}


def cycle_features(run: str) -> list[list[float]]:
    p = ROOT / "results" / run / "log.npz"
    L = dict(np.load(p))
    out = dict(L); out["freq"] = float(L["freq"]); out["warmup"] = float(L["warmup"])
    ang = hind_angles(out)
    t, f0, t0 = L["time"], float(L["freq"]), float(L["warmup"])
    period = 1.0 / f0
    rows = []
    for c in range(int((t[-1] - t0) // period)):
        m = (t >= t0 + c * period) & (t < t0 + (c + 1) * period)
        if m.sum() < 5:
            continue
        r = []
        for key, fn in (("stifle", np.min), ("stifle", np.max), ("hock", np.min), ("hock", np.max)):
            r.append(np.mean([fn(ang[leg][key][m]) for leg in ("LH", "RH")]))
        r.append(np.mean([np.ptp(ang[leg]["hip"][m]) for leg in ("LH", "RH")]))
        rows.append(r)
    return rows


def build(seeds: list[int]) -> pd.DataFrame:
    recs = []
    for code, label in SETTINGS.items():
        for sd in seeds:
            if not (ROOT / "results" / f"{code}{sd}_summary.csv").exists():
                continue
            for grade in (2, 3, 4):
                run = f"{code}{sd}_g{grade}"
                if (ROOT / "results" / run / "log.npz").exists():
                    for i, r in enumerate(cycle_features(run)):
                        recs.append(dict(run=run, seed=sd, setting=label, grade=grade, cycle=i, **dict(zip(FEATS, r))))
            run0 = f"{code}{sd}_g0"
            if code in ("swF2d", "swLF2s") and (ROOT / "results" / run0 / "log.npz").exists():
                for i, r in enumerate(cycle_features(run0)):
                    recs.append(dict(run=run0, seed=sd, setting="healthy", grade=0, cycle=i, **dict(zip(FEATS, r))))
    return pd.DataFrame(recs)


def evaluate(df: pd.DataFrame, target: str, cv: str, sigma: float, model_name: str, rng) -> tuple[float, float, np.ndarray, list]:
    X = df[FEATS].values + rng.normal(0, sigma, size=(len(df), len(FEATS)))
    y = df[target].astype(str).values
    groups = df["seed"].values if cv == "seed" else df["run"].values
    n_splits = min(len(np.unique(groups)), 3 if cv == "seed" else 5)
    accs, yt, yp = [], [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if model_name == "random forest":
            clf = RandomForestClassifier(n_estimators=200, random_state=0)
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(float(np.mean(pred == y[te]))); yt += list(y[te]); yp += list(pred)
    classes = sorted(set(y))
    return float(np.mean(accs)), float(np.std(accs)), confusion_matrix(yt, yp, labels=classes), classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="run", choices=["run", "seed"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10])
    ap.add_argument("--leo", action="store_true", help="leo-model sweeps (swL*s<seed>); outputs get a _leo suffix")
    a = ap.parse_args()
    global SETTINGS, SFX
    if a.leo:
        SETTINGS, SFX = LEO_SETTINGS, "_leo"
    df = build(a.seeds)
    if df.empty:
        print("no data"); return
    df["cause_grade"] = df["setting"] + " G" + df["grade"].astype(str)
    df["limit"] = [LIMIT[(a_, b_)] for a_, b_ in zip(df.setting, df.grade)]
    print(f"{len(df)} cycles from {df.run.nunique()} runs, seeds {sorted(df.seed.unique())}, settings {sorted(df.setting.unique())}")
    imp = df[df.grade >= 3]
    tasks = {"grade (0,2,3,4)": (df, "grade"), "cause at grade 3-4": (imp[imp.setting != "healthy"], "setting")}
    rows, cms = [], {}
    for tname, (d, target) in tasks.items():
        chance = d[target].value_counts(normalize=True).max()
        for model in ("random forest", "logistic regression"):
            for sg in a.sigmas:
                rng = np.random.default_rng(0)
                acc, sd, cm, cls = evaluate(d, target, a.cv, sg, model, rng)
                rows.append(dict(task=tname, model=model, noise_deg=sg, accuracy=acc, sd=sd, majority_baseline=chance, n=len(d)))
                if model == "random forest" and sg in (0, 6):
                    cms[(tname, sg)] = (cm, cls)
    R = pd.DataFrame(rows); R.to_csv(ROOT / "results" / f"pilot_classifier{SFX}.csv", index=False)
    print(R.pivot_table(index=["task", "model"], columns="noise_deg", values="accuracy").round(2).to_string())
    print("majority-class baseline:", R.groupby("task")["majority_baseline"].first().round(2).to_dict())
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    rrows = []
    for model_name in ("random forest", "ridge"):
        for sg in a.sigmas:
            rng = np.random.default_rng(0)
            X = df[FEATS].values + rng.normal(0, sg, size=(len(df), len(FEATS))); y = df["limit"].values
            groups = df["seed"].values if a.cv == "seed" else df["run"].values
            maes, base = [], []
            for tr, te in GroupKFold(n_splits=min(len(np.unique(groups)), 3 if a.cv == "seed" else 5)).split(X, y, groups):
                reg = RandomForestRegressor(n_estimators=200, random_state=0) if model_name == "random forest" else make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                reg.fit(X[tr], y[tr]); maes.append(float(np.mean(np.abs(reg.predict(X[te]) - y[te])))); base.append(float(np.mean(np.abs(y[tr].mean() - y[te]))))
            rrows.append(dict(model=model_name, noise_deg=sg, mae_deg=np.mean(maes), mae_sd=np.std(maes), baseline_mae_deg=np.mean(base)))
    RR = pd.DataFrame(rrows); RR.to_csv(ROOT / "results" / f"pilot_regression{SFX}.csv", index=False)
    print("\nknee extension limit regression, MAE (deg); baseline = predict the training mean:")
    print(RR.pivot_table(index="model", columns="noise_deg", values="mae_deg").round(1).to_string()); print("baseline MAE:", round(RR.baseline_mae_deg.mean(), 1))
    out = ROOT / "results" / "figures"; out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for (tname, model), g in R.groupby(["task", "model"]):
        ax.plot(g.noise_deg, g.accuracy, marker="o", ms=3, ls="-" if model == "random forest" else "--", label=f"{tname}, {model}")
    for tname, g in R.groupby("task"):
        ax.axhline(g.majority_baseline.iloc[0], color="gray", lw=0.6, ls=":")
    ax.set_xlabel("angle-extraction noise sigma (deg)"); ax.set_ylabel("accuracy (held-out)"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=5.5, frameon=False); ax.set_title("Pilot: recovering grade / cause from joint angles", fontsize=8)
    fig.savefig(out / f"pilot_accuracy_vs_noise{SFX}.pdf", bbox_inches="tight"); fig.savefig(out / f"pilot_accuracy_vs_noise{SFX}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(1, len(cms), figsize=(3.2 * len(cms), 3))
    axs = np.atleast_1d(axs)
    for ax, ((tname, sg), (cm, cls)) in zip(axs, cms.items()):
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(cls))); ax.set_yticklabels(cls, fontsize=6)
        ax.set_title(f"{tname}\nnoise {sg} deg (random forest)", fontsize=7)
        for i in range(len(cls)):
            for j in range(len(cls)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6, color="k" if cmn[i, j] < 0.6 else "w")
    fig.savefig(out / f"pilot_confusion{SFX}.pdf", bbox_inches="tight"); fig.savefig(out / f"pilot_confusion{SFX}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved figures")


if __name__ == "__main__":
    main()
