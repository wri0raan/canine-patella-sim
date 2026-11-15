from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml

SETTINGS = [("swM1", "H1 energy-only, mechanical limit", "#1f77b4", "s"),
            ("swF1", "H1 energy-only, functional limit", "#2ca02c", "^"),
            ("swF2", "H2 keep healthy excursions, functional limit", "#d62728", "o")]
CLIN = {0: dict(stifle_min=121.8, stifle_max=167.1, stifle_rom=48.93, hock_min=105.8, hock_max=157.4, hock_rom=47.59, hip_rom=19.02),
        3: dict(stifle_min=96.88, stifle_max=125.1, stifle_rom=28.77, hock_min=89.18, hock_max=168.7, hock_rom=78.38, hip_rom=35.53)}
METRICS = ["stifle_min", "stifle_max", "stifle_rom", "hock_min", "hock_max", "hock_rom", "hip_rom", "speed"]


def leg_avg(r, j, k):
    return (r[f"LH_{j}_{k}"] + r[f"RH_{j}_{k}"]) / 2


def row_metrics(r) -> dict:
    return dict(stifle_min=leg_avg(r, "stifle", "min"), stifle_max=leg_avg(r, "stifle", "max"), stifle_rom=leg_avg(r, "stifle", "rom"),
                hock_min=leg_avg(r, "hock", "min"), hock_max=leg_avg(r, "hock", "max"), hock_rom=leg_avg(r, "hock", "rom"),
                hip_rom=leg_avg(r, "hip", "rom"), speed=r["speed"])


def load(prefix: str, seeds: list[int], leo: bool = False) -> pd.DataFrame:
    rows = []
    for code, label, _, _ in SETTINGS:
        for sd in seeds:
            fc = code.replace("sw", "swL", 1) if leo else code
            p = ROOT / "results" / f"{fc}{prefix}{sd}_summary.csv"
            if not p.exists():
                continue
            for _, r in pd.read_csv(p).iterrows():
                rows.append(dict(setting=label, code=code, seed=sd, grade=int(r.grade), fell=bool(r.fell), **row_metrics(r)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="d")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--leo", action="store_true", help="Leo-model sweeps (tags swL*): use --prefix s; outputs get a _leo suffix")
    a = ap.parse_args()
    sfx = "_leo" if a.leo else ""
    df = load(a.prefix, a.seeds, a.leo)
    if df.empty:
        print("no summaries found"); return
    g = df.groupby(["setting", "grade"])
    mean, sd, n = g[METRICS].mean(), g[METRICS].std(ddof=0), g.size()
    table = mean.add_suffix("_mean").join(sd.add_suffix("_sd")).join(n.rename("n_seeds")).reset_index()
    table.to_csv(ROOT / "results" / f"final_table{sfx}.csv", index=False)
    print(table.round(1).to_string(index=False))
    errs = []
    for (setting, grade), _ in g:
        if grade in CLIN:
            m = mean.loc[(setting, grade)]; s_ = sd.loc[(setting, grade)]
            for k, v in CLIN[grade].items():
                errs.append(dict(setting=setting, grade=grade, metric=k, clinical=v, sim_mean=m[k], sim_sd=s_[k], error=m[k] - v))
    E = pd.DataFrame(errs); E.to_csv(ROOT / "results" / f"final_error_table{sfx}.csv", index=False)
    print("\nerror vs clinical (sim - clinical, deg):")
    print(E.pivot_table(index=["setting", "metric"], columns="grade", values="error").round(1).to_string())
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.3))
    ax = axs[0]
    grades = sorted(df.grade.unique()); off = {0: -0.2, 1: 0.0, 2: 0.2}
    for i, (code, label, c, mk) in enumerate(SETTINGS):
        sub = df[df.code == code]
        for gr in grades:
            s = sub[sub.grade == gr]
            if s.empty:
                continue
            lo, hi = s.stifle_min.mean(), s.stifle_max.mean()
            ax.plot([gr + off[i]] * 2, [lo, hi], color=c, lw=3, label=label if gr == grades[0] else None)
            ax.errorbar([gr + off[i]] * 2, [lo, hi], yerr=[s.stifle_min.std(ddof=0), s.stifle_max.std(ddof=0)], fmt="none", ecolor="k", lw=0.8, capsize=2)
    for gr, d in CLIN.items():
        ax.plot([gr + 0.4] * 2, [d["stifle_min"], d["stifle_max"]], color="k", lw=3, label="clinical (measured)" if gr == 0 else None)
    ax.set_title("Stifle angle window at trot"); ax.set_ylabel("included angle (deg)"); ax.set_xlabel("Patellar luxation grade")
    ax.set_xticks(grades); ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    for ax, key, title in ((axs[1], "hock_rom", "Hock ROM"), (axs[2], "hip_rom", "Hip ROM (trunk-femur)")):
        for code, label, c, mk in SETTINGS:
            sub = df[df.code == code].groupby("grade")[key].agg(["mean", "std"]).fillna(0)
            ax.errorbar(sub.index, sub["mean"], yerr=sub["std"], fmt=mk + "--", color=c, ms=4, capsize=2)
        ax.plot(list(CLIN), [CLIN[k][key] for k in CLIN], "ko-", ms=5, label="clinical (measured)")
        ax.set_title(title); ax.set_xlabel("Patellar luxation grade"); ax.set_ylabel("ROM per stride (deg)"); ax.set_xticks(grades)
    axs[1].legend(fontsize=6, frameon=False)
    fig.suptitle(f"Mean +- sd over {df.seed.nunique()} CMA-ES seeds; clinical: Chayatup 2025 (grade 0 = controls, grade III)", fontsize=7, y=1.03)
    out = ROOT / "results" / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / f"final_grades{sfx}.pdf", bbox_inches="tight"); fig.savefig(out / f"final_grades{sfx}.png", dpi=300, bbox_inches="tight")
    print("saved figure")


if __name__ == "__main__":
    main()
