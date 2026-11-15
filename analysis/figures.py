from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
COL = {"clinical": "#444444", "mechanical": "#1f77b4", "functional": "#d62728"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def rom_vs_grade(tag_suffix: str = ""):
    imp = load_yaml("configs/impairment/patella.yaml")["grades"]
    sweeps = [("H1 energy-only, mechanical limit", f"swM1{tag_suffix}", "#1f77b4", "s"),
              ("H1 energy-only, functional limit", f"swF1{tag_suffix}", "#2ca02c", "^"),
              ("H2 keep healthy excursions, functional limit", f"swF2{tag_suffix}", "#d62728", "o")]
    dfs = [(lab, pd.read_csv(ROOT / "results" / f"{t}_summary.csv").sort_values("grade"), c, m)
           for lab, t, c, m in sweeps if (ROOT / "results" / f"{t}_summary.csv").exists()]
    if not dfs:
        print("rom_vs_grade: no sweep summaries"); return
    avg = lambda r, j, k: (r[f"LH_{j}_{k}"] + r[f"RH_{j}_{k}"]) / 2
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax = axs[0]
    off = {0: -0.18, 1: 0.0, 2: 0.18}
    for i, (lab, df, c, m) in enumerate(dfs):
        for _, r in df.iterrows():
            g = int(r.grade) + off[i]
            ax.plot([g, g], [avg(r, "stifle", "min"), avg(r, "stifle", "max")], color=c, lw=3, solid_capstyle="butt",
                    label=lab if int(r.grade) == 0 else None)
    for g, lo, hi, meas in ((0, 121.8, 167.1, True), (3, 96.9, 125.1, True)):
        ax.plot([g + 0.38, g + 0.38], [lo, hi], color="k", lw=3, label="clinical (measured)" if g == 0 else None)
    ax.set_title("Stifle angle window at trot"); ax.set_ylabel("included angle (deg)")
    ax.set_xticks([0, 2, 3, 4]); ax.set_xlabel("Patellar luxation grade")
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    for ax, j, ck, title in ((axs[1], "hock", "hock_arom", "Hock ROM"), (axs[2], "hip", "hip_arom", "Hip ROM (trunk-femur)")):
        for lab, df, c, m in dfs:
            ax.plot(df["grade"], [avg(r, j, "rom") for _, r in df.iterrows()], m + "--", color=c, ms=4)
        ax.plot([0, 3], [imp[0][ck], imp[3][ck]], "ko-", ms=5, label="clinical (measured: grade 0 and III)")
        ax.set_title(title); ax.set_xlabel("Patellar luxation grade"); ax.set_ylabel("ROM per stride (deg)")
        ax.set_xticks([0, 2, 3, 4])
    axs[1].legend(fontsize=6, frameon=False)
    fig.suptitle("Preliminary: first healthy baseline, only grade 0 and III are clinical measurements", fontsize=7, y=1.03)
    save(fig, "rom_vs_grade_prelim")


def joint_traces(run: str, name: str):
    d = ROOT / "results" / run / "log.npz"
    if not d.exists():
        print(f"joint_traces: {run} not evaluated yet"); return
    from analysis.metrics import hind_angles
    L = dict(np.load(d))
    out = dict(L); out["freq"] = float(L["freq"]); out["warmup"] = float(L["warmup"])
    ang = hind_angles(out)
    t = L["time"]; m = (t >= out["warmup"]) & (t <= out["warmup"] + 3.0 / out["freq"])
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4))
    for j, c in (("hip", "tab:green"), ("stifle", "tab:blue"), ("hock", "tab:orange")):
        ax.plot(t[m] - t[m][0], ang["LH"][j][m], color=c, label=j)
    ax.set_xlabel("time (s)"); ax.set_ylabel("included angle (deg)"); ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"Left hind limb joint angles ({run})", fontsize=8)
    save(fig, name)


if __name__ == "__main__":
    rom_vs_grade()
    for run in ("h6",):
        joint_traces(run, f"joint_traces_{run}")
