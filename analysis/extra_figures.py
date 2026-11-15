from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"; OUT = R / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
both = lambda d, k: (d[f"LH_{k}"] + d[f"RH_{k}"]) / 2

fig, axs = plt.subplots(1, 4, figsize=(12, 2.9))
for name, col, lab in (("sensH1", "tab:green", "H1 energy-only"), ("sensH2", "tab:red", "H2 keep healthy excursions")):
    d = pd.read_csv(R / f"{name}_summary.csv").sort_values("limit")
    axs[0].fill_between(d.limit, both(d, "stifle_min"), both(d, "stifle_max"), color=col, alpha=0.25)
    axs[0].plot(d.limit, both(d, "stifle_max"), color=col, label=lab); axs[0].plot(d.limit, both(d, "stifle_min"), color=col)
    axs[1].plot(d.limit, both(d, "stifle_rom"), color=col, marker="o", ms=3, label=lab)
    axs[2].plot(d.limit, both(d, "hock_rom"), color=col, marker="o", ms=3)
    axs[3].plot(d.limit, both(d, "hip_rom"), color=col, marker="o", ms=3)
for ax, t, clin in zip(axs, ("Stifle angle window (deg)", "Stifle ROM (deg)", "Hock ROM (deg)", "Hip ROM (deg)"), (None, 28.77, 78.38, 35.53)):
    ax.set_title(t, fontsize=9); ax.set_xlabel("knee extension limit (deg)"); ax.invert_xaxis()
    if clin: ax.axhline(clin, color="k", ls="--", lw=0.8); ax.text(ax.get_xlim()[0], clin, " Grade III clinical", va="bottom", fontsize=6.5)
axs[0].axhspan(96.88, 125.1, color="k", alpha=0.12, lw=0); axs[0].legend(fontsize=6.5, frameon=False, loc="lower left")
fig.savefig(OUT / "sensitivity.pdf", bbox_inches="tight"); fig.savefig(OUT / "sensitivity.png", dpi=300, bbox_inches="tight"); plt.close(fig)

h2 = pd.read_csv(R / "swF2d1_summary.csv"); h3 = pd.read_csv(R / "swH3d1_summary.csv"); un = pd.read_csv(R / "swU2d1_summary.csv")
fig, axs = plt.subplots(1, 4, figsize=(12, 2.9))
for d, c, l in ((h2, "tab:red", "H2"), (h3, "tab:purple", "H3 pain avoidance")):
    axs[0].plot(d.grade, (d.peak_grf_LH + d.peak_grf_RH) / 2, marker="o", ms=3, color=c, label=l)
    axs[1].plot(d.grade, (d.LH_duty + d.RH_duty) / 2, marker="o", ms=3, color=c)
    axs[2].plot(d.grade, both(d, "stifle_rom"), marker="o", ms=3, color=c)
axs[0].set_title("Hind peak GRF (N)", fontsize=9); axs[1].set_title("Hind duty factor", fontsize=9); axs[2].set_title("Hind stifle ROM (deg)", fontsize=9)
axs[0].legend(fontsize=6.5, frameon=False)
axs[3].plot(un.grade, un.LH_stifle_rom, marker="o", ms=3, color="tab:orange", label="affected LH stifle ROM")
axs[3].plot(un.grade, un.RH_stifle_rom, marker="o", ms=3, color="tab:blue", label="unaffected RH stifle ROM")
axs[3].set_title("Unilateral (LH only)", fontsize=9); axs[3].legend(fontsize=6.5, frameon=False)
for ax in axs: ax.set_xlabel("grade"); ax.set_xticks([0, 2, 3, 4])
fig.savefig(OUT / "h3_unilateral.pdf", bbox_inches="tight"); fig.savefig(OUT / "h3_unilateral.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved")
