import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
X = pd.read_csv("results/ident_results/X_refit_eval.csv").set_index("source")
sim = ["sim fitA", "sim fitR2", "sim fitR3"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
for a, (col, name, clin) in zip(ax, (("stif_rom", "무릎", 48.9), ("hock_rom_mtp", "비절", 47.6))):
    vals = [X.loc[s, col] for s in sim]; labs = ["기존 모델\n(fitA)", "재학습 1차\n(fitR2)", "재학습 2차\n(fitR3)"]
    a.bar(labs, vals, color=["#9aa0a6", "#5b8def", "#2f6fdd"], width=0.6)
    for i, v in enumerate(vals): a.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=10)
    a.axhline(clin, color="k", ls="--", lw=1.2); a.text(2.45, clin + 1, f"임상 대조군 {clin}", ha="right", fontsize=9)
    mt = [X.loc[f"marked truth {c} (joint centres, incl. marking noise)", col] for c in ("a", "b")]; a.axhspan(min(mt), max(mt), color="#f4a261", alpha=0.35); a.text(-0.45, max(mt) + 1, f"leo 표시한 점 {min(mt):.0f}~{max(mt):.0f}", fontsize=9)
    a.set_title(f"{name} 각도의 가동범위 (도)"); a.set_ylim(0, 85)
plt.tight_layout(); plt.savefig("results/figures/refit_rom.png", dpi=150); print("saved")
