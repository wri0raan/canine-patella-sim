import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.virtual_markers import sim_full
from analysis.identify_angles import angle_series

unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)


def outward(P, J, D, d):
    b = unit(unit(P - J) + unit(D - J)); return J - d * b


def variant(L, ax, scale, lam, ds, dh):
    L2 = dict(L); L2["rump"] = L["rump"] + lam * scale * ax
    st = outward(L["rump"], L["stifle"], L["hock"], ds * scale); hk = outward(L["stifle"], L["hock"], L["hpaw"], dh * scale)
    L2["stifle"], L2["hock"] = st, hk; return L2


def stats(path, lam, ds, dh):
    L, ax, fps = sim_full(path); scale = float(np.median(np.linalg.norm(L["stifle"] - L["hock"], axis=1)))
    A = angle_series(variant(L, ax, scale, lam, ds, dh), fps); m = slice(int(2 * fps), len(A["stif"]))
    return float(A["stif"][m].mean()), float(np.percentile(A["stif"][m], 97) - np.percentile(A["stif"][m], 3))


rows = []
for lam, ds, dh in itertools.product((0.0, 0.2, 0.4, 0.7), (0.0, 0.08, 0.15), (0.0, 0.08)):
    r = {c: stats(f"results/ident/{c}.npz", lam, ds, dh) for c in ("fitA", "fitA_G2", "fitB", "fitB_G2", "fitB_G3")}
    rows.append(dict(lam=lam, d_stifle=ds, d_hock=dh, normal_A=r["fitA"][0], normal_B=r["fitB"][0], G2_A=r["fitA_G2"][0] - r["fitA"][0], G2_B=r["fitB_G2"][0] - r["fitB"][0], G3_B=r["fitB_G3"][0] - r["fitB"][0],
                     rom_ratio_G2_B=r["fitB_G2"][1] / r["fitB"][1], rom_ratio_G3_B=r["fitB_G3"][1] / r["fitB"][1]))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/O_marker_sensitivity.csv", index=False)
pd.set_option("display.width", 220); print(R.round(2).to_string())
g2 = pd.concat([R.G2_A, R.G2_B]); print("\ngrade 2 change over the grid (A and B): mean %.1f sd %.1f min %.1f max %.1f" % (g2.mean(), g2.std(), g2.min(), g2.max()))
print("grade 3 change over the grid (B): mean %.1f sd %.1f min %.1f max %.1f" % (R.G3_B.mean(), R.G3_B.std(), R.G3_B.min(), R.G3_B.max()))
nn = pd.concat([R.normal_A, R.normal_B]); print("normal stifle mean over the grid: %.0f .. %.0f (sd %.0f)" % (nn.min(), nn.max(), nn.std()))
