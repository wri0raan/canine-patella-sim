import sys
import numpy as np, pandas as pd
from scipy.signal import find_peaks
sys.path.insert(0, ".")
from analysis.identify_angles import sim_L, angle_series, lp

RUNS = [r for r in sys.argv[1:]] or ["fitB", "fitA", "trotS3", "trotS4", "trotS4_G2", "trotS4_G3"]
rows = []
for r in RUNS:
    try: L, fps = sim_L(f"results/ident/{r}.npz")
    except Exception as e: print("missing", r); continue
    import os; n = min(len(L["rump"]), int(float(os.environ.get("CUT", "1e9")) * fps)); m = slice(int(2 * fps), n)
    v = (L["rump"][n - 1, 0] - L["rump"][int(2 * fps), 0]) / ((n - int(2 * fps)) / fps)
    L = {k: v[:n] for k, v in L.items()}; A = angle_series(L, fps); hp = lp(L["hpaw"][:, 0] - L["rump"][:, 0], fps)[m]; pk, _ = find_peaks(hp, distance=int(0.12 * fps), prominence=0.3 * np.ptp(hp))
    f = (len(pk) - 1) / ((pk[-1] - pk[0]) / fps) if len(pk) > 2 else np.nan
    ht = L["rump"][m, 1]
    rows.append(dict(run=r, seconds=(n - int(2 * fps)) / fps, speed=v, stride_hz=f, rump_height_min=float(ht.min()), stif_mean=float(A["stif"][m].mean()), stif_rom=float(np.percentile(A["stif"][m], 97) - np.percentile(A["stif"][m], 3)),
                     hock_mean=float(A["hock"][m].mean()), hock_rom=float(np.percentile(A["hock"][m], 97) - np.percentile(A["hock"][m], 3))))
R = pd.DataFrame(rows)
for g in ("trotS4",):
    for s, k in (("_G2", "G2"), ("_G3", "G3")):
        if g + s in set(R.run) and g in set(R.run): R.loc[R.run == g + s, "d_stif_vs_healthy"] = float(R[R.run == g + s].stif_mean.iloc[0] - R[R.run == g].stif_mean.iloc[0])
for pre, h in (("fitA", "fitA"), ("fitB", "fitB")):
    for s in ("_G2", "_G3"):
        if pre + s in set(R.run) and h in set(R.run): R.loc[R.run == pre + s, "d_stif_vs_healthy"] = float(R[R.run == pre + s].stif_mean.iloc[0] - R[R.run == h].stif_mean.iloc[0])
R.to_csv("results/ident_results/P_trot_check.csv", index=False); pd.set_option("display.width", 220); print(R.round(2).to_string())
