import json
from pathlib import Path
import numpy as np, pandas as pd
R = Path("results")
KEYS = ["speed", "freq_hz", "LH_stifle_rom", "LH_hock_rom", "LH_hip_rom", "LH_duty", "peak_grf_LH"]
rows = []
for d in sorted(R.glob("cmp*")):
    if not d.is_dir() or d.name.startswith(("cmp2_", "cmpB2_")) or not (d / "summary.json").exists(): continue
    s2 = R / d.name.replace("cmp_", "cmp2_").replace("cmpB_", "cmpB2_")
    if not (s2 / "summary.json").exists(): continue
    a, b = json.load(open(d / "summary.json")), json.load(open(s2 / "summary.json"))
    rows.append(dict(cond=d.name, body="bone" if "cmpB" in d.name else "original", fell1=a["fell"], fell2=b["fell"], **{f"{k}_s1": a.get(k, np.nan) for k in KEYS}, **{f"{k}_s2": b.get(k, np.nan) for k in KEYS}))
C = pd.DataFrame(rows); C.to_csv(R / "composition_seed_compare.csv", index=False)
rows = []
for t1, t2 in (("swBF2s1", "swBF2s2"), ("swBM1s1", "swBM1s2"), ("swBF1s1", "swBF1s2")):
    a, b = pd.read_csv(R / f"{t1}_summary.csv"), pd.read_csv(R / f"{t2}_summary.csv")
    for g in sorted(a.grade.unique()):
        x, y = a[a.grade == g].iloc[0], b[b.grade == g].iloc[0]
        rows.append(dict(sweep=t1[:-1], grade=int(g), **{f"{k}_s1": x[k] for k in KEYS}, **{f"{k}_s2": y[k] for k in KEYS}))
S = pd.DataFrame(rows); S.to_csv(R / "sweep_seed_compare.csv", index=False)
pd.set_option("display.width", 220)
for name, T in (("composition", C), ("sweeps", S)):
    print("\n" + name)
    sp = {k: np.abs(T[f"{k}_s1"] - T[f"{k}_s2"]) for k in KEYS}
    print(pd.DataFrame({k: dict(mean_abs_diff=v.mean(), max_abs_diff=v.max(), mean_value=0.5 * (T[f"{k}_s1"] + T[f"{k}_s2"]).mean()) for k, v in sp.items()}).T.round(3).to_string())
print(C.round(2).to_string()); print(S.round(2).to_string())
