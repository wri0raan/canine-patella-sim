import json
import numpy as np
R = {
    "xr4": dict(tip=(1425, 1965), hock=(1343, 2071), pat=(806, 1200), stifle=(927, 1256), plateau=(945, 1343), hip=(1244, 585)),
    "xr6": dict(tip=(1470, 2110), hock=(1388, 2200), pat=(758, 1385), stifle=(894, 1400), plateau=(918, 1506), hip=(1065, 647)),
}
out = {}
for k, v in R.items():
    P = {n: np.array(p, float) for n, p in v.items()}; T = np.linalg.norm(P["hock"] - P["plateau"]); fem = np.linalg.norm(P["hip"] - P["stifle"])
    prox = (P["hock"][1] - P["tip"][1]) / T; caud = (P["tip"][0] - P["hock"][0]) / T
    ax = (P["hip"] - P["stifle"]) / np.linalg.norm(P["hip"] - P["stifle"]); nrm = np.array([-ax[1], ax[0]]); nrm = nrm if nrm[0] < 0 else -nrm
    cran = float((P["stifle"] - P["pat"]) @ -nrm) / T if False else float((P["pat"] - P["stifle"]) @ nrm) / T
    out[k] = dict(T_px=float(T), femur_over_tibia=float(fem / T), heel_prox=float(prox), heel_caud=float(caud), knee_cran=float(cran))
    print(k, {a: round(b, 3) for a, b in out[k].items()})
m = {a: float(np.mean([out[k][a] for k in out])) for a in ("femur_over_tibia", "heel_prox", "heel_caud", "knee_cran")}; out["mean"] = m; print("mean", {a: round(b, 3) for a, b in m.items()})
json.dump(out, open("results/ident_results/V_xray_offsets.json", "w"), indent=1)
