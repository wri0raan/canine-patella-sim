import glob, json, itertools
import numpy as np, pandas as pd

OFF = json.load(open("results/ident_results/V_xray_offsets.json"))["mean"]
L = {p.replace("\\", "/").split("/")[-1][:-5]: json.load(open(p))["pts"] for p in glob.glob("data/real/leo5/label/user/labels/*.json")}
unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
ang = lambda p, q, r: float(np.degrees(np.arccos(np.clip(unit(np.array(p) - q) @ unit(np.array(r) - q), -1, 1))))
Tleg = {}
for s in ("n", "f"):
    v = [np.linalg.norm(np.array(P[s + "stif"]) - np.array(P[s + "hock"])) for P in L.values() if P.get(s + "stif") and P.get(s + "hock")]; Tleg[s] = float(np.mean(v))
print("mean marked tibia (crop px):", {k: round(v, 1) for k, v in Tleg.items()}, "| offsets (x tibia):", {k: round(v, 3) for k, v in OFF.items()})


def convert(P, s):
    st, hk, pw = (np.array(P[s + k], float) for k in ("stif", "hock", "paw")); T = Tleg[s]
    sh = unit(hk - st); n = np.array([-sh[1], sh[0]]); n = n if n[0] > 0 else -n
    st_c = st - OFF["knee_cran"] * T * n
    m = unit(pw - hk); c = np.array([-m[1], m[0]]); c = c if c[0] < 0 else -c
    hk_c = hk + OFF["heel_prox"] * T * m - OFF["heel_caud"] * T * c
    return st_c, hk_c, pw


rows, ang_rows = [], []
fem = [np.linalg.norm(np.array(P["hip"]) - np.array(P["nstif"])) / Tleg["n"] for P in L.values() if P.get("hip") and P.get("nstif")]
print(f"raw hip mark femur/tibia = {np.mean(fem):.2f} (radiograph 0.91): the hip mark is NOT the hip joint centre")
for clip in ("a", "b"):
    cg = np.load(f"data/real/leo5/work/c{clip}_crop.npz"); scale = float(cg["size"]) / cg["side"]; fz = np.load(f"data/real/leo5/work/fused/{clip}_tracks.npz")
    zh = np.load(f"data/real/leo5/work/hrnet_w32/{clip}_kp.npz"); ixh = {n: i for i, n in enumerate(zh["names"])}
    for key in sorted(k for k in L if k.startswith(clip + "_")):
        f = int(key.split("_")[1]); P = L[key]; sc = scale[f]; org = np.array([cg["x0"][f], cg["y0"][f]])
        for model in ("fused", "hrnet_w32"):
            if model == "fused": legs = {l: np.stack([(fz[l][f, i, :2] - org) * sc for i in range(3)]) for l in ("hindA", "hindB")}
            else: legs = {l: np.stack([zh["kp"][f, ixh[f"back_{l}_{k}"], :2] for k in ("thai", "knee", "paw")]) for l in ("left", "right")}
            best = None
            for perm in itertools.permutations(legs):
                e = sum(np.linalg.norm(np.array(P[s + k]) - legs[l][i]) for s, l in zip(("n", "f"), perm) for i, k in enumerate(("stif", "hock", "paw")) if P.get(s + k)); best = (e, perm) if best is None or e < best[0] else best
            for s, l in zip(("n", "f"), best[1]):
                if not all(P.get(s + k) for k in ("stif", "hock", "paw")): continue
                st_c, hk_c, pw = convert(P, s); tr = legs[l]
                rows.append(dict(clip=clip, frame=f, model=model, leg=s, stif_raw=np.linalg.norm(tr[0] - np.array(P[s + "stif"])) / Tleg[s], stif_conv=np.linalg.norm(tr[0] - st_c) / Tleg[s],
                                 hock_raw=np.linalg.norm(tr[1] - np.array(P[s + "hock"])) / Tleg[s], hock_conv=np.linalg.norm(tr[1] - hk_c) / Tleg[s]))
                if P.get("tail"):
                    tl = np.array(P["tail"]); ang_rows.append(dict(clip=clip, frame=f, model=model, leg=s, truth_stif_raw=ang(tl, P[s + "stif"], P[s + "hock"]), truth_stif_conv=ang(tl, st_c, hk_c), track_stif=ang(tl, tr[0], tr[1]),
                                                                  truth_hock_raw=ang(P[s + "stif"], P[s + "hock"], P[s + "paw"]), truth_hock_conv=ang(st_c, hk_c, pw), track_hock=ang(tr[0], tr[1], tr[2])))
E = pd.DataFrame(rows); E.to_csv("results/ident_results/V_convert_error.csv", index=False); A = pd.DataFrame(ang_rows); A.to_csv("results/ident_results/V_convert_angles.csv", index=False)
pd.set_option("display.width", 200)
print("\nposition error, median, fraction of tibia (raw marks vs joint-centre conversion):"); print(E.groupby(["clip", "model"])[["stif_raw", "stif_conv", "hock_raw", "hock_conv"]].median().round(3))
for (c, m), g in A.groupby(["clip", "model"]):
    print(f"\nclip {c} {m}: truth mean stif raw {g.truth_stif_raw.mean():.0f} / conv {g.truth_stif_conv.mean():.0f} | track {g.track_stif.mean():.0f} || hock raw {g.truth_hock_raw.mean():.0f} / conv {g.truth_hock_conv.mean():.0f} | track {g.track_hock.mean():.0f}")
    for j in ("stif", "hock"):
        for k in ("raw", "conv"):
            e = g[f"track_{j}"] - g[f"truth_{j}_{k}"]; print(f"   {j:5s} vs {k:4s} truth: bias {e.mean():+6.1f}  |err| median {np.median(np.abs(e)):5.1f}  sd {e.std():5.1f}  r {np.corrcoef(g[f'track_{j}'], g[f'truth_{j}_{k}'])[0, 1]:+.2f}")
