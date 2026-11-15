import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, ".")

MODELS = [m for m in ("hrnet_w32", "rtmpose_x", "resnet_50") if __import__("os").path.exists(f"data/real/leo5/work/{m}/a_kp.npz")]
SEGS = {"tibia": ("back_{s}_thai", "back_{s}_knee"), "metatarsus": ("back_{s}_knee", "back_{s}_paw"), "forearm": ("front_{s}_thai", "front_{s}_knee"), "fore_paw": ("front_{s}_knee", "front_{s}_paw")}


def load(model, clip):
    z = np.load(f"data/real/leo5/work/{model}/{clip}_kp.npz"); c = np.load(f"data/real/leo5/work/c{clip}_crop.npz")
    kp = z["kp"].copy(); n = min(len(kp), len(c["x0"])); kp = kp[:n]; sc = c["side"][:n] / float(c["size"])
    kp[:, :, 0] = kp[:, :, 0] * sc[:, None] + c["x0"][:n, None]; kp[:, :, 1] = kp[:, :, 1] * sc[:, None] + c["y0"][:n, None]
    box = c["box"][:n]; inside = (box[:, 0] > 30) & (box[:, 2] < 3840 - 30)
    return kp, list(z["names"]), inside


rows, T = [], {}
for clip in ("a", "b"):
    for m in MODELS:
        kp, names, inside = load(m, clip); ix = {k: i for i, k in enumerate(names)}; T[(m, clip)] = (kp, ix, inside)
for (m, clip), (kp, ix, inside) in T.items():
    C = lambda nm: kp[:, ix[nm], 2]; P = lambda nm: kp[:, ix[nm], :2]
    hs = max(("left", "right"), key=lambda s: sum(np.median(C(f"back_{s}_{k}")[inside]) for k in ("thai", "knee", "paw"))); fs = max(("left", "right"), key=lambda s: sum(np.median(C(f"front_{s}_{k}")[inside]) for k in ("thai", "knee", "paw")))
    dog = np.median(np.linalg.norm(P("back_base") - P("back_end"), axis=1)[inside]); row = dict(model=m, clip=clip, frames=int(inside.sum()), hind_side=hs, fore_side=fs, back_px=dog)
    for nm, (p, q) in SEGS.items():
        s = hs if nm in ("tibia", "metatarsus") else fs; ok = inside & (C(p.format(s=s)) > 0.3) & (C(q.format(s=s)) > 0.3)
        d = np.linalg.norm(P(p.format(s=s)) - P(q.format(s=s)), axis=1)[ok]; row[f"cv_{nm}"] = float(d.std() / d.mean()) if ok.sum() > 20 else np.nan
    for nm in ("knee", "paw", "thai"):
        x = P(f"back_{hs}_{nm}")[inside]; row[f"jit_{nm}"] = float(np.sqrt(np.mean(np.sum(np.diff(x, 2, axis=0) ** 2, axis=1))) / dog)
    row["conf_hind"] = float(np.mean([np.median(C(f"back_{hs}_{k}")[inside]) for k in ("thai", "knee", "paw")])); rows.append(row)
Q = pd.DataFrame(rows)
ag = []
for clip in ("a", "b"):
    for m1, m2 in itertools.combinations(MODELS, 2):
        k1, i1, in1 = T[(m1, clip)]; k2, i2, in2 = T[(m2, clip)]; hs = Q[(Q.model == m1) & (Q["clip"] == clip)].hind_side.iloc[0]; ok = in1 & in2; dog = Q[(Q.model == m1) & (Q["clip"] == clip)].back_px.iloc[0]
        r = dict(clip=clip, pair=f"{m1} vs {m2}")
        for nm in ("thai", "knee", "paw"): r[f"dist_{nm}"] = float(np.median(np.linalg.norm(k1[:, i1[f"back_{hs}_{nm}"], :2] - k2[:, i2[f"back_{hs}_{nm}"], :2], axis=1)[ok]) / dog)
        ag.append(r)
pd.set_option("display.width", 250); print(Q.round(3).to_string()); print(); A = pd.DataFrame(ag); print(A.round(3).to_string())
Q.to_csv("results/ident_results/S_leo5_qc.csv", index=False); A.to_csv("results/ident_results/S_leo5_agreement.csv", index=False)
