import glob, json, itertools
import numpy as np, pandas as pd

L = {p.replace("\\", "/").split("/")[-1][:-5]: json.load(open(p))["pts"] for p in glob.glob("data/real/leo5/label/user/labels/*.json")}
rows = []
for clip in ("a", "b"):
    cg = np.load(f"data/real/leo5/work/c{clip}_crop.npz"); scale = float(cg["size"]) / cg["side"]
    tr = {m: (lambda z: (z["kp"], {n: i for i, n in enumerate(z["names"])}))(np.load(f"data/real/leo5/work/{m}/{clip}_kp.npz")) for m in ("hrnet_w32", "resnet_50")}
    fz = np.load(f"data/real/leo5/work/fused/{clip}_tracks.npz"); fix = {n: i for i, n in enumerate(fz["names"])}
    for key in sorted(k for k in L if k.startswith(clip + "_")):
        f = int(key.split("_")[1]); pts = L[key]; sc = scale[f]; x0, y0 = cg["x0"][f], cg["y0"][f]; hip = pts.get("hip")
        for model in ("hrnet_w32", "resnet_50", "fused"):
            if model == "fused": legs = {l: np.stack([(fz[l][f, i, :2] - np.array([x0, y0])) * sc for i in range(3)]) for l in ("hindA", "hindB")}
            else:
                kp, ix = tr[model]; legs = {l: np.stack([kp[f, ix[f"back_{l}_{k}"], :2] for k in ("thai", "knee", "paw")]) for l in ("left", "right")}
            names = list(legs); best = None
            for perm in itertools.permutations(names):
                err = sum(np.linalg.norm(np.array(pts[s + k]) - legs[l][i]) for s, l in zip(("n", "f"), perm) for i, k in enumerate(("stif", "hock", "paw")) if pts.get(s + k)); best = (err, perm) if best is None or err < best[0] else best
            for s, l in zip(("n", "f"), best[1]):
                u = pts.get(s + "stif")
                if u and hip:
                    u = np.array(u); h = np.array(hip); ax = (h - u) / np.linalg.norm(h - u); pe = np.array([-ax[1], ax[0]]); d = legs[l][0] - u
                    rows.append(dict(clip=clip, frame=f, model=model, what=s + "_stifle", dx=d[0], dy=d[1], along_to_hip=d @ ax, perp=d @ pe, thigh_len=np.linalg.norm(h - u)))
        if pts.get("tail") and pts.get("hip"):
            kp, ix = tr["hrnet_w32"]; u = np.array(pts["tail"]); be = kp[f, ix["back_end"], :2]; bb = kp[f, ix["back_base"], :2]; ax = (be - bb) / np.linalg.norm(be - bb); pe = np.array([-ax[1], ax[0]]); d = be - u
            rows.append(dict(clip=clip, frame=f, model="hrnet_w32", what="tail_root(back_end)", dx=d[0], dy=d[1], along_to_hip=d @ ax, perp=d @ pe, thigh_len=np.nan))
R = pd.DataFrame(rows); cm = 45.0 / 270.0
back = []
for clip in ("a", "b"):
    z = np.load(f"data/real/leo5/work/hrnet_w32/{clip}_kp.npz"); ix = {n: i for i, n in enumerate(z["names"])}; back.append(np.median(np.linalg.norm(z["kp"][:, ix["back_base"], :2] - z["kp"][:, ix["back_end"], :2], axis=1)))
cm = 45.0 / float(np.mean(back)); print("back length (crop px, median over clips):", round(float(np.mean(back)), 1), "-> 1 crop px =", round(cm, 3), "cm")
for c in ("dx", "dy", "along_to_hip", "perp", "thigh_len"): R[c + "_cm"] = R[c] * cm
R.to_csv("results/ident_results/U_label_bias_rows.csv", index=False)
S = R.groupby(["model", "what"]).agg(n=("dx", "size"), dx_cm=("dx_cm", "mean"), dx_sd=("dx_cm", "std"), dy_cm=("dy_cm", "mean"), dy_sd=("dy_cm", "std"), along_hip_cm=("along_to_hip_cm", "mean"), along_sd=("along_to_hip_cm", "std"),
                                      perp_cm=("perp_cm", "mean"), perp_sd=("perp_cm", "std"), thigh_cm=("thigh_len_cm", "mean")).reset_index()
S.to_csv("results/ident_results/U_label_bias.csv", index=False); pd.set_option("display.width", 240); print(S.round(1).to_string())
