import numpy as np, pandas as pd
rows = []
for model in ("hrnet_w32", "resnet_50", "fused"):
    for clip in ("a", "b"):
        z = np.load(f"data/real/leo5/work/{model}/{clip}_tracks.npz"); ins = z["inside"]; names = list(z["names"]); ix = {n: i for i, n in enumerate(names)}; body = z["body"]
        dog = np.median(np.linalg.norm(body[:, ix["back_base"], :2] - body[:, ix["back_end"], :2], axis=1)[ins])
        for leg, key in (("hindA", "hind"), ("hindB", "hind"), ("foreA", "fore"), ("foreB", "fore")):
            X = z[leg]; c = X[:, :, 2]; r = dict(model=model, clip=clip, leg=leg, frames=int(ins.sum()), conf=float(np.median(c[ins])))
            for nm, (i, j) in {"seg1": (0, 1), "seg2": (1, 2)}.items():
                ok = ins & (c[:, i] > 0.3) & (c[:, j] > 0.3); d = np.linalg.norm(X[:, i, :2] - X[:, j, :2], axis=1)[ok]; r[f"cv_{nm}"] = float(d.std() / d.mean()) if ok.sum() > 20 else np.nan
            for nm, i in {"thai": 0, "knee": 1, "paw": 2}.items(): r[f"jit_{nm}"] = float(np.sqrt(np.mean(np.sum(np.diff(X[ins][:, i, :2], 2, axis=0) ** 2, axis=1))) / dog)
            rows.append(r)
Q = pd.DataFrame(rows); pd.set_option("display.width", 220); print(Q.round(3).to_string()); Q.to_csv("results/ident_results/S_leo5_qc2.csv", index=False)
