import glob, json, itertools
import numpy as np, pandas as pd

L = {}
for p in glob.glob("data/real/leo5/label/user/labels/*.json"):
    d = json.load(open(p)); L[p.replace("\\", "/").split("/")[-1][:-5]] = d["pts"]
SIDE = lambda c: np.load(f"data/real/leo5/work/c{c}_crop.npz")
rows = []
for clip in ("a", "b"):
    cg = SIDE(clip); scale = float(cg["size"]) / cg["side"]
    tr = {}
    for m in ("hrnet_w32", "resnet_50"):
        z = np.load(f"data/real/leo5/work/{m}/{clip}_kp.npz"); tr[m] = (z["kp"], list(z["names"]))
    fz = np.load(f"data/real/leo5/work/fused/{clip}_tracks.npz"); names = list(fz["names"]); ix = {n: i for i, n in enumerate(names)}
    for key in sorted(k for k in L if k.startswith(clip + "_")):
        f = int(key.split("_")[1]); pts = L[key]; sc = scale[f]; x0, y0 = cg["x0"][f], cg["y0"][f]
        kp0 = tr["hrnet_w32"][0]; ixh = {n: i for i, n in enumerate(tr["hrnet_w32"][1])}
        back = float(np.linalg.norm(kp0[f, ixh["back_base"], :2] - kp0[f, ixh["back_end"], :2]))
        def cand(model):
            if model == "fused":
                out = {}
                for i, k in enumerate(("stif", "hock", "paw")):
                    for leg in ("hindA", "hindB"): out[(leg, k)] = (fz[leg][f, i, :2] - np.array([x0, y0])) * sc
                return out, ("hindA", "hindB")
            kp, nm = tr[model]; ixm = {n: i for i, n in enumerate(nm)}; out = {}
            for leg in ("left", "right"):
                for i, k in enumerate(("thai", "knee", "paw")): out[(leg, ("stif", "hock", "paw")[i])] = kp[f, ixm[f"back_{leg}_{k}"], :2]
            return out, ("left", "right")
        for model in ("hrnet_w32", "resnet_50", "fused"):
            out, legs = cand(model); best = None
            for perm in itertools.permutations(legs):
                err = 0.0; n = 0
                for side, leg in zip(("n", "f"), perm):
                    for k in ("stif", "hock", "paw"):
                        v = pts.get(side + k)
                        if v: err += np.linalg.norm(np.array(v) - out[(leg, k)]); n += 1
                if best is None or err < best[0]: best = (err, perm)
            for side, leg in zip(("n", "f"), best[1]):
                for k in ("stif", "hock", "paw"):
                    v = pts.get(side + k)
                    if v: rows.append(dict(clip=clip, frame=f, model=model, leg="near" if side == "n" else "far", joint=k, err_px=float(np.linalg.norm(np.array(v) - out[(leg, k)])), back_px=back))
        v = pts.get("tail")
        if v:
            for nm_ in ("back_end", "tail_base"):
                kp, nm = tr["hrnet_w32"]; rows.append(dict(clip=clip, frame=f, model="hrnet_w32", leg="body", joint="tail~" + nm_, err_px=float(np.linalg.norm(np.array(v) - kp[f, ixh[nm_], :2])), back_px=back))
R = pd.DataFrame(rows); R["err_pct_back"] = 100 * R.err_px / R.back_px; R["err_cm"] = R.err_pct_back / 100 * 45
R.to_csv("results/ident_results/U_label_error.csv", index=False)
S = R.groupby(["clip", "model", "leg", "joint"]).agg(n=("err_px", "size"), median_cm=("err_cm", "median"), mean_cm=("err_cm", "mean"), p90_cm=("err_cm", lambda x: np.percentile(x, 90)), median_pct_back=("err_pct_back", "median")).reset_index()
S.to_csv("results/ident_results/U_label_summary.csv", index=False); pd.set_option("display.width", 220); print(S.round(2).to_string())
print("\nlabel counts (marked / skipped):", {k: sum(1 for p in L.values() if p.get(k)) for k in ("tail", "hip", "nstif", "nhock", "npaw", "fstif", "fhock", "fpaw")}, "of", len(L))
