import glob, json, itertools
import numpy as np, pandas as pd

L = {p.replace("\\", "/").split("/")[-1][:-5]: json.load(open(p))["pts"] for p in glob.glob("data/real/leo5/label/user/labels/*.json")}


def ang(p, q, r):
    u, v = np.array(p) - q, np.array(r) - q; return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))


rec = []
for clip in ("a", "b"):
    cg = np.load(f"data/real/leo5/work/c{clip}_crop.npz"); scale = float(cg["size"]) / cg["side"]
    fz = np.load(f"data/real/leo5/work/fused/{clip}_tracks.npz"); z = np.load(f"data/real/leo5/work/hrnet_w32/{clip}_kp.npz"); ix = {n: i for i, n in enumerate(z["names"])}
    for key in sorted(k for k in L if k.startswith(clip + "_")):
        f = int(key.split("_")[1]); pts = L[key]; sc = scale[f]; org = np.array([cg["x0"][f], cg["y0"][f]])
        legs = {l: (fz[l][f, :, :2] - org) * sc for l in ("hindA", "hindB")}; best = None
        for perm in itertools.permutations(legs):
            err = sum(np.linalg.norm(np.array(pts[s + k]) - legs[l][i]) for s, l in zip(("n", "f"), perm) for i, k in enumerate(("stif", "hock", "paw")) if pts.get(s + k)); best = (err, perm) if best is None or err < best[0] else best
        for s, l in zip(("n", "f"), best[1]):
            if all(pts.get(s + k) for k in ("stif", "hock", "paw")) and pts.get("tail"):
                rec.append(dict(clip=clip, frame=f, leg=s, tail=np.array(pts["tail"]), be=z["kp"][f, ix["back_end"], :2], u_st=np.array(pts[s + "stif"]), u_hk=np.array(pts[s + "hock"]), u_pw=np.array(pts[s + "paw"]), m_st=legs[l][0], m_hk=legs[l][1], m_pw=legs[l][2]))
tib = np.mean([np.linalg.norm(r["u_st"] - r["u_hk"]) for r in rec])
A = [r for r in rec if r["clip"] == "a"]; B = [r for r in rec if r["clip"] == "b"]
d = np.array([r["m_hk"] - r["m_st"] for r in A]); u = np.array([r["u_st"] - r["m_st"] for r in A]); beta = float((u * d).sum() / (d * d).sum())
def err(R, b=0.0): return np.array([np.linalg.norm(r["u_st"] - (r["m_st"] + b * (r["m_hk"] - r["m_st"]))) for r in R]) / tib
out = pd.DataFrame([dict(set="train a (fit)", n=len(A), err_before=np.median(err(A)), err_after=np.median(err(A, beta))), dict(set="test b (held out)", n=len(B), err_before=np.median(err(B)), err_after=np.median(err(B, beta)))])
print("beta =", round(beta, 3), "| marked tibia", round(tib, 1), "px; stifle error in tibia lengths (median):"); print(out.round(3).to_string(index=False)); out.assign(beta=beta).to_csv("results/ident_results/U_correct.csv", index=False)
rows = []
for r in rec:
    st_c = r["m_st"] + beta * (r["m_hk"] - r["m_st"])
    rows.append(dict(clip=r["clip"], frame=r["frame"], leg=r["leg"], truth_stif=ang(r["tail"], r["u_st"], r["u_hk"]), track_stif=ang(r["tail"], r["m_st"], r["m_hk"]), corr_stif=ang(r["tail"], st_c, r["m_hk"]),
                     truth_stif_be=ang(r["be"], r["u_st"], r["u_hk"]), truth_hock=ang(r["u_st"], r["u_hk"], r["u_pw"]), track_hock=ang(r["m_st"], r["m_hk"], r["m_pw"]), corr_hock=ang(st_c, r["m_hk"], r["m_pw"])))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/U_angle_check.csv", index=False)
for c in ("a", "b"):
    g = R[R["clip"] == c]; print(f"\nclip {c} ({'fit' if c == 'a' else 'TEST'}) n={len(g)}  angle error vs marked truth (deg): median |err|, mean signed, corr with truth")
    for j in ("stif", "hock"):
        for k in ("track", "corr"):
            e = g[f"{k}_{j}"] - g[f"truth_{j}"]; print(f"  {j:5s} {k:5s} |err| median {np.median(np.abs(e)):5.1f}  bias {e.mean():+6.1f}  sd {e.std():5.1f}  r {np.corrcoef(g[f'{k}_{j}'], g[f'truth_{j}'])[0, 1]:+.2f}")
    print("  truth means: stifle %.1f (tail-rump), %.1f (back_end rump), hock %.1f; truth sd stifle %.1f hock %.1f" % (g.truth_stif.mean(), g.truth_stif_be.mean(), g.truth_hock.mean(), g.truth_stif.std(), g.truth_hock.std()))
