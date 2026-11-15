import sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, ".")
from analysis.virtual_markers import sim_full
from analysis.identify_angles import lp
from scipy.signal import butter, filtfilt

FPSV = 120.0; b6, a6 = butter(3, 6.0 / (FPSV / 2)); lpv = lambda x: filtfilt(b6, a6, x, axis=0)
unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
ang = lambda p, q, r: np.degrees(np.arccos(np.clip((unit(p - q) * unit(r - q)).sum(-1), -1, 1)))


def sim_stifle(run, caud=0.38, dors=0.22):
    path = f"results/ident/{run}.npz"; L, ax, fps = sim_full(path); z = np.load(path, allow_pickle=True); names = list(z["names"][: len(z["names"]) - 4]); P = z["pos"]; ix = {k: i for i, k in enumerate(names)}
    xz = lambda k: P[:, ix[k]][:, [0, 2]]; hip, st, hk = xz("upper_leg_L"), xz("lower_leg_L"), xz("foot_L"); dd = np.stack([-ax[:, 1], ax[:, 0]], 1); dd = dd if dd[:, 1].mean() > 0 else -dd
    T = np.median(np.linalg.norm(st - hk, axis=1)); m = slice(int(2 * fps), None)
    return float(ang(lp(hip + T * (caud * ax + dors * dd), fps)[m], lp(st, fps)[m], lp(hk, fps)[m]).mean())


base = {r: sim_stifle(r) for r in ("fitA", "fitB")}; g2 = {r: sim_stifle(r) for r in ("fitA_G2", "fitB_G2")}; g3 = {"fitB_G3": sim_stifle("fitB_G3")}
d2 = np.mean([g2["fitA_G2"] - base["fitA"], g2["fitB_G2"] - base["fitB"]]); d3 = g3["fitB_G3"] - base["fitB"]; mu_n = float(np.mean(list(base.values())))
MU = {"normal": mu_n, "abnormal": mu_n + d2, "severe": mu_n + d3}
print("sim stifle mean (X-ray rump):", {k: round(v, 1) for k, v in base.items()}, "| grade changes G2 %+.1f, G3 %+.1f" % (d2, d3)); print("class means:", {k: round(v, 1) for k, v in MU.items()})
sd_sim = float(np.std([sim_stifle("fitA", 0.38 + a, 0.22 + b) for a in (-0.05, 0, 0.05) for b in (-0.05, 0, 0.05)]))

L = {p.replace("\\", "/").split("/")[-1][:-5]: json.load(open(p))["pts"] for p in glob.glob("data/real/leo5/label/user/labels/*.json")}
def offs(clip):
    z = np.load(f"data/real/leo5/work/hrnet_w32/{clip}_kp.npz"); ix = {n: i for i, n in enumerate(z["names"])}; v = []
    for key in sorted(k for k in L if k.startswith(clip + "_")):
        f = int(key.split("_")[1]); tl = L[key].get("tail")
        if tl: be, bb = z["kp"][f, ix["back_end"], :2], z["kp"][f, ix["back_base"], :2]; v.append(float((np.array(tl) - be) @ unit(be - bb)) / 126.8)
    return np.array(v)
oa = offs("a"); c_fit = float(oa.mean()); c_se = float(oa.std(ddof=1) / np.sqrt(len(oa)))
def video_stifle(clip, leg, c):
    fz = np.load(f"data/real/leo5/work/fused/{clip}_tracks.npz"); ix = {n: i for i, n in enumerate(fz["names"])}; ins = np.nonzero(fz["inside"])[0]; a0, b0 = ins[0], ins[-1] + 1
    body = fz["body"][a0:b0]; be, bb = lpv(body[:, ix["back_end"], :2]), lpv(body[:, ix["back_base"], :2]); X = fz[leg][a0:b0]; st, hk = lpv(X[:, 0, :2]), lpv(X[:, 1, :2]); Tv = np.median(np.linalg.norm(st - hk, axis=1))
    return float(ang(be + c * Tv * unit(be - bb), st, hk).mean())
obs = {(cl, lg): video_stifle(cl, lg, c_fit) for cl in ("a", "b") for lg in ("hindA", "hindB")}
sd_c = float(np.std([video_stifle("a", "hindA", c_fit + d) for d in (-c_se, 0, c_se)])); sd_trk = 3.0
vals = np.array(list(obs.values())); clip_means = {cl: float(np.mean([obs[(cl, lg)] for lg in ("hindA", "hindB")])) for cl in ("a", "b")}
sd_meas = float(np.std(list(clip_means.values()), ddof=1)); sd_leg = float(vals.std(ddof=1))
SD = float(np.sqrt(sd_meas ** 2 + sd_sim ** 2 + sd_c ** 2 + sd_trk ** 2))
print(f"video stifle (tail_est rump): per leg {[round(v, 1) for v in vals]}, clip means {clip_means}, c_fit {c_fit:.2f} T (se {c_se:.2f})")
print(f"sd: between clips {sd_meas:.1f}, between legs {sd_leg:.1f}, sim definition {sd_sim:.1f}, rump offset {sd_c:.1f}, tracker {sd_trk:.1f} -> total {SD:.1f} deg")


def post(x, sd=SD):
    lik = {k: norm.pdf(x, m, sd) for k, m in MU.items()}; s = sum(lik.values()); return {k: v / s for k, v in lik.items()}
rows = []
def add(kind, name, x, truth):
    p = post(x); call = max(p, key=p.get); rows.append(dict(test=kind, case=name, truth=truth, stifle_mean=x, call=call, ok=call == truth, pmax=p[call], p_normal=p["normal"], p_abnormal=p["abnormal"], p_severe=p["severe"]))
for cl, x in clip_means.items(): add("real healthy (clip mean)", f"clip {cl}", x, "normal")
for (cl, lg), x in obs.items(): add("real healthy (single leg)", f"clip {cl} {lg}", x, "normal")
for cl, x in clip_means.items():
    add("semi-synthetic", f"clip {cl} + G2 change", x + d2, "abnormal"); add("semi-synthetic", f"clip {cl} + G3 change", x + d3, "severe")
rng = np.random.default_rng(0)
for k, truth in (("fitA", "normal"), ("fitB", "normal"), ("fitA_G2", "abnormal"), ("fitB_G2", "abnormal"), ("fitB_G3", "severe")):
    x0 = sim_stifle(k)
    for i in range(20): add("sim runs + noise (sd 5)", f"{k}#{i}", x0 + rng.normal(0, 5.0), truth)
R = pd.DataFrame(rows); R.to_csv("results/ident_results/W_judge_model_based.csv", index=False); pd.set_option("display.width", 200)
print("\n" + R[R.test.str.startswith(("real", "semi"))][["test", "case", "truth", "stifle_mean", "call", "pmax", "p_normal", "p_abnormal", "p_severe"]].round(2).to_string(index=False))
S = R.groupby(["test", "truth"]).agg(n=("ok", "size"), acc=("ok", "mean"), pmax=("pmax", "mean")).round(2); print("\n" + S.to_string())
