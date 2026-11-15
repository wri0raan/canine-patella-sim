import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, ".")
from analysis.video_style_features import video_landmarks, features
from analysis.identify_angles import video_L, angle_series, feats, sim_L, windows_of
import analysis.leo_segments as LS

LIB = "results/judge_library.json"
KEYPOINT_SD = 17.0
SIM_STAND = 148.0
CLASSES = ["normal", "abnormal", "severe"]
WALK = (0.4, 0.8)
LEO_DELTA = {"normal": 0.0, "abnormal": -9.2, "severe": -22.3}
LEO_MODEL_SD = 2.0
BASE_SD_FLOOR = 2.5


def video_offset():
    vals = []
    for f, fpp in (("data/real/leo2/work/photos_kp.npz", 2), ("data/real/leo2/work/photo9_kp.npz", 1)):
        z = np.load(f, allow_pickle=True); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
        for k in range(0, kp.shape[0], fpp):
            S = kp[k]; g = lambda n: S[ix[n], :2]; c = lambda n: S[ix[n], 2]; face = np.sign(g("nose")[0] - g("tail_base")[0]) or 1
            for hs in ("back_left", "back_right"):
                if min(c(n) for n in (f"{hs}_thai", f"{hs}_knee", "back_end")) < 0.5: continue
                P = lambda n: np.array([[g(n)[0] * face, -g(n)[1]]])
                from analysis.video_style_features import angle2
                vals.append(angle2(P("back_end"), P(f"{hs}_thai"), P(f"{hs}_knee"))[0])
    return SIM_STAND - float(np.median(vals)), len(vals)


def build_library():
    if os.path.exists(LIB): return json.load(open(LIB))
    D = pd.read_csv("results/ident_results/dataset.csv"); gr = D[(D.hyp != "body") | D.cond.str.match(r"idg_M(10|08)F00_S")]; lab = dict(zip(gr.run, gr.grade)); rows = []
    for r in sorted(gr.run.unique()):
        try: L, fps = sim_L(f"results/ident/{r}.npz")
        except Exception: continue
        for w in windows_of(angle_series(L, fps), fps): rows.append(dict(run=r, grade=lab[r], stif_mean=w["stif_mean"]))
    for ch in "AB":
        for suf, g in (("", 0), ("_G2", 2), ("_G3", 3)):
            try: L, fps = sim_L(f"results/ident/fit{ch}{suf}.npz")
            except Exception: continue
            for w in windows_of(angle_series(L, fps), fps): rows.append(dict(run=f"fit{ch}{suf}", grade=g, stif_mean=w["stif_mean"]))
    F = pd.DataFrame(rows); F["cls"] = np.where(F.grade == 0, "normal", np.where(F.grade == 2, "abnormal", "severe"))
    per_run = F.groupby(["cls", "run"]).stif_mean.mean().reset_index()
    lib = {c: dict(mean=float(per_run[per_run.cls == c].stif_mean.mean()), sd=float(per_run[per_run.cls == c].stif_mean.std(ddof=1)), n_runs=int((per_run.cls == c).sum())) for c in CLASSES}
    json.dump(lib, open(LIB, "w"), indent=1); return lib


def clip_features(kp, names, a, b, fps, weight):
    tib = 0.127 * (weight / 7.0) ** (1 / 3)
    V = video_landmarks(kp, names, a, b, tibia_m=tib); f = features(V, fps)
    A = angle_series(video_L(kp, names, a, b), fps); af = feats(A, 0, len(A["hock"]))
    return dict(speed=f["speed"], stride_hz=f["stride_hz"], stif_mean=af["stif_mean"], hock_mean=af["hock_mean"])


def judge(kp, names, a, b, fps, weight, baseline=None, leo_model=False, shift=0.0):
    lib = build_library(); off, nphoto = video_offset(); f = clip_features(kp, names, a, b, fps, weight); notes = []
    sp = f["speed"]; speed_cat = "느림" if sp < 0.6 else "보통" if sp <= 1.0 else "빠름"
    if not (WALK[0] <= sp <= WALK[1]): notes.append(f"속도 {sp:.2f} m/s: 모델 걷기 범위({WALK[0]}-{WALK[1]}) 밖, 신뢰 낮음")
    if not (6 <= weight <= 12): notes.append(f"체중 {weight} kg: 모델 몸 범위(6-12 kg) 밖")
    if baseline:
        bm = np.array([clip_features(k, n, x, y, fp, weight)["stif_mean"] for (k, n, x, y, fp) in baseline]); d = f["stif_mean"] + shift - bm.mean()
        mode = f"기준선 비교 (기준 {len(bm)}개, 변화 {d:+.1f}도)"
        if leo_model:
            sdb = max(float(bm.std(ddof=1)) if len(bm) > 1 else 6.0, BASE_SD_FLOOR) * np.sqrt(1 + 1 / len(bm))
            mu = dict(LEO_DELTA); x = d; sds = {c: np.sqrt(sdb ** 2 + (0 if c == "normal" else LEO_MODEL_SD ** 2)) for c in CLASSES}; mode += ", leo 모델 보정"
        else:
            sdb = max(float(bm.std(ddof=1)) if len(bm) > 1 else 6.0, 4.0)
            sd = np.sqrt(2 * sdb ** 2)
            mu = {c: lib[c]["mean"] - lib["normal"]["mean"] for c in CLASSES}; x = d
            sds = {c: np.sqrt(sd ** 2 + lib[c]["sd"] ** 2 + lib["normal"]["sd"] ** 2) for c in CLASSES}
    else:
        x = f["stif_mean"] + off; mu = {c: lib[c]["mean"] for c in CLASSES}; sds = {c: np.sqrt(lib[c]["sd"] ** 2 + KEYPOINT_SD ** 2) for c in CLASSES}
        mode = f"절대값 비교 (영상 무릎 각도 {f['stif_mean']:.0f} + 보정 {off:+.0f} = {x:.0f}도)"; notes.append("기준선 없음: 키포인트 흩어짐 17도가 등급 간 차이보다 커서 신뢰 낮음")
    like = np.array([norm.pdf(x, mu[c], sds[c]) for c in CLASSES]); p = like / like.sum(); call = CLASSES[int(p.argmax())]
    return dict(call=call, p_normal=p[0], p_abnormal=p[1], p_severe=p[2], speed=sp, speed_cat=speed_cat, stif_mean=f["stif_mean"], mode=mode, reliable=bool(p.max() >= 0.8), notes="; ".join(notes))


def load(path):
    z = np.load(path, allow_pickle=True); return z["kp"], list(z["names"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--kp", required=True); ap.add_argument("--weight", type=float, required=True); ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--seg", nargs=2, type=float); ap.add_argument("--baseline", default=""); a = ap.parse_args()
    kp, names = load(a.kp); ix = {n: i for i, n in enumerate(names)}
    if a.seg: s0, s1 = int(a.seg[0] * a.fps), int(a.seg[1] * a.fps) - 1
    else:
        segs, _, _ = LS.presence(kp, ix, a.fps); s0, s1 = max(segs, key=lambda s: s[1] - s[0]); print("auto segment %.1f-%.1f s" % (s0 / a.fps, s1 / a.fps))
    base = []
    for tok in [t for t in a.baseline.split(",") if t]:
        pth, t0, t1 = tok.split(":"); k2, n2 = load(pth); base.append((k2, n2, int(float(t0) * a.fps), int(float(t1) * a.fps) - 1, a.fps))
    r = judge(kp, names, s0, s1, a.fps, a.weight, base or None); print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False, indent=1))
