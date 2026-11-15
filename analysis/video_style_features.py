from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter, find_peaks


def angle2(p, q, r):
    u, v = p - q, r - q
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def features(L: dict, fps: float) -> dict:
    t = np.arange(len(L["back_a"])) / fps
    ctr = 0.5 * (L["back_a"] + L["back_b"])
    speed = float(abs(np.polyfit(t, ctr[:, 0], 1)[0]))
    tib = float(np.median(np.linalg.norm(L["stifle"] - L["hock"], axis=1)))
    u = L["hpaw"][:, 0] - ctr[:, 0]
    w = max(5, int(0.1 * fps) | 1)
    uf = savgol_filter(u, w, 2) if len(u) > w else u
    pk, _ = find_peaks(uf, distance=int(0.14 * fps), prominence=0.12 * np.ptp(uf))
    stride = (len(pk) - 1) / ((pk[-1] - pk[0]) / fps) if len(pk) > 2 else float("nan")
    hock = angle2(L["stifle"], L["hock"], L["hpaw"]); carp = angle2(L["elbow"], L["carpus"], L["fpaw"])
    win = max(5, int(0.05 * fps) | 1)
    hock_s = savgol_filter(hock, win, 2); carp_s = savgol_filter(carp, win, 2)
    bounce = float(np.std(savgol_filter(ctr[:, 1], max(5, int(0.1 * fps) | 1), 2)) / tib)
    return dict(speed=speed, stride_hz=float(stride), hock_min=float(np.percentile(hock_s, 3)), hock_max=float(np.percentile(hock_s, 97)),
                carpus_min=float(np.percentile(carp_s, 3)), carpus_max=float(np.percentile(carp_s, 97)), bounce=bounce,
                step=speed / stride / tib if stride == stride and stride > 0 else float("nan"), tibia=tib)


def sim_landmarks(path: str):
    z = np.load(path, allow_pickle=True); names = list(z["names"][: len(z["names"]) - 4]); P = z["pos"]; fps = float(z["fps"])
    ix = {n: i for i, n in enumerate(names)}; ntot = P.shape[1]
    xz = lambda n: P[:, ix[n]][:, [0, 2]]
    tip = lambda k: P[:, ntot - 4 + k][:, [0, 2]]
    L = dict(back_a=xz("C_7"), back_b=xz("pelvis"), stifle=xz("lower_leg_L"), hock=xz("foot_L"), hpaw=tip(2), elbow=xz("lower_arm_L"), carpus=xz("hand_anchor_L"), fpaw=tip(0))
    return L, fps


def from_sim_npz(path: str, t_skip: float = 0.0) -> dict:
    L, fps = sim_landmarks(path); s = int(t_skip * fps); return features({k: v[s:] for k, v in L.items()}, fps)


def windows(L: dict, fps: float, win_s: float = 4.0, hop_s: float = 2.0):
    n = len(L["back_a"]); w = int(win_s * fps); h = int(hop_s * fps); out = []
    for a in range(0, n - w + 1, h): out.append(features({k: v[a:a + w] for k, v in L.items()}, fps))
    return out


def video_landmarks(kp, names, a: int, b: int, tibia_m: float = 0.127):
    import sys; sys.path.insert(0, "analysis")
    from analysis.leo_segments import smooth_series
    ix = {n: i for i, n in enumerate(names)}; S = kp[a:b + 1]
    P = lambda nm: smooth_series(S[:, ix[nm], :2], S[:, ix[nm], 2], thr=0.3, win=9); C = lambda nm: S[:, ix[nm], 2]
    face = np.sign(np.median(P("nose")[:, 0] - P("tail_base")[:, 0])) or 1.0
    best = lambda legs: max(legs, key=lambda s: sum(np.median(C(f"{s}_{k}")) for k in ("thai", "knee", "paw")))
    hs = best(["back_left", "back_right"]); fs = best(["front_left", "front_right"])
    tibs = []
    for s in ("left", "right"):
        m = (C(f"back_{s}_thai") > 0.5) & (C(f"back_{s}_knee") > 0.5)
        if m.sum() > 10: tibs.append(np.median(np.linalg.norm(P(f"back_{s}_thai")[m] - P(f"back_{s}_knee")[m], axis=1)))
    mpp = tibia_m / float(np.mean(tibs))
    conv = lambda arr: np.stack([arr[:, 0] * face, -arr[:, 1]], 1) * mpp
    L = dict(back_a=conv(P("back_base")), back_b=conv(P("back_end")), stifle=conv(P(f"{hs}_thai")), hock=conv(P(f"{hs}_knee")), hpaw=conv(P(f"{hs}_paw")),
             elbow=conv(P(f"{fs}_thai")), carpus=conv(P(f"{fs}_knee")), fpaw=conv(P(f"{fs}_paw")))
    return L


def from_video_kp(kp, names, a: int, b: int, fps: float, tibia_m: float = 0.127) -> dict:
    return features(video_landmarks(kp, names, a, b, tibia_m), fps)
