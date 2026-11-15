from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def convert(h5, out):
    import pandas as pd
    d = pd.read_hdf(h5)
    names = list(dict.fromkeys([c[-2] for c in d.columns]))
    arr = np.stack([np.stack([d.xs((b, k), axis=1, level=(-2, -1)).values[:, 0] for k in ("x", "y", "likelihood")], axis=1) for b in names], axis=1)
    np.savez(out, kp=arr, names=np.array(names))
    print("saved", out, arr.shape)


def smooth_series(a, conf, thr=0.3, win=9):
    from scipy.signal import savgol_filter
    a = a.copy(); bad = conf < thr
    for dd in range(a.shape[1]):
        x = a[:, dd]; idx = np.arange(len(x))
        if (~bad).sum() > 10 and bad.any():
            x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        a[:, dd] = savgol_filter(x, win, 2) if len(x) > win else x
    return a


def presence(kp, ix, fps):
    body = ["back_base", "back_end", "tail_base", "neck_base"]
    conf = np.median(np.stack([kp[:, ix[b], 2] for b in body], 1), 1)
    L = np.linalg.norm(kp[:, ix["back_base"], :2] - kp[:, ix["back_end"], :2], axis=1)
    ok = (conf > 0.45) & (L > 80) & (L < 1500)
    gap = int(0.15 * fps); idx = np.nonzero(ok)[0]; runs = []
    if len(idx):
        s = p = idx[0]
        for i in idx[1:]:
            if i - p > gap: runs.append((s, p)); s = i
            p = i
        runs.append((s, p))
    return [(a, b) for a, b in runs if (b - a) / fps >= 0.6], conf, L


def analyse_segment(kp, ix, a, b, fps, back_len_m=0.45):
    from scipy.signal import find_peaks
    seg = kp[a:b + 1]; n = len(seg)
    P = lambda nm: smooth_series(seg[:, ix[nm], :2], seg[:, ix[nm], 2])
    nose, tail, bb, be = P("nose"), P("tail_base"), P("back_base"), P("back_end")
    f = (nose - tail).mean(0); f /= np.linalg.norm(f)
    ctr = 0.5 * (bb + be); Lpx = float(np.median(np.linalg.norm(bb - be, axis=1)))
    scale = back_len_m / Lpx
    t = np.arange(n) / fps
    vx = np.polyfit(t, ctr[:, 0], 1)[0] * scale
    speed = float(abs(np.polyfit(t, (ctr @ f), 1)[0]) * scale)
    legs = {"LF": "front_left", "RF": "front_right", "LH": "back_left", "RH": "back_right"}
    td, conf, u = {}, {}, {}
    for leg, pre in legs.items():
        paw = P(pre + "_paw"); u[leg] = ((paw - ctr) @ f) / Lpx
        pk, _ = find_peaks(u[leg], distance=int(0.14 * fps), prominence=0.12)
        td[leg] = pk; conf[leg] = float(np.mean(seg[:, ix[pre + "_paw"], 2]))
    per_list = np.concatenate([np.diff(v) for v in td.values() if len(v) > 2]) if any(len(v) > 2 for v in td.values()) else np.array([])
    per = float(np.median(per_list)) / fps if len(per_list) else float("nan")
    out = dict(start=a / fps, end=b / fps, dur=(b - a) / fps, facing="left" if f[0] < 0 else "right", speed_ms=speed, speed_bl=speed / back_len_m,
               stride_period=per, stride_hz=1.0 / per if per == per and per > 0 else float("nan"), n_strides=int(np.median([len(v) for v in td.values()])),
               paw_conf=float(np.mean(list(conf.values()))), body_px=Lpx)
    def lag(x, y):
        if len(td[x]) < 3 or len(td[y]) < 3 or per != per: return float("nan")
        d = []
        for tx in td[x]:
            j = int(np.argmin(np.abs(td[y] - tx))); d.append(((tx - td[y][j]) / fps / per + 0.5) % 1.0 - 0.5)
        d = np.array(d); return float(np.angle(np.mean(np.exp(2j * np.pi * d))) / (2 * np.pi))
    out.update(lag_LF_RH=lag("LF", "RH"), lag_RF_LH=lag("RF", "LH"), lag_LF_RF=lag("LF", "RF"), lag_LF_LH=lag("LF", "LH"), lag_RF_RH=lag("RF", "RH"))
    ab = lambda k: abs(out[k]) if out[k] == out[k] else 9.0
    dg1, dg2, lat1, lat2, fr = ab("lag_LF_RH"), ab("lag_RF_LH"), ab("lag_LF_LH"), ab("lag_RF_RH"), ab("lag_LF_RF")
    if per != per or out["n_strides"] < 2: g = "unclear"
    elif dg1 < 0.17 and dg2 < 0.17 and fr > 0.33: g = "trot"
    elif lat1 < 0.17 and lat2 < 0.17 and fr > 0.33: g = "pace"
    elif 0.17 <= dg1 <= 0.33 and 0.17 <= dg2 <= 0.33: g = "walk"
    elif fr < 0.2: g = "bound/gallop"
    elif (dg1 < 0.2 and dg2 < 0.3) or (dg2 < 0.2 and dg1 < 0.3): g = "trot-like (one diagonal loose)"
    else: g = "mixed"
    out["gait"] = g
    return out, td


def main():
    cmd = sys.argv[1]
    if cmd == "convert":
        convert(sys.argv[2], sys.argv[3]); return
    z = np.load(sys.argv[2]); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
    fps = float(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 120.0
    outp = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("results/leo_segments.csv")
    segs, conf, L = presence(kp, ix, fps)
    rows = []
    print(f"{len(segs)} passes with the dog visible")
    for a, b in segs:
        r, td = analyse_segment(kp, ix, a, b, fps); rows.append(r)
        print("  %6.1f-%6.1f s (%.1f s) facing %-5s speed %.2f m/s (%.1f BL/s) stride %.2f s n=%d paw conf %.2f -> %s  lags LF-RH %+.2f RF-LH %+.2f LF-RF %+.2f LF-LH %+.2f RF-RH %+.2f"
              % (r["start"], r["end"], r["dur"], r["facing"], r["speed_ms"], r["speed_bl"], r["stride_period"], r["n_strides"], r["paw_conf"], r["gait"],
                 r["lag_LF_RH"], r["lag_RF_LH"], r["lag_LF_RF"], r["lag_LF_LH"], r["lag_RF_RH"]))
    import csv
    outp.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(outp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("saved", outp)


if __name__ == "__main__":
    main()
