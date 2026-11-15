import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from leo_segments import presence, analyse_segment, smooth_series

z = np.load(sys.argv[1]); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
prefix = sys.argv[2]
fps = float(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 120.0
min_str = int(sys.argv[sys.argv.index("--min-strides") + 1]) if "--min-strides" in sys.argv else 3


def ang(a, b, c):
    v1, v2 = a - b, c - b
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)), -1, 1)))


def fold(sig, td, per, fps, n=60):
    cyc = [np.interp(np.linspace(td[i], td[i + 1], n), np.arange(len(sig)), sig) for i in range(len(td) - 1) if 0.6 * per < (td[i + 1] - td[i]) / fps < 1.5 * per]
    return np.array(cyc)


segs, _, _ = presence(kp, ix, fps)
rows, curves = [], {}
for a, b in segs:
    r, td = analyse_segment(kp, ix, a, b, fps)
    if r["n_strides"] < min_str or r["stride_period"] != r["stride_period"]: continue
    seg = kp[a:b + 1]; P = lambda nm: smooth_series(seg[:, ix[nm], :2], seg[:, ix[nm], 2])
    nose, tail, bb, be = P("nose"), P("tail_base"), P("back_base"), P("back_end"); f = (nose - tail).mean(0); f /= np.linalg.norm(f)
    L = float(np.median(np.linalg.norm(bb - be, axis=1))); down = np.array([-f[1], f[0]]); down = down if down[1] > 0 else -down
    per = r["stride_period"]; rec = dict(r)
    for leg, pre, key in (("LF", "front_left", "carpus"), ("RF", "front_right", "carpus"), ("LH", "back_left", "hock"), ("RH", "back_right", "hock")):
        th, kn, pw = P(pre + "_thai"), P(pre + "_knee"), P(pre + "_paw")
        c = fold(ang(th, kn, pw), td[leg], per, fps)
        meta = np.degrees(np.arctan2((kn - pw) @ (-down), np.abs((pw - kn) @ f)))
        cm = fold(meta, td[leg], per, fps)
        if len(c) >= 2:
            m = c.mean(0); curves[(a / fps, leg)] = m
            rec[f"{leg}_{key}_min"] = float(m.min()); rec[f"{leg}_{key}_max"] = float(m.max()); rec[f"{leg}_{key}_rom"] = float(np.ptp(m))
            rec[f"{leg}_{key}_phase_min"] = float(np.argmin(m) / 60); rec[f"{leg}_{key}_n"] = len(c)
            rec[f"{leg}_distal_incl_first_third"] = float(cm.mean(0)[:20].mean()) if len(cm) else float("nan")
    for fo, ho in (("LF", "RH"), ("RF", "LH")):
        d = []
        for tx in td[fo]:
            j = int(np.argmin(np.abs(td[ho] - tx))); v = (tx - td[ho][j]) / fps / per
            if abs(v) < 0.3: d.append(v)
        rec[f"lag_{fo}_{ho}_median"] = float(np.median(d)) if d else float("nan")
    rows.append(rec)
import csv
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("start", "end", "dur", "gait"), k))
with open(prefix + ".csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
np.savez(prefix + "_curves.npz", **{f"{k[0]:.1f}_{k[1]}": v for k, v in curves.items()})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4)); ph = np.linspace(0, 1, 60)
cm_ = plt.cm.viridis(np.linspace(0, 1, max(len(rows), 1)))
for (st, leg), m in curves.items():
    k = [i for i, r in enumerate(rows) if abs(r["start"] - st) < 0.05][0]
    axs[0 if leg in ("LF", "RF") else 1].plot(ph, m, color=cm_[k], lw=1, label=f"{rows[k]['start']:.0f}s {rows[k]['gait'][:5]}" if leg in ("LF", "LH") else None)
axs[0].set_title("Leo carpus angle (deg), stride from touchdown", fontsize=9); axs[1].set_title("Leo hock angle (deg)", fontsize=9)
axs[0].legend(fontsize=6, frameon=False); [a_.set_xlabel("stride phase") for a_ in axs]; fig.tight_layout(); fig.savefig(prefix + ".png", dpi=140)
print(f"{len(rows)} passes with >= {min_str} strides")
for r in rows:
    g = lambda k: r.get(k, float("nan"))
    print("  %5.1f s %-30s v %.2f m/s  fore-hind lag LF-RH %+.2f RF-LH %+.2f | carpus min/max/rom LF %.0f/%.0f/%.0f | hock LH %.0f/%.0f/%.0f"
          % (r["start"], r["gait"], r["speed_ms"], g("lag_LF_RH_median"), g("lag_RF_LH_median"), g("LF_carpus_min"), g("LF_carpus_max"), g("LF_carpus_rom"), g("LH_hock_min"), g("LH_hock_max"), g("LH_hock_rom")))
