import sys
import numpy as np
from scipy.signal import savgol_filter, find_peaks
FPS = 59.94


def load(p):
    z = np.load(p); k = z["kp"]; nm = list(z["names"]); return k, {n: i for i, n in enumerate(nm)}


def smooth(a, conf, thr=0.3):
    a = a.copy(); bad = conf < thr
    for d in range(a.shape[1]):
        x = a[:, d]; idx = np.arange(len(x)); x[bad] = np.interp(idx[bad], idx[~bad], x[~bad]) if (~bad).sum() > 10 else x[bad]
        a[:, d] = savgol_filter(x, 9, 2)
    return a


def analyse(path):
    k, ix = load(path); n = len(k)
    P = lambda nm: smooth(k[:, ix[nm], :2], k[:, ix[nm], 2])
    nose, tail = P("nose"), P("tail_base"); bb, be = P("back_base"), P("back_end")
    f = (nose - tail).mean(0); f /= np.linalg.norm(f)
    ctr = 0.5 * (bb + be); L = np.median(np.linalg.norm(bb - be, axis=1))
    down = np.array([-f[1], f[0]]); down = down if down[1] > 0 else -down
    U = lambda p: ((p - ctr) @ f) / L
    V = lambda p: ((p - ctr) @ down) / L
    legs = {"LF": "front_left", "RF": "front_right", "LH": "back_left", "RH": "back_right"}
    out = {}
    t = np.arange(n) / FPS
    for leg, pre in legs.items():
        paw, kn, th = P(pre + "_paw"), P(pre + "_knee"), P(pre + "_thai")
        u = U(paw)
        pk, _ = find_peaks(u, distance=int(0.22 * FPS), prominence=0.08)
        def ang(a, b, c):
            v1, v2 = a - b, c - b
            return np.degrees(np.arccos(np.clip((v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)), -1, 1)))
        out[leg] = dict(t=t, u=u, td=pk, distal=ang(th, kn, paw), meta_incl=np.degrees(np.arctan2(-(paw - kn) @ down, np.abs((paw - kn) @ f))), paw_y=paw[:, 1], knee=kn, thai=th, paw=paw)
    return out, f, L


if __name__ == "__main__":
    for p in sys.argv[1:]:
        out, f, L = analyse(p)
        per = np.median(np.concatenate([np.diff(o["td"]) for o in out.values() if len(o["td"]) > 2])) / FPS
        print(f"== {Path(p).name if False else p.split('/')[-1]}  body length {L:.0f}px, stride period ~{per:.2f}s, forward dir {np.round(f, 2)}")
        for leg, o in out.items(): print("  ", leg, "touchdowns", len(o["td"]))
        lags = []
        for fo, ho in (("LF", "RH"), ("RF", "LH"), ("LF", "LH"), ("RF", "RH")):
            d = []
            for a in out[fo]["td"]:
                if len(out[ho]["td"]):
                    j = np.argmin(np.abs(out[ho]["td"] - a)); dd = (a - out[ho]["td"][j]) / FPS / per
                    if abs(dd) < 0.25: d.append(dd)
            print(f"   fore {fo} - hind {ho}: median lag {np.median(d) if d else float('nan'):+.3f} cycles (n={len(d)}; negative = fore first)")
