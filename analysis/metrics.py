from __future__ import annotations

import numpy as np

LEG_COL = {"LF": 0, "RF": 1, "LH": 2, "RH": 3}


def _ang(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    c = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def hind_angles(out: dict) -> dict[str, dict[str, np.ndarray]]:
    A = out["anchors"]
    trunk = out["torso"] - out["pelvis"]
    res = {}
    for li, leg in enumerate(("LH", "RH")):
        hip, st, hk, toe = (A[:, k, li] for k in range(4))
        res[leg] = dict(stifle=_ang(hip - st, hk - st), hock=_ang(st - hk, toe - hk), hip=_ang(trunk, st - hip))
    return res


def per_cycle_rom(sig: np.ndarray, t: np.ndarray, freq: float, t0: float) -> dict[str, float]:
    period = 1.0 / freq
    n_cyc = int((t[-1] - t0) // period)
    roms, mxs, mns = [], [], []
    for c in range(n_cyc):
        m = (t >= t0 + c * period) & (t < t0 + (c + 1) * period)
        if m.sum() < 5:
            continue
        roms.append(sig[m].max() - sig[m].min()); mxs.append(sig[m].max()); mns.append(sig[m].min())
    if not roms:
        return dict(rom=np.nan, rom_sd=np.nan, max=np.nan, min=np.nan, n_cycles=0)
    return dict(rom=float(np.mean(roms)), rom_sd=float(np.std(roms)), max=float(np.mean(mxs)),
                min=float(np.mean(mns)), n_cycles=len(roms))


def duty_factor(force: np.ndarray, thresh: float = 2.0) -> float:
    return float(np.mean(force > thresh))


def summarize(out: dict) -> dict:
    t, f0 = out["time"], out["freq"]
    t0 = out["warmup"]
    ang = hind_angles(out)
    s = dict(freq_hz=f0, fell=out["fell"], t_fall=out["t_fall"], speed=out["v_mean"], progress=out["progress"],
             stride_len=out["v_mean"] / f0)
    mask = t >= t0
    for leg in ("LH", "RH"):
        for j in ("stifle", "hock", "hip"):
            r = per_cycle_rom(ang[leg][j], t, f0, t0)
            for k, v in r.items():
                s[f"{leg}_{j}_{k}"] = v
        s[f"{leg}_duty"] = duty_factor(out["grf"][mask, LEG_COL[leg]])
    for leg in ("LF", "RF"):
        s[f"{leg}_duty"] = duty_factor(out["grf"][mask, LEG_COL[leg]])
    s["peak_grf_LH"] = float(out["grf"][mask, LEG_COL["LH"]].max()) if mask.any() else np.nan
    s["peak_grf_RH"] = float(out["grf"][mask, LEG_COL["RH"]].max()) if mask.any() else np.nan
    return s
