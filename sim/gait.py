from __future__ import annotations

import numpy as np

FORE_JOINTS = ("shoulder_{s}_extend", "elbow_{s}", "wrist_{s}")
HIND_JOINTS = ("hip_{s}_extend", "knee_{s}", "ankle_{s}")
ABD_JOINT = {"fore": "scapula_{s}_abduct", "hind": "hip_{s}_abduct"}
LEGS = {"LF": ("fore", "L", 0.0), "RH": ("hind", "R", 0.0), "RF": ("fore", "R", 0.5), "LH": ("hind", "L", 0.5)}
JOINTS_OF = {"fore": FORE_JOINTS, "hind": HIND_JOINTS}
N_TRUNK = 8
PER_GROUP = 10


class GaitSpec:
    GUESS = {
        "fore": ([0.20, 0.30, 0.25], [0.0, 1.6, 1.8], [0.0, -0.15, -0.10], [0.10]),
        "hind": ([0.20, 0.30, 0.30], [0.0, 1.6, 1.8], [0.0, -0.10, -0.05], [0.10]),
    }

    def __init__(self, gcfg: dict, free_legs=()):
        self.g = gcfg
        self.free_legs = tuple(free_legs)
        self.groups = ["fore", "hind"] + list(self.free_legs)
        self.harm = int(gcfg.get("harmonics", 1))
        self.pg = PER_GROUP + (6 if self.harm >= 2 else 0)
        self.spine = bool(gcfg.get("spine", False))

    def group_of(self, leg: str) -> str:
        return leg if leg in self.free_legs else LEGS[leg][0]

    @property
    def n_params(self) -> int:
        return 1 + self.pg * len(self.groups) + (N_TRUNK if self.spine else 0)

    def bounds(self):
        g = self.g
        lo, hi = [g.get("freq_lo", g["freq_bounds"][0])], [g["freq_bounds"][1]]
        for _ in self.groups:
            lo += [g["amp_bounds"][0]] * 3 + [-np.pi] * 3 + [g["bias_bounds"][0]] * 3 + [g["abd_bounds"][0]]
            hi += [g["amp_bounds"][1]] * 3 + [np.pi] * 3 + [g["bias_bounds"][1]] * 3 + [g["abd_bounds"][1]]
            if self.harm >= 2:
                lo += [0.0] * 3 + [-np.pi] * 3
                hi += [g.get("amp2_max", 0.5)] * 3 + [np.pi] * 3
        if self.spine:
            lo += [0.0, -np.pi, -0.3, 0.0, -np.pi, 0.0, -np.pi, -0.2]
            hi += [g.get("lumbar_amp_max", 0.8), np.pi, 0.3, g.get("lumbar_bend_max", 0.5), np.pi, g.get("scapula_amp_max", 0.35), np.pi, 0.2]
        return np.array(lo), np.array(hi)

    def initial_guess(self) -> np.ndarray:
        x = [2.2]
        for grp in self.groups:
            kind = grp if grp in self.GUESS else LEGS[grp][0]
            amp, ph, bi, ab = self.GUESS[kind]
            x += amp + ph + bi + ab
            if self.harm >= 2:
                x += [0.0] * 3 + [0.0] * 3
        if self.spine:
            x += [0.0] * N_TRUNK
        return np.array(x)

    def unpack(self, x):
        f = float(x[0])
        out = {}
        for k, grp in enumerate(self.groups):
            s = 1 + self.pg * k
            h2 = (np.array(x[s + 10 : s + 13]), np.array(x[s + 13 : s + 16])) if self.harm >= 2 else (np.zeros(3), np.zeros(3))
            out[grp] = (np.array(x[s : s + 3]), np.array(x[s + 3 : s + 6]), np.array(x[s + 6 : s + 9]), float(x[s + 9]), h2)
        st = 1 + self.pg * len(self.groups)
        out["trunk"] = np.array(x[st : st + N_TRUNK]) if self.spine and len(x) >= st + N_TRUNK else np.zeros(N_TRUNK)
        return f, out

    def warm_start(self, src: "GaitSpec", x_src: np.ndarray) -> np.ndarray:
        f, P = src.unpack(x_src)
        x = [f]
        for grp in self.groups:
            kind = grp if grp in ("fore", "hind") else LEGS[grp][0]
            leg_src = grp if grp in src.free_legs else kind
            amp, ph, bi, ab, h2 = P[leg_src]
            x += list(amp) + list(ph) + list(bi) + [ab]
            if self.harm >= 2:
                x += list(h2[0]) + list(h2[1])
        if self.spine:
            x += list(P["trunk"])
        return np.array(x)


class TrotController:

    def __init__(self, spec: GaitSpec, name_to_k: dict[str, int], q_hold: np.ndarray):
        self.spec = spec
        self.q_hold = q_hold
        idx, legs, off = [], [], []
        abd_idx, abd_leg = [], []
        for leg, (kind, side, o) in LEGS.items():
            for jn in JOINTS_OF[kind]:
                idx.append(name_to_k[jn.format(s=side)]); legs.append(leg); off.append(o)
            abd_idx.append(name_to_k[ABD_JOINT[kind].format(s=side)]); abd_leg.append(leg)
        self.idx, self.legs, self.offset = np.array(idx), legs, np.array(off)
        self.abd_idx, self.abd_leg = np.array(abd_idx), abd_leg
        self.sc_idx = np.array([name_to_k["scapula_L_extend"], name_to_k["scapula_R_extend"]]) if "scapula_L_extend" in name_to_k else None
        self.sc_off = np.array([0.0, 0.5])
        self.trunk = np.zeros(N_TRUNK)
        name_to_k["_ctl"] = self
        self.tmpl = None
        if spec.g.get("template"):
            from pathlib import Path
            rc = np.load(Path(__file__).resolve().parents[1] / "data/ref/youtube/work/ref_curves.npz")
            def prep(c, K=6, n=720):
                F = np.fft.rfft(np.asarray(c, float)); F[K + 1:] = 0
                cc = np.fft.irfft(F, n=n); cc = cc - cc.mean(); return cc / (np.ptp(cc) / 2)
            self.tmpl = {"fore": prep(rc["carpus"]), "hind": prep(rc["hock"])}
            self.tmpl_sel = np.array([n % 3 == 2 for n in range(len(self.legs))])
            self.tmpl_hind = np.array([LEGS[l][0] == "hind" for l in self.legs])
            self.tmpl_mix = float(spec.g.get("template_mix", 1.0))
            sh = spec.g.get("template_shift", {"fore": 0.0, "hind": 0.0})
            self.tmpl_shift = np.where(self.tmpl_hind, float(sh["hind"]), float(sh["fore"]))

    def _template(self, phi):
        out = np.zeros_like(phi)
        for key, mask in (("fore", ~self.tmpl_hind), ("hind", self.tmpl_hind)):
            tbl = self.tmpl[key]; u = (phi[mask] % 1.0) * len(tbl)
            i0 = np.floor(u).astype(int) % len(tbl); fr = u - np.floor(u)
            out[mask] = tbl[i0] * (1 - fr) + tbl[(i0 + 1) % len(tbl)] * fr
        return out

    def set_params(self, x: np.ndarray):
        f, P = self.spec.unpack(x)
        self.f = f
        amp, ph, bi, a2, p2 = [], [], [], [], []
        for n, leg in enumerate(self.legs):
            a, p, b, _, h2 = P[self.spec.group_of(leg)]
            j = n % 3
            amp.append(a[j]); ph.append(p[j]); bi.append(b[j]); a2.append(h2[0][j]); p2.append(h2[1][j])
        self.amp, self.ph, self.bi = np.array(amp), np.array(ph), np.array(bi)
        self.amp2, self.ph2 = np.array(a2), np.array(p2)
        self.abd = np.array([P[self.spec.group_of(leg)][3] for leg in self.abd_leg])
        self.trunk = np.asarray(P["trunk"], float)
        self.spine_on = bool(self.spec.spine and np.any(self.trunk[[0, 2, 3, 5, 7]] != 0.0))

    def targets(self, t: float, ramp: float) -> np.ndarray:
        q = self.q_hold.copy()
        th = 2 * np.pi * (self.f * t + self.offset)
        wave = self.amp * np.sin(th + self.ph) + self.amp2 * np.sin(2 * th + self.ph2)
        if self.tmpl is not None:
            phi = self.f * t + self.offset + self.ph / (2 * np.pi) + self.tmpl_shift
            wave = np.where(self.tmpl_sel, (1 - self.tmpl_mix) * wave + self.tmpl_mix * self.amp * self._template(phi), wave)
        q[self.idx] += ramp * (self.bi + wave)
        q[self.abd_idx] += ramp * self.abd
        if self.spec.spine and self.sc_idx is not None:
            sa, sp, sb = self.trunk[5], self.trunk[6], self.trunk[7]
            q[self.sc_idx] += ramp * (sb + sa * np.sin(2 * np.pi * (self.f * t + self.sc_off) + sp))
        return q

    def lumbar_ctrl(self, t: float, ramp: float):
        la, lp, lb, ba, bp = self.trunk[0], self.trunk[1], self.trunk[2], self.trunk[3], self.trunk[4]
        ext = ramp * (lb + la * np.sin(2 * np.pi * 2 * self.f * t + lp))
        bend = ramp * ba * np.sin(2 * np.pi * self.f * t + bp)
        return float(np.clip(ext, -1.0, 1.0)), float(np.clip(bend, -1.0, 1.0))
