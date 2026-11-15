from __future__ import annotations

import mujoco
import numpy as np

from analysis.metrics import hind_angles
from sim.config import ROOT
from sim.dog_env import body_id, joint_id, load_dog, make_position_servos
from sim.gait import GaitSpec, TrotController
from sim.impairment import KneeLimiter

LEG_ROOTS = {"upper_leg_L": "LH", "upper_leg_R": "RH", "scapula_L": "LF", "scapula_R": "RF"}
LEG_ORDER = ["LF", "RF", "LH", "RH"]


class Sim:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.m, self.d = load_dog(timestep=cfg["sim"]["timestep"], integrator=cfg["sim"]["integrator"], leo=cfg.get("model"))
        self.sv = make_position_servos(self.m, kp_scale=cfg["servo"]["kp_scale"], kd_s=cfg["servo"]["kd_s"],
                                       digit_kp=cfg["servo"].get("digit_kp"), digit_kd=cfg["servo"].get("digit_kd", 0.3))
        if cfg.get("body"):
            from sim.dog_scale import apply_composition
            self.composition = apply_composition(self.m, self.d, self.sv, cfg["body"])
        hc = cfg.get("head")
        self.head = hc
        if hc:
            names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(self.m.nu)]
            self.head_act = names.index("cervical_extend")
            self.head_c1, self.head_c7 = body_id(self.m, "C_1"), body_id(self.m, "C_7")
        names_ = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(self.m.nu)]
        self.lumb_ext_act, self.lumb_bend_act = names_.index("lumbar_extend"), names_.index("lumbar_bend")
        tc = cfg.get("tail")
        self.tail = tc
        if tc:
            names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(self.m.nu)]
            self.tail_act = names.index("caudal_extend")
            self.tail_b1, self.tail_b2 = body_id(self.m, "Ca_1"), body_id(self.m, "Ca_21")
        mujoco.mj_resetData(self.m, self.d)
        self.limiter = KneeLimiter(self.m, self.sv)
        self.name_to_k = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j): k for k, j in enumerate(self.sv["joint_ids"])}
        self.torso = body_id(self.m, "torso")
        self.pelvis = body_id(self.m, "pelvis")
        self.geom_leg = self._geom_leg()
        self.anchor_ids = {
            (leg, p): joint_id(self.m, name.format(s=side))
            for leg, side in (("LH", "L"), ("RH", "R"))
            for p, name in (("hip", "hip_{s}_extend"), ("stifle", "knee_{s}"), ("hock", "ankle_{s}"), ("toe", "toe_{s}"))
        }
        self.anchor_idx = [[self.anchor_ids[(leg, p)] for leg in ("LH", "RH")] for p in ("hip", "stifle", "hock", "toe")]
        self.dig = []
        for nm in ("finger_L", "finger_R", "toe_L", "toe_R"):
            b = body_id(self.m, nm)
            gs = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b]
            lv = max((self.m.geom_pos[g] for g in gs), key=lambda p: np.linalg.norm(p))
            self.dig.append((b, np.asarray(lv, float)))
        self.dig_geom = []
        for b, lv in self.dig:
            gs = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b]
            self.dig_geom.append(max(gs, key=lambda g: np.linalg.norm(self.m.geom_pos[g])))
        self.ang_j = [(joint_id(self.m, a_), joint_id(self.m, b_)) for a_, b_ in
                      (("elbow_L", "wrist_L"), ("elbow_R", "wrist_R"), ("knee_L", "ankle_L"), ("knee_R", "ankle_R"))]
        jn_ = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.foot_geoms = []
        for b1, b2 in (("hand_L", "finger_L"), ("hand_R", "finger_R"), ("foot_L", "toe_L"), ("foot_R", "toe_R")):
            ids = {body_id(self.m, b1), body_id(self.m, b2)}
            self.foot_geoms.append([g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) in ids and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])])
        self.lat_j = [joint_id(self.m, n_) for n_ in ("shoulder_L_extend", "shoulder_R_extend", "hip_L_extend", "hip_R_extend")]
        self.lumb_adr = [int(self.m.jnt_qposadr[jn_(f"L_{i}_extend")]) for i in (1, 3, 5, 7)]
        self.ref_curves = None
        if self.cfg["cost"].get("w_ref"):
            rc = np.load(ROOT / self.cfg["cost"].get("ref_file", "data/ref/youtube/work/ref_curves.npz"))
            zs = lambda v: (v - v.mean()) / v.std()
            self.ref_curves = {"fore": zs(rc["carpus"]), "hind": zs(rc["hock"]), "carpus_rom": float(np.ptp(rc["carpus"]))}

    def head_ctrl(self, d) -> float:
        ctl = self.name_to_k.get("_ctl") if hasattr(self, "name_to_k") else None
        if ctl is not None and getattr(ctl, "spine_on", False):
            tt = float(d.time); ramp = min(1.0, tt / self.cfg["gait"]["ramp_in"]) if self.cfg["gait"]["ramp_in"] > 0 else 1.0
            d.ctrl[self.lumb_ext_act], d.ctrl[self.lumb_bend_act] = ctl.lumbar_ctrl(tt, ramp)
        if self.tail:
            t = self.tail
            v = d.xpos[self.tail_b2] - d.xpos[self.tail_b1]; f = d.xmat[self.torso].reshape(3, 3)[:, 0].copy(); f[2] = 0.0; f /= (np.linalg.norm(f) + 1e-9)
            rho = np.degrees(np.arctan2(v[2], -(v @ f)))
            d.ctrl[self.tail_act] = float(np.clip(t["ctrl0"] + t["gain"] * (t["target_rho_deg"] - rho), t["min"], t["max"]))
        h = self.head
        v = d.xpos[self.head_c1] - d.xpos[self.head_c7]
        el = np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
        return float(np.clip(h["ctrl0"] - h["gain"] * (h["target_elev_deg"] - el), h["min"], h["max"]))

    def _geom_leg(self) -> np.ndarray:
        m = self.m
        body_leg = {}
        for b in range(m.nbody):
            cur = b
            while cur > 0:
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, cur)
                if n in LEG_ROOTS:
                    body_leg[b] = LEG_ORDER.index(LEG_ROOTS[n])
                    break
                cur = m.body_parentid[cur]
        return np.array([body_leg.get(int(m.geom_bodyid[g]), -1) for g in range(m.ngeom)])

    def run(self, spec: GaitSpec, x: np.ndarray, record: bool = False, duration: float | None = None) -> dict:
        m, d, sv, cfg = self.m, self.d, self.sv, self.cfg
        gc = cfg["gait"]
        T = duration or gc["eval_duration"]
        dt = m.opt.timestep
        mujoco.mj_resetData(m, d)
        self.limiter.clip_qpos(d)
        q_hold = d.qpos[sv["qadr"]].copy()
        ctl = TrotController(spec, self.name_to_k, q_hold)
        ctl.set_params(x)
        n = int(T / dt)
        wu = int(gc["warmup"] / dt)
        pos = np.zeros((n, 3)); yaw = np.zeros(n); up = np.zeros(n); tau2 = np.zeros(n)
        grf = np.zeros((n, 4)); anch = np.zeros((n, 4, 2, 3)); pelv = np.zeros((n, 3))
        fq = np.zeros((n, len(ctl.idx)))
        wv = np.zeros((n, 2))
        tiplat = np.zeros((n, 4)); jlat = np.zeros((n, 4))
        tipz = np.zeros((n, 4)); lumb = np.zeros(n); pdiff = np.zeros(n)
        dang = np.zeros((n, 4))
        dpitch = np.zeros((n, 4))
        if record:
            qlog = np.zeros((n, len(sv["act_ids"]))); tau_log = np.zeros_like(qlog)
        fell_at, n_used = None, n
        f6 = np.zeros(6)
        for i in range(n):
            t = i * dt
            ramp = min(1.0, t / gc["ramp_in"]) if gc["ramp_in"] > 0 else 1.0
            d.ctrl[sv["act_ids"]] = ctl.targets(t, ramp)
            if self.head:
                d.ctrl[self.head_act] = self.head_ctrl(d)
            mujoco.mj_step(m, d)
            R = d.xmat[self.torso].reshape(3, 3)
            pos[i] = d.xpos[self.torso]; pelv[i] = d.xpos[self.pelvis]
            yaw[i] = np.arctan2(R[1, 0], R[0, 0]); up[i] = R[2, 2]
            fq[i] = d.qpos[sv["qadr"][ctl.idx]]
            wv[i] = d.qvel[3:5]
            tau = d.actuator_force[sv["act_ids"]]
            tau2[i] = np.sum(tau[ctl.idx] ** 2)
            for pi in range(4):
                anch[i, pi, 0] = d.xanchor[self.anchor_idx[pi][0]]; anch[i, pi, 1] = d.xanchor[self.anchor_idx[pi][1]]
            for kk, (bb_, lv_) in enumerate(self.dig):
                vv = d.xmat[bb_].reshape(3, 3) @ lv_
                dpitch[i, kk] = np.degrees(np.arctan2(vv[2], np.hypot(vv[0], vv[1])))
            for kk in range(4):
                lo_ = 1e9
                for g_ in self.foot_geoms[kk]:
                    Rg = d.geom_xmat[g_]; sz = m.geom_size[g_]; ty = m.geom_type[g_]; zc = d.geom_xpos[g_][2]
                    if ty == mujoco.mjtGeom.mjGEOM_SPHERE: e_ = sz[0]
                    elif ty == mujoco.mjtGeom.mjGEOM_CAPSULE: e_ = abs(Rg[8]) * sz[1] + sz[0]
                    elif ty == mujoco.mjtGeom.mjGEOM_ELLIPSOID: e_ = float(np.sqrt((Rg[6] * sz[0]) ** 2 + (Rg[7] * sz[1]) ** 2 + (Rg[8] * sz[2]) ** 2))
                    elif ty == mujoco.mjtGeom.mjGEOM_BOX: e_ = abs(Rg[6]) * sz[0] + abs(Rg[7]) * sz[1] + abs(Rg[8]) * sz[2]
                    else: e_ = m.geom_rbound[g_]
                    if zc - e_ < lo_: lo_ = zc - e_
                tipz[i, kk] = lo_
            fw_ = R[:, 0].copy(); fw_[2] = 0.0; fw_ /= (np.linalg.norm(fw_) + 1e-9); lat_ = np.array([-fw_[1], fw_[0], 0.0])
            for kk in range(4):
                tiplat[i, kk] = (d.xpos[self.dig[kk][0]] - d.xpos[self.torso]) @ lat_
                jlat[i, kk] = (d.xanchor[self.lat_j[kk]] - d.xpos[self.torso]) @ lat_
            lumb[i] = np.degrees(d.qpos[self.lumb_adr]).sum()
            Rp = d.xmat[self.pelvis].reshape(3, 3); pdiff[i] = np.degrees(np.arcsin(-R[2, 0])) - np.degrees(np.arcsin(-Rp[2, 0]))
            if self.ref_curves is not None:
                fwd = R[:, 0].copy(); fwd[2] = 0.0; fwd /= (np.linalg.norm(fwd) + 1e-9)
                for kk in range(4):
                    ja, jb = self.ang_j[kk]
                    pa, pb, pt = d.xanchor[ja], d.xanchor[jb], d.geom_xpos[self.dig_geom[kk]]
                    v1 = np.array([(pa - pb) @ fwd, pa[2] - pb[2]]); v2 = np.array([(pt - pb) @ fwd, pt[2] - pb[2]])
                    dang[i, kk] = np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1.0, 1.0)))
            nc = d.ncon
            if nc:
                g1 = d.contact.geom1[:nc]; g2 = d.contact.geom2[:nc]
                other = np.where(g1 == 0, g2, g1)
                sel = np.nonzero(((g1 == 0) | (g2 == 0)) & (self.geom_leg[other] >= 0))[0]
                for c in sel:
                    mujoco.mj_contactForce(m, d, int(c), f6)
                    grf[i, self.geom_leg[other[c]]] += f6[0]
            if record:
                qlog[i] = d.qpos[sv["qadr"]]; tau_log[i] = tau
            if (d.xpos[self.torso][2] < gc["fall"]["min_torso_z"] or up[i] < gc["fall"]["min_up"]
                    or not np.isfinite(d.qpos).all()):
                fell_at, n_used = t, i + 1
                break
        out = dict(fell=fell_at is not None, t_fall=fell_at if fell_at is not None else T, T=T)
        w0 = min(wu, n_used - 1)
        w = slice(w0, n_used)
        span = max((n_used - 1 - w0) * dt, dt)
        out["v_mean"] = float((pos[n_used - 1, 0] - pos[w0, 0]) / span)
        r0 = min(int(gc["ramp_in"] / dt), n_used - 1)
        out["progress"] = float((pos[n_used - 1, 0] - pos[r0, 0]) / (gc["target_speed"] * (T - gc["ramp_in"])))
        out["lat_rate"] = float(abs(pos[n_used - 1, 1] - pos[w0, 1]) / span)
        out["yaw_abs"] = float(np.max(np.abs(yaw[w] - yaw[0])))
        tw = np.arange(n_used)[w] * dt
        out["yaw_rate"] = float(abs(np.polyfit(tw, np.unwrap(yaw[w]), 1)[0]) * 180.0 / np.pi) if len(tw) > 20 else 0.0
        out["tilt"] = float(np.std(1.0 - up[w])) if n_used > wu + 2 else 1.0
        out["energy"] = float(np.mean(tau2[w]))
        out["z_mean"] = float(np.mean(pos[w, 2]))
        out["freq"] = float(ctl.f)
        pen_meta, n_st = 0.0, 0
        for li in (0, 1):
            v = anch[w, 3, li] - anch[w, 2, li]
            incl = np.degrees(np.arctan2(-v[:, 2], np.hypot(v[:, 0], v[:, 1])))
            st = grf[w, 2 + li] > 2.0
            if st.any():
                pen_meta += float(np.sum(np.maximum(0.0, self.cfg["cost"].get("meta_min", 75.0) - incl[st]) ** 2)); n_st += int(st.sum())
        out["meta_pen"] = pen_meta / max(n_st, 1)
        out["meta_mean"] = float(np.mean([np.degrees(np.arctan2(-(anch[w, 3, li] - anch[w, 2, li])[:, 2], np.hypot(*(anch[w, 3, li] - anch[w, 2, li])[:, :2].T)))[grf[w, 2 + li] > 2.0].mean() if (grf[w, 2 + li] > 2.0).any() else 0.0 for li in (0, 1)]))
        out["wobble"] = float(np.sqrt(np.mean(wv[w] ** 2))) if n_used > w0 + 2 else 9.9
        out["shape_err"] = None
        steady = n_used - w0 > int(1.0 / dt)
        F = grf[w]
        out["duty"] = (F > 2.0).mean(axis=0).tolist() if steady else [0.0] * 4
        tot = F.sum()
        out["hind_load"] = float(F[:, 2:].sum() / tot) if steady and tot > 1e-6 else 0.0
        if steady:
            A = dict(anchors=anch[w], torso=pos[w], pelvis=pelv[w])
            ang = hind_angles(A)
            win = {}
            for j in ("stifle", "hock", "hip"):
                lo = np.mean([np.percentile(ang[leg][j], 2) for leg in ang])
                hi = np.mean([np.percentile(ang[leg][j], 98) for leg in ang])
                win[j] = (float(lo), float(hi))
            out["win"] = win
            pen_to, n_to = 0.0, 0
            tol = self.cfg["cost"].get("toeoff_tol", 12.0)
            for li, leg in enumerate(("LH", "RH")):
                st = grf[w, 2 + li] > 2.0
                ev = np.nonzero(st[:-1] & ~st[1:])[0]
                hi_leg = np.percentile(ang[leg]["stifle"], 98)
                for k in ev:
                    pen_to += max(0.0, (hi_leg - tol) - ang[leg]["stifle"][k]) ** 2; n_to += 1
            out["toeoff_pen"] = pen_to / n_to if n_to else 400.0
            fa = np.degrees(fq[w])
            dtw = dt
            Fw = grf[w]
            td = {}
            for k, leg in enumerate(("LF", "RF", "LH", "RH")):
                stk = Fw[:, k] > 2.0
                ev = np.nonzero(~stk[:-1] & stk[1:])[0]
                td[leg] = (ev * dtw * ctl.f) % 1.0
            def _lag(x, y):
                dd = [((px - py + 0.5) % 1.0) - 0.5 for px in td[x] for py in td[y]]
                dd = [v for v in dd if abs(v) < 0.25]
                return float(np.median(dd)) if dd else 0.0
            out["td_lag"] = 0.5 * (_lag("LF", "RH") + _lag("RF", "LH"))
            pen_r, n_r = 0.0, 0
            for col, k in ((0, 0), (6, 1), (9, 2), (3, 3)):
                stk = Fw[:, k] > 2.0
                if stk.sum() > 5:
                    vel = np.gradient(fa[:, col], dtw)
                    pen_r += float(np.sum(np.maximum(0.0, -vel[stk]) ** 2)); n_r += int(stk.sum())
            out["retract_pen"] = pen_r / max(n_r, 1)
            pen_p, n_p, pm = 0.0, 0, []
            pmin = self.cfg["cost"].get("paw_min", -20.0)
            for k in range(4):
                stk = Fw[:, k] > 2.0
                if stk.sum() > 5:
                    pk_ = dpitch[w][stk, k]; pen_p += float(np.sum(np.maximum(0.0, pmin - pk_) ** 2)); n_p += int(stk.sum()); pm.append(float(np.median(pk_)))
            if self.ref_curves is not None:
                from scipy.signal import medfilt
                period = 1.0 / ctl.f; errs, roms = [], []
                for k in range(4):
                    st_ = medfilt((Fw[:, k] > 2.0).astype(float), 21) > 0.5
                    tdi = np.nonzero(~st_[:-1] & st_[1:])[0] + 1
                    sig = dang[w][:, k]; cyc = []
                    for a_, b_ in zip(tdi[:-1], tdi[1:]):
                        if 0.6 * period < (b_ - a_) * dt < 1.4 * period:
                            cyc.append(np.interp(np.linspace(a_, b_, 60), np.arange(len(sig)), sig))
                    if len(cyc) >= 2:
                        mc = np.mean(cyc, axis=0); zc = (mc - mc.mean()) / (mc.std() + 1e-3)
                        errs.append(float(np.sqrt(np.mean((zc - self.ref_curves["fore" if k < 2 else "hind"]) ** 2))))
                        if k < 2: roms.append(float(np.ptp(mc)))
                    else:
                        errs.append(2.0)
                out["ref_err"] = float(np.mean(errs)); out["carpus_rom"] = float(np.mean(roms)) if roms else 0.0
            pen_c, cl = 0.0, []
            cmin = self.cfg["cost"].get("clear_min", 0.03)
            for k in range(4):
                stk = Fw[:, k] > 2.0
                if stk.sum() > 5 and (~stk).sum() > 5:
                    base_ = np.median(tipz[w][stk, k]); pk95 = np.percentile(tipz[w][~stk, k], 95) - base_
                    cl.append(float(pk95)); pen_c += max(0.0, cmin - pk95) ** 2
            out["clear_pen"] = pen_c / 4.0; out["clearance"] = cl
            pen_s, sp = 0.0, []
            smax = self.cfg["cost"].get("sole_pitch", -45.0)
            for k in (2, 3):
                stk = Fw[:, k] > 2.0
                if stk.sum() > 5 and (~stk).sum() > 5:
                    p5 = float(np.percentile(dpitch[w][~stk, k], 5)); sp.append(p5); pen_s += max(0.0, p5 - smax) ** 2
            out["sole_pen"] = pen_s / 2.0; out["swing_pitch"] = sp
            pen_t, tr = 0.0, []
            ttol = self.cfg["cost"].get("track_tol", 0.01)
            for k in range(4):
                stk = Fw[:, k] > 2.0
                if stk.sum() > 5:
                    ex = abs(float(np.median(tiplat[w][stk, k]))) - abs(float(np.median(jlat[w][stk, k]))); tr.append(ex); pen_t += max(0.0, ex - ttol) ** 2
            out["track_pen"] = pen_t / 4.0; out["track_excess"] = tr
            lm, pm = float(np.mean(lumb[w])), float(np.mean(pdiff[w]))
            out["spine_pen"] = max(0.0, abs(lm) - 10.0) ** 2 / 100.0 + max(0.0, abs(pm) - 8.0) ** 2 / 64.0
            out["lumb_mean"], out["pdiff_mean"] = lm, pm
            out["paw_pen"] = pen_p / max(n_p, 1)
            out["paw_pitch"] = pm
            bw = 10.24 * 9.81
            out["peak_pen"] = float(np.mean([max(0.0, Fw[:, k].max() - 1.5 * bw) ** 2 for k in range(4)]))
            out["fore_rom"] = [float(np.mean([np.percentile(fa[:, c], 98) - np.percentile(fa[:, c], 2) for c in (j, 6 + j)])) for j in range(3)]
            out["hip_joint_rom"] = float(np.mean([np.percentile(fa[:, c], 98) - np.percentile(fa[:, c], 2) for c in (3, 9)]))
            if self.cfg["cost"].get("w_shape"):
                from sim.shape import shape_error
                offs = {leg: ctl.offset[3 * k] for k, leg in enumerate(("LF", "RH", "RF", "LH"))}
                out["shape_err"] = shape_error(fa, t_win := (np.arange(n_used)[w] * dt), ctl.f, offs)
        else:
            out["win"] = None
            out["fore_rom"] = None
            out["hip_joint_rom"] = None
        out["cost"] = self.cost(out)
        if record:
            s = slice(0, n_used)
            out.update(time=np.arange(n_used) * dt, q=qlog[s], tau=tau_log[s], anchors=anch[s], grf=grf[s],
                       torso=pos[s], pelvis=pelv[s], yaw=yaw[s], up=up[s], freq=ctl.f, warmup=gc["warmup"])
        return out

    def cost(self, o: dict) -> float:
        c = self.cfg["cost"]
        pr = o["progress"]
        cost = c["w_speed"] * (1.0 - min(max(pr, 0.0), 1.0)) ** 2 + c["w_speed"] * max(0.0, pr - 1.15) ** 2
        if c.get("w_yawrate") and o.get("yaw_rate") is not None:
            cost += c["w_yawrate"] * (max(0.0, o["yaw_rate"] - c.get("yawrate_ok", 2.0)) / 3.0) ** 2
        cost += c["w_energy"] * o["energy"] + c["w_lateral"] * o["lat_rate"] + c["w_yaw"] * o["yaw_abs"]
        cost += c["w_tilt"] * o["tilt"]
        if o["fell"]:
            cost += c["fall_const"] + c["w_fall"] * (1.0 - o["t_fall"] / o["T"])
        if o["win"] is None:
            return float(cost + c["w_duty"] + c["w_load"])
        cost += c["w_duty"] * float(np.mean(((np.array(o["duty"]) - c["duty_target"]) / 0.15) ** 2))
        lo, hi = c["hind_load_range"]
        hl = o["hind_load"]
        cost += c["w_load"] * (max(0.0, lo - hl, hl - hi) / 0.1) ** 2
        cost += c["w_height"] * (max(0.0, c["min_height"] - o["z_mean"]) / 0.05) ** 2
        cal = c.get("calibrate")
        if cal:
            sc = c["calib_scale"]
            t = c["calib_targets"]
            w = o["win"]
            err = ((w["stifle"][1] - t["stifle_mea"]) / sc) ** 2 + ((w["stifle"][0] - t["stifle_mfa"]) / sc) ** 2
            err += ((w["hock"][1] - t["hock_mea"]) / sc) ** 2 + ((w["hock"][0] - t["hock_mfa"]) / sc) ** 2
            if c.get("calib_hip", True):
                err += (((w["hip"][1] - w["hip"][0]) - t["hip_rom"]) / sc) ** 2
            if c.get("calib_hip_joint") and o["hip_joint_rom"] is not None:
                err += ((o["hip_joint_rom"] - t["hip_rom"]) / sc) ** 2
            cost += c["w_calib"] * err
        if c.get("stride_target"):
            ratio = o["v_mean"] / o["freq"] / c["height_ref"]
            cost += c["w_stride"] * (max(0.0, c["stride_target"] - ratio) / 0.35) ** 2
        if c.get("w_freq") and o.get("freq") is not None:
            cost += c["w_freq"] * ((o["freq"] - c["freq_target"]) / 0.15) ** 2
        if c.get("calibrate_fore") and o["fore_rom"] is not None:
            ft = c["fore_targets"]
            tg = [ft["shoulder_rom"], ft["elbow_rom"], ft["wrist_rom"]]
            cost += c["w_fore"] * float(np.mean([((r - x) / c["fore_scale"]) ** 2 for r, x in zip(o["fore_rom"], tg)]))
        if c.get("w_tdlag") and o.get("td_lag") is not None:
            cost += c["w_tdlag"] * (max(0.0, o["td_lag"] + c.get("tdlag_target", 0.03)) / 0.04) ** 2
        if c.get("w_retract") and o.get("retract_pen") is not None:
            cost += c["w_retract"] * o["retract_pen"] / c.get("retract_scale", 40.0) ** 2
        if c.get("w_ref") and o.get("ref_err") is not None:
            cost += c["w_ref"] * o["ref_err"] ** 2 + c.get("w_carpus_rom", 0.0) * (max(0.0, c.get("carpus_rom_min", 45.0) - o["carpus_rom"]) / 10.0) ** 2
        if c.get("w_track") and o.get("track_pen") is not None:
            cost += c["w_track"] * o["track_pen"] / 0.01 ** 2
        if c.get("w_clear") and o.get("clear_pen") is not None:
            cost += c["w_clear"] * o["clear_pen"] / 0.02 ** 2
        if c.get("w_spine") and o.get("spine_pen") is not None:
            cost += c["w_spine"] * o["spine_pen"]
        if c.get("w_sole") and o.get("sole_pen") is not None:
            cost += c["w_sole"] * o["sole_pen"] / c.get("sole_scale", 20.0) ** 2
        if c.get("w_paw") and o.get("paw_pen") is not None:
            cost += c["w_paw"] * o["paw_pen"] / c.get("paw_scale", 15.0) ** 2
        if c.get("w_peak") and o.get("peak_pen") is not None:
            cost += c["w_peak"] * o["peak_pen"] / c.get("peak_scale", 50.0) ** 2
        if c.get("w_toeoff") and o.get("toeoff_pen") is not None:
            cost += c["w_toeoff"] * o["toeoff_pen"] / c.get("toeoff_scale", 15.0) ** 2
        if c.get("w_meta") and o.get("meta_pen") is not None:
            cost += c["w_meta"] * o["meta_pen"] / c["meta_scale"] ** 2
        if c.get("w_wobble") and o.get("wobble") is not None:
            cost += c["w_wobble"] * (max(0.0, o["wobble"] - c["wobble_ok"]) / c["wobble_ok"]) ** 2
        if c.get("w_shape") and o.get("shape_err") is not None:
            cost += c["w_shape"] * o["shape_err"] ** 2
        return float(cost)
