import argparse, sys
import imageio, mujoco, numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--seconds", type=float, default=8.0); ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--labels", nargs="+", default=None); a = ap.parse_args()
*runs, out = a.runs; imp = load_yaml("configs/impairment/patella.yaml")
W, H = (700 if len(runs) <= 2 else 560 if len(runs) <= 3 else 440), 520; font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 20); fsm = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 16)
o_bone = mujoco.MjvOption(); o_bone.geomgroup[:] = [1, 1, 1, 0, 1, 1]; o_bone.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
MUSCLE = (0.85, 0.20, 0.20, 0.45); FAT = (0.98, 0.85, 0.25, 0.30)
SEG = [("upper_leg_{s}", "lower_leg_{s}", 0.019, "hind"), ("lower_leg_{s}", "foot_{s}", 0.010, "hind"), ("upper_arm_{s}", "lower_arm_{s}", 0.015, "fore"), ("lower_arm_{s}", "hand_anchor_{s}", 0.009, "fore")]


def setup(run):
    d0 = ROOT / "results" / run; cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
    sim = Sim(cfg); m, d, sv = sim.m, sim.d, sim.sv; sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); q0 = d.qpos[sv["qadr"]].copy()
    ctl = TrotController(GaitSpec(cfg["gait"], list(z["free_legs"])), sim.name_to_k, q0); ctl.set_params(z["x"])
    body = cfg.get("body") or {}
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    trunk_geoms = [g for g in range(m.ngeom) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) in ("torso", "pelvis") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_ELLIPSOID]
    return dict(sim=sim, cfg=cfg, ctl=ctl, r=mujoco.Renderer(m, height=H, width=W), body=body, bid=bid, trunk=trunk_geoms, run=run)


def add_layers(S):
    sc = S["r"].scene; d = S["sim"].d; m = S["sim"].m; body = S["body"]; bid = S["bid"]
    def cap(p1, p2, rad, rgba):
        if sc.ngeom >= sc.maxgeom: return
        g = sc.geoms[sc.ngeom]; mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, rad, p1, p2); sc.ngeom += 1
    for s in "LR":
        for a_, b_, r0, kind in SEG:
            k = float(body.get("muscle_hind" if kind == "hind" else "muscle_fore", 1.0))
            cap(d.xpos[bid(a_.format(s=s))], d.xpos[bid(b_.format(s=s))], r0 * np.sqrt(k), MUSCLE)
    f = float(body.get("fat", 0.0))
    if f > 0:
        sf = (1.0 + f) ** (1 / 3)
        for g_ in S["trunk"]:
            if sc.ngeom >= sc.maxgeom: break
            g = sc.geoms[sc.ngeom]; size = m.geom_size[g_] * np.array([1.0, sf, sf]) * 1.08
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ELLIPSOID, size, d.geom_xpos[g_], d.geom_xmat[g_], np.array(FAT, np.float32)); sc.ngeom += 1


S = [setup(r_) for r_ in runs]; labels = a.labels or runs
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = (1.0 if len(runs) <= 2 else 1.25), 90.0, -5.0
dt = S[0]["sim"].m.opt.timestep; every = int(round(1.0 / (a.fps * dt))); frames = []
for i in range(int((2.0 + a.seconds) / dt)):
    t = i * dt
    for s in S:
        sim = s["sim"]; m, d, sv = sim.m, sim.d, sim.sv
        d.ctrl[sv["act_ids"]] = s["ctl"].targets(t, min(1.0, t / s["cfg"]["gait"]["ramp_in"]))
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
    if t >= 2.0 and i % every == 0:
        tiles = []
        for s, lab in zip(S, labels):
            sim = s["sim"]; d = sim.d; cam.lookat[:] = [d.xpos[sim.torso][0], d.xpos[sim.torso][1], 0.18]
            s["r"].update_scene(d, camera=cam, scene_option=o_bone); add_layers(s); im = Image.fromarray(s["r"].render().copy()); dr = ImageDraw.Draw(im)
            b = s["body"]; dr.text((8, 6), lab, font=font, fill=(255, 255, 0))
            dr.text((8, 32), "근력 %.1f배  지방 +%d%%" % (float(b.get("muscle_hind", 1.0)), round(100 * float(b.get("fat", 0.0)))), font=fsm, fill=(255, 220, 160)); tiles.append(np.array(im))
        im = Image.fromarray(np.hstack(tiles)); ImageDraw.Draw(im).text((8, H - 24), "근육(빨강)과 지방(노랑)은 모식적 시각화이며 해부학적 형태가 아님", font=fsm, fill=(220, 220, 220)); frames.append(np.array(im))
imageio.mimsave(out, frames, fps=a.fps, macro_block_size=1); print("wrote", out, len(frames), "frames")
for s in S: s["r"].close()
