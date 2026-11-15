import argparse, sys
import imageio, mujoco, numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--seconds", type=float, default=10.0); ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--labels", nargs="+", default=None); a = ap.parse_args()
*runs, out = a.runs; imp = load_yaml("configs/impairment/patella.yaml")
W, H = (700 if len(runs) <= 2 else 480 if len(runs) <= 4 else 400), 520; font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)
o_bone = mujoco.MjvOption(); o_bone.geomgroup[:] = [1, 1, 1, 0, 1, 1]; o_bone.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0


def setup(run):
    d0 = ROOT / "results" / run; cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
    sim = Sim(cfg); m, d, sv = sim.m, sim.d, sim.sv; sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); q0 = d.qpos[sv["qadr"]].copy()
    ctl = TrotController(GaitSpec(cfg["gait"], list(z["free_legs"])), sim.name_to_k, q0); ctl.set_params(z["x"])
    return dict(sim=sim, cfg=cfg, ctl=ctl, r=mujoco.Renderer(m, height=H, width=W), x0=None, run=run)


S = [setup(r_) for r_ in runs]
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = (1.0 if len(runs) <= 2 else 1.25), 90.0, -5.0
dt = S[0]["sim"].m.opt.timestep; every = int(round(1.0 / (a.fps * dt))); frames = []; labels = a.labels or runs
for i in range(int(a.seconds / dt)):
    t = i * dt
    for s in S:
        sim = s["sim"]; m, d, sv = sim.m, sim.d, sim.sv
        d.ctrl[sv["act_ids"]] = s["ctl"].targets(t, min(1.0, t / s["cfg"]["gait"]["ramp_in"]))
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
        if s["x0"] is None and t >= 2.0: s["x0"] = (d.xpos[sim.torso][0], t)
    if i % every == 0:
        tiles = []
        for s, lab in zip(S, labels):
            sim = s["sim"]; d = sim.d; cam.lookat[:] = [d.xpos[sim.torso][0], d.xpos[sim.torso][1], 0.18]
            s["r"].update_scene(d, camera=cam, scene_option=o_bone); im = Image.fromarray(s["r"].render().copy()); dr = ImageDraw.Draw(im)
            v = (d.xpos[sim.torso][0] - s["x0"][0]) / (t - s["x0"][1]) if s["x0"] is not None and t > s["x0"][1] + 1 else float("nan")
            dr.text((8, 6), lab + ("  (%.2f m/s)" % v if v == v else ""), font=font, fill=(255, 255, 0)); tiles.append(np.array(im))
        im = Image.fromarray(np.hstack(tiles)); ImageDraw.Draw(im).text((W - 40, H - 30), f"{t:4.1f} s", font=font, fill=(255, 255, 255)); frames.append(np.array(im))
imageio.mimsave(out, frames, fps=a.fps, macro_block_size=1); print("wrote", out, len(frames), "frames")
for s in S: s["r"].close()
