import argparse, sys
import imageio, mujoco, numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--seconds", type=float, default=12.0); ap.add_argument("--fps", type=int, default=60)
a = ap.parse_args()
d0 = ROOT / "results" / a.run; cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
imp = load_yaml("configs/impairment/patella.yaml")
sim = Sim(cfg); sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"])); m, d, sv = sim.m, sim.d, sim.sv
spec = GaitSpec(cfg["gait"], list(z["free_legs"])); mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d)
ctl = TrotController(spec, sim.name_to_k, d.qpos[sv["qadr"]].copy()); ctl.set_params(z["x"])
r = mujoco.Renderer(m, height=540, width=640); font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 20)
opts = []
for groups in ([1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1]):
    o = mujoco.MjvOption(); o.geomgroup[:] = groups; o.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0; opts.append(o)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 1.0, 90.0, -6.0
every = int(round(1.0 / (a.fps * m.opt.timestep))); frames = []
for i in range(int(a.seconds / m.opt.timestep)):
    t = i * m.opt.timestep; d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
    if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
    mujoco.mj_step(m, d)
    if i % every == 0:
        cam.lookat[:] = [d.xpos[sim.torso][0], d.xpos[sim.torso][1], 0.18]
        tiles = []
        for o in opts:
            r.update_scene(d, camera=cam, scene_option=o); tiles.append(r.render().copy())
        im = Image.fromarray(np.hstack(tiles)); dr = ImageDraw.Draw(im)
        dr.text((8, 6), f"{a.run}: 도형", font=font, fill=(255, 255, 0)); dr.text((648, 6), "뼈대", font=font, fill=(255, 255, 0)); dr.text((1000, 6), f"t = {t:4.1f} s", font=font, fill=(255, 255, 255))
        frames.append(np.array(im))
r.close(); imageio.mimsave(a.out, frames, fps=a.fps, macro_block_size=1); print("wrote", a.out, len(frames), "frames")
