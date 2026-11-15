import argparse, sys
import imageio, mujoco, numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from sim.config import ROOT, load_yaml
from sim.dog_scale import apply_leo
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--seconds", type=float, default=12.0); ap.add_argument("--fps", type=int, default=60)
a = ap.parse_args(); imp = load_yaml("configs/impairment/patella.yaml")
if a.run == "stand":
    cfg = apply_leo(load_yaml("configs/gait/trot.yaml")); z = None
else:
    d0 = ROOT / "results" / a.run; cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
sim = Sim(cfg); m, d, sv = sim.m, sim.d, sim.sv; sim.limiter.set_grade(imp, 0, "functional", ("LH", "RH")) if z is None else sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); q0 = d.qpos[sv["qadr"]].copy()
ctl = None
if z is not None:
    ctl = TrotController(GaitSpec(cfg["gait"], list(z["free_legs"])), sim.name_to_k, q0); ctl.set_params(z["x"])
W, H = 640, 520; r = mujoco.Renderer(m, height=H, width=W); font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)
o_skin = mujoco.MjvOption(); o_skin.geomgroup[:] = [1, 1, 0, 0, 0, 0]; o_skin.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1
o_bone = mujoco.MjvOption(); o_bone.geomgroup[:] = [1, 1, 1, 0, 1, 1]; o_bone.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 0.95, 90.0, -5.0
every = int(round(1.0 / (a.fps * m.opt.timestep))); frames = []
def shot(label_t=None):
    cam.lookat[:] = [d.xpos[sim.torso][0], d.xpos[sim.torso][1], 0.18]; tiles = []
    for o in (o_skin, o_bone):
        r.update_scene(d, camera=cam, scene_option=o); tiles.append(r.render().copy())
    im = Image.fromarray(np.hstack(tiles)); dr = ImageDraw.Draw(im); dr.text((8, 6), "피부", font=font, fill=(255, 255, 0)); dr.text((W + 8, 6), "뼈대", font=font, fill=(255, 255, 0))
    if label_t is not None: dr.text((1100, 6), f"{label_t:4.1f} s", font=font, fill=(255, 255, 255))
    return np.array(im)
if z is None:
    for i in range(int(2.0 / m.opt.timestep)):
        d.ctrl[sv["act_ids"]] = q0
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
    Image.fromarray(shot()).save(a.out); print("saved", a.out)
else:
    for i in range(int(a.seconds / m.opt.timestep)):
        t = i * m.opt.timestep; d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
        if i % every == 0: frames.append(shot(t))
    imageio.mimsave(a.out, frames, fps=a.fps, macro_block_size=1); print("wrote", a.out, len(frames), "frames")
r.close()
