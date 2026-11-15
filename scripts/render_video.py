import argparse
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--seconds", type=float, default=6.0)
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
ap.add_argument("--out", default=None)
a = ap.parse_args()

d0 = ROOT / "results" / a.run
cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8"))
imp = load_yaml("configs/impairment/patella.yaml")
z = np.load(d0 / "best.npz", allow_pickle=True)
spec = GaitSpec(cfg["gait"], list(z["free_legs"]))
sim = Sim(cfg)
sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
m, d, sv = sim.m, sim.d, sim.sv

mujoco.mj_resetData(m, d)
sim.limiter.clip_qpos(d)
q_hold = d.qpos[sv["qadr"]].copy()
ctl = TrotController(spec, sim.name_to_k, q_hold)
ctl.set_params(z["x"])

renderer = mujoco.Renderer(m, height=a.height, width=a.width)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 1.6, 90.0, -8.0
out = a.out or str(d0 / "side.mp4")
frames = []
n = int(a.seconds / m.opt.timestep)
every = int(round(1.0 / (a.fps * m.opt.timestep)))
for i in range(n):
    t = i * m.opt.timestep
    ramp = min(1.0, t / cfg["gait"]["ramp_in"])
    d.ctrl[sv["act_ids"]] = ctl.targets(t, ramp)
    if sim.head:
        d.ctrl[sim.head_act] = sim.head_ctrl(d)
    mujoco.mj_step(m, d)
    if i % every == 0:
        cam.lookat[:] = d.xpos[sim.torso] * np.array([1, 1, 0]) + np.array([0, 0, 0.25])
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render())
imageio.mimsave(out, frames, fps=a.fps, macro_block_size=1)
print("wrote", out, len(frames), "frames")
