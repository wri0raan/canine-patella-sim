import argparse
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("items", nargs="+", help="label=run")
ap.add_argument("--cols", type=int, default=2)
ap.add_argument("--seconds", type=float, default=5.0)
ap.add_argument("--skip", type=float, default=2.0, help="seconds to run before recording")
ap.add_argument("--w", type=int, default=640)
ap.add_argument("--h", type=int, default=360)
ap.add_argument("--sample_fps", type=int, default=60)
ap.add_argument("--play_fps", type=int, default=30, help="playback fps; sample/play = slow-motion factor")
ap.add_argument("--lang", default="ko"); ap.add_argument("--bone", action="store_true", help="draw the skeleton instead of the skin")
a = ap.parse_args()
font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)
small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 17)
imp = load_yaml("configs/impairment/patella.yaml")
OPT = mujoco.MjvOption()
if a.bone:
    OPT.geomgroup[:] = [1, 1, 1, 0, 1, 1]; OPT.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0


def render_run(run):
    d0 = ROOT / "results" / run
    cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8"))
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
    r = mujoco.Renderer(m, height=a.h, width=a.w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.6, 90.0, -8.0
    every = int(round(1.0 / (a.sample_fps * m.opt.timestep)))
    n = int((a.skip + a.seconds) / m.opt.timestep)
    frames = []
    for i in range(n):
        t = i * m.opt.timestep
        d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
        if sim.head:
            d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
        if t >= a.skip and i % every == 0:
            cam.lookat[:] = d.xpos[sim.torso] * np.array([1, 1, 0]) + np.array([0, 0, 0.25])
            r.update_scene(d, camera=cam, scene_option=OPT)
            frames.append(r.render().copy())
    return frames


tiles = []
for it in a.items:
    label, run = it.split("=", 1)
    print("render", run, flush=True)
    fr = render_run(run)
    out = []
    for k, f in enumerate(fr):
        im = Image.fromarray(f); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, a.w, 34], fill=(0, 0, 0)); dr.text((8, 4), label, font=font, fill=(255, 255, 255))
        out.append(np.array(im))
    tiles.append(out)
n = min(len(t) for t in tiles)
rows = []
for i in range(n):
    cells = [t[i] for t in tiles]
    while len(cells) % a.cols:
        cells.append(np.zeros_like(cells[0]))
    rows.append(np.vstack([np.hstack(cells[r:r + a.cols]) for r in range(0, len(cells), a.cols)]))
slow = a.sample_fps / a.play_fps
for i in range(n):
    im = Image.fromarray(rows[i]); ImageDraw.Draw(im).text((im.width - 150, im.height - 26), f"{1/slow:.1f}x " + ("속도" if a.lang == "ko" else "speed"), font=small, fill=(255, 255, 0))
    rows[i] = np.array(im)
imageio.mimsave(a.out, rows, fps=a.play_fps, macro_block_size=1)
print("wrote", a.out, n, "frames")
