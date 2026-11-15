import argparse, sys
import imageio, mujoco, numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--leo-start", type=float, default=43.7)
ap.add_argument("--seconds", type=float, default=3.0); ap.add_argument("--sim-start", type=float, default=4.0); a = ap.parse_args()
W, H = 640, 480
kp = np.load(ROOT / "data/real/leo2/work/leo_kp.npz")["kp"]; r = imageio.get_reader(str(ROOT / "data/real/leo2/leo_1080p120.mp4"))
f0 = int(a.leo_start * 120); n = int(a.seconds * 120); leo = []
ctr = None
for k in range(n):
    i = f0 + k; fr = Image.fromarray(r.get_data(i)); ok = kp[i, :, 2] > 0.3
    cx, cy = (np.median(kp[i, ok, 0]), np.median(kp[i, ok, 1])) if ok.sum() > 5 else (ctr if ctr else (960, 540))
    ctr = (cx, cy) if ctr is None else (0.9 * ctr[0] + 0.1 * cx, 0.9 * ctr[1] + 0.1 * cy)
    x0 = int(np.clip(ctr[0] - 360, 0, 1920 - 720)); y0 = int(np.clip(ctr[1] - 300, 0, 1080 - 540))
    leo.append(np.array(fr.crop((x0, y0, x0 + 720, y0 + 540)).resize((W, H))))
r.close()
d0 = ROOT / "results" / a.run; cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
imp = load_yaml("configs/impairment/patella.yaml"); sim = Sim(cfg); sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
m, d, sv = sim.m, sim.d, sim.sv; mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d)
ctl = TrotController(GaitSpec(cfg["gait"], list(z["free_legs"])), sim.name_to_k, d.qpos[sv["qadr"]].copy()); ctl.set_params(z["x"])
rd = mujoco.Renderer(m, height=H, width=W); o = mujoco.MjvOption(); o.geomgroup[:] = [1, 1, 1, 0, 1, 1]; o.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 0.95, 90.0, -4.0
step = 4; simf = []
for i in range(int((a.sim_start + a.seconds) / m.opt.timestep) + 1):
    t = i * m.opt.timestep; d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
    if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
    mujoco.mj_step(m, d)
    if t >= a.sim_start and i % step == 0 and len(simf) < n * 0.5 * 2:
        cam.lookat[:] = [d.xpos[sim.torso][0], d.xpos[sim.torso][1], 0.17]; rd.update_scene(d, camera=cam, scene_option=o); simf.append(rd.render().copy())
rd.close(); font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22); out = []
for k in range(min(len(leo), len(simf))):
    im = Image.fromarray(np.hstack([leo[k], simf[k]])); dr = ImageDraw.Draw(im)
    dr.text((8, 6), "레오 영상 (0.5배속)", font=font, fill=(255, 255, 0)); dr.text((W + 8, 6), f"시뮬레이션 {a.run} (0.5배속)", font=font, fill=(255, 255, 0)); out.append(np.array(im))
imageio.mimsave(a.out, out, fps=60, macro_block_size=1); print("wrote", a.out, len(out), "frames")
