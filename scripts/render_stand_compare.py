import sys
import mujoco, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from sim.config import load_yaml
from sim.rollout import Sim
imp = load_yaml("configs/impairment/patella.yaml"); leo = load_yaml("configs/model/leo.yaml")
rows = []
font = ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 20)
opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 0, 1, 0, 0]; opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
for label, mc in (("원래 모델 (10.2 kg, 어깨 약 52 cm)", None), ("레오 체형 (7 kg, 어깨 약 33 cm)", leo)):
    cfg = load_yaml("configs/gait/trot.yaml"); cfg["model"] = mc
    sim = Sim(cfg); m, d, sv = sim.m, sim.d, sim.sv; sim.limiter.set_grade(imp, 0, "functional", ("LH", "RH"))
    mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); q = d.qpos[sv["qadr"]].copy()
    for i in range(int(2.0 / m.opt.timestep)):
        d.ctrl[sv["act_ids"]] = q
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, height=420, width=560); tiles = []
    for az, el in ((90, -5), (45, -12), (0, -5)):
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 1.5, az, el
        cam.lookat[:] = [d.xpos[sim.torso][0], d.xpos[sim.torso][1], 0.22]
        r.update_scene(d, camera=cam, scene_option=opt); tiles.append(r.render().copy())
    r.close()
    row = np.hstack(tiles); im = Image.fromarray(row); ImageDraw.Draw(im).text((8, 6), label, font=font, fill=(255, 255, 0)); rows.append(np.array(im))
Image.fromarray(np.vstack(rows)).save(sys.argv[1]); print("saved", sys.argv[1])
