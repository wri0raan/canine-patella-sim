import argparse, sys
import mujoco, numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--t", type=float, default=6.0); a = ap.parse_args()
d0 = ROOT / "results" / a.run; cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
imp = load_yaml("configs/impairment/patella.yaml"); sim = Sim(cfg); sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
m, d, sv = sim.m, sim.d, sim.sv; mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d)
ctl = TrotController(GaitSpec(cfg["gait"], list(z["free_legs"])), sim.name_to_k, d.qpos[sv["qadr"]].copy()); ctl.set_params(z["x"])
r = mujoco.Renderer(m, height=520, width=520); font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 20)
os_ = mujoco.MjvOption(); os_.geomgroup[:] = [1, 1, 0, 0, 0, 0]; ob = mujoco.MjvOption(); ob.geomgroup[:] = [1, 1, 1, 0, 1, 1]; ob.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 1.0, 90.0, -89.0
jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n); bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
legs = {"LF": ("shoulder_L_extend", "finger_L"), "RF": ("shoulder_R_extend", "finger_R"), "LH": ("hip_L_extend", "toe_L"), "RH": ("hip_R_extend", "toe_R")}
rec = {k: [] for k in legs}
for i in range(int((a.t + 4.0) / m.opt.timestep)):
    t = i * m.opt.timestep; d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
    if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
    mujoco.mj_step(m, d)
    if t >= 3 and i % 25 == 0:
        R = d.xmat[sim.torso].reshape(3, 3); fw = R[:, 0].copy(); fw[2] = 0; fw /= np.linalg.norm(fw); lat = np.array([-fw[1], fw[0], 0])
        for k, (jn, bn) in legs.items():
            rec[k].append(((d.xpos[bid(bn)] - d.xanchor[jid(jn)]) @ lat, (d.xpos[bid(bn)] - d.xanchor[jid(jn)]) @ fw, abs(d.xpos[bid(bn)][2] - 0) ))
    if abs(t - a.t) < 1e-6 + m.opt.timestep / 2:
        cam.lookat[:] = [d.xpos[sim.torso][0], d.xpos[sim.torso][1], 0.0]; tiles = []
        for o in (os_, ob):
            r.update_scene(d, camera=cam, scene_option=o); tiles.append(r.render().copy())
        im = Image.fromarray(np.hstack(tiles)); dr = ImageDraw.Draw(im); dr.text((8, 6), "탑뷰: 피부", font=font, fill=(255, 255, 0)); dr.text((528, 6), "탑뷰: 뼈대", font=font, fill=(255, 255, 0)); im.save(a.out)
r.close()
print("paw position relative to its shoulder/hip joint (cm), median over strides: lateral (+ = outward is sign of side) and forward; splay angle from the vertical plane")
for k in legs:
    lat, fwd = np.median([x[0] for x in rec[k]]), np.median([x[1] for x in rec[k]]); side = 1 if k[0] == "L" else -1
    print("  %s lateral %+5.1f cm (outward %+5.1f) forward %+5.1f cm" % (k, 100 * lat, 100 * lat * side, 100 * fwd))
print("saved", a.out)
