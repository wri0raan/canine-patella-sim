import sys
import mujoco, numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, ".")
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
plt.rcParams["font.family"] = "Malgun Gothic"
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 100
Z = np.load(ROOT / "results" / "bc_motion_fitR3.npz", allow_pickle=True); t_target = float(Z["t"][idx]); print("frame", idx, "t =", t_target)
imp = load_yaml("configs/impairment/patella.yaml"); d0 = ROOT / "results" / "fitR3"
cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
sim = Sim(cfg); m, d, sv = sim.m, sim.d, sim.sv
sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); q0 = d.qpos[sv["qadr"]].copy()
ctl = TrotController(GaitSpec(cfg["gait"], list(z["free_legs"])), sim.name_to_k, q0); ctl.set_params(z["x"])
for i in range(int(round(t_target / m.opt.timestep)) + 1):
    t = i * m.opt.timestep; d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
    if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
    mujoco.mj_step(m, d)
W, H = 800, 450; m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
r = mujoco.Renderer(m, height=H, width=W)
o_bone = mujoco.MjvOption(); o_bone.geomgroup[:] = [1, 1, 1, 0, 1, 1]; o_bone.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 1.0, 90.0, -5.0
cam.lookat[:] = [d.xpos[sim.torso][0], d.xpos[sim.torso][1], 0.18]
r.update_scene(d, camera=cam, scene_option=o_bone); bone = r.render().copy()
o_skin = mujoco.MjvOption(); o_skin.geomgroup[:] = [1, 1, 0, 0, 0, 0]; o_skin.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1
r.update_scene(d, camera=cam, scene_option=o_skin); skin = r.render().copy(); r.close()
CROP = (slice(20, 420), slice(110, 690)); bone = bone[CROP]; bc = skin[CROP]
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.3))
for a, im, ttl in ((ax[0], bone, "(a)"), (ax[1], bc, "(b)")):
    a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=12, fontweight="bold", loc="left")
    for s in a.spines.values(): s.set_color("#999999"); s.set_linewidth(0.8)
plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04)
plt.savefig("results/figures/thesis/F26_skeleton_skin.png", dpi=220); plt.close(); print("F26")
