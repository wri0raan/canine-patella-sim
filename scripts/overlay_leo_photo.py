import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
from sim.config import load_yaml
from sim.rollout import Sim

photo, kpf, out = sys.argv[1], sys.argv[2], sys.argv[3]
txs = [float(v) for v in sys.argv[4:]]
z = np.load(kpf); kp, names = z["kp"][0], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
g = lambda n: kp[ix[n], :2]; c = lambda n: kp[ix[n], 2]
img = Image.open(photo).convert("RGB"); img.thumbnail((1920, 1920)); img = img.resize((img.width // 2 * 2, img.height // 2 * 2))
facing = 1.0 if g("nose")[0] > g("tail_base")[0] else -1.0
best = None
for hs, fs in (("back_left", "front_left"), ("back_right", "front_right")):
    cs = min(c(hs + "_paw"), c(hs + "_knee"), c(hs + "_thai"), c(fs + "_paw"))
    if best is None or cs > best[0]: best = (cs, hs, fs)
_, hs, fs = best
ground = max(g(hs + "_paw")[1], g(fs + "_paw")[1])
s_cm_px = 0.5 * (16.0 / (ground - g(hs + "_thai")[1]) + 8.5 / (ground - g(hs + "_knee")[1]))
px_per_m = 100.0 / s_cm_px
print("near side", hs, "| ground y %.0f | %.3f cm/px (%.0f px/m) | facing %s" % (ground, s_cm_px, px_per_m, "right" if facing > 0 else "left"))

leo = load_yaml("configs/model/leo.yaml")
panels = []
for tx in txs:
    from sim.dog_scale import apply_leo
    cfg = apply_leo(load_yaml("configs/gait/trot.yaml")); cfg["model"]["trunk_x"] = tx
    sim = Sim(cfg); m, d, sv = sim.m, sim.d, sim.sv
    imp = load_yaml("configs/impairment/patella.yaml"); sim.limiter.set_grade(imp, 0, "functional", ("LH", "RH"))
    mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); q = d.qpos[sv["qadr"]].copy()
    for i in range(int(2.0 / m.opt.timestep)):
        d.ctrl[sv["act_ids"]] = q
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
    D, fovy = 20.0, 2.0; H = W = 1000
    m.vis.global_.fovy = fovy; m.vis.global_.offwidth = 1600; m.vis.global_.offheight = 1600
    scale = H / (2 * D * np.tan(np.radians(fovy) / 2))
    r = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = D, 90.0, 0.0
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    cx = d.xpos[sim.torso][0]; cam.lookat[:] = [cx, d.xpos[sim.torso][1], 0.2]
    on = mujoco.MjvOption(); on.geomgroup[:] = [0, 0, 0, 0, 1, 1]; on.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
    off = mujoco.MjvOption(); off.geomgroup[:] = [0, 0, 0, 0, 0, 0]; off.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
    r.update_scene(d, camera=cam, scene_option=on); a = r.render().astype(int)
    r.update_scene(d, camera=cam, scene_option=off); b = r.render().astype(int)
    r.close()
    mask = (np.abs(a - b).sum(2) > 30)
    wx = lambda x: W / 2 + (x - cx) * scale; wz = lambda zc: H / 2 - (zc - 0.2) * scale
    paw = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "finger_L")]
    f = (px_per_m / scale)
    sk = Image.fromarray(np.dstack([a.astype(np.uint8), (mask * 215).astype(np.uint8)]), "RGBA")
    sk = sk.resize((int(W * f), int(H * f)), Image.LANCZOS)
    fx, fy = g(fs + "_paw")
    ox = int(fx - wx(paw[0]) * f); oy = int(ground - wz(0.0) * f)
    if facing < 0:
        sk = sk.transpose(Image.FLIP_LEFT_RIGHT); ox = int(fx - (W - wx(paw[0])) * f)
    canvas = img.convert("RGBA"); canvas.alpha_composite(sk, (max(ox, 0), max(oy, 0)) if ox >= 0 and oy >= 0 else (0, 0)) if (ox >= 0 and oy >= 0) else None
    if ox < 0 or oy < 0:
        lay = Image.new("RGBA", canvas.size, (0, 0, 0, 0)); lay.paste(sk, (ox, oy), sk); canvas = Image.alpha_composite(canvas, lay)
    jx = lambda n: (ox + wx(d.xanchor[jid(n)][0]) * f, oy + wz(d.xanchor[jid(n)][2]) * f)
    dr = ImageDraw.Draw(canvas)
    for n, col in (("hip_L_extend", (255, 60, 60, 255)), ("shoulder_L_extend", (60, 160, 255, 255)), ("knee_L", (255, 200, 0, 255))):
        x, y = jx(n); dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=col, width=3)
    font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22); dr.text((10, 10), f"trunk_x {tx:.2f}  (red = hip joint, blue = shoulder, yellow = stifle)", font=font, fill=(255, 255, 0, 255))
    xs = [g(n)[0] for n in names if c(n) > 0.3]; ys = [g(n)[1] for n in names if c(n) > 0.3]
    box = (int(max(0, min(xs) - 220)), int(max(0, min(ys) - 160)), int(min(canvas.width, max(xs) + 220)), int(min(canvas.height, ground + 80)))
    panels.append(canvas.convert("RGB").crop(box))
    print("trunk_x %.2f: hip joint x %.0f px, shoulder x %.0f px (photo front paw x %.0f)" % (tx, jx("hip_L_extend")[0], jx("shoulder_L_extend")[0], fx))
Wt = sum(p.width for p in panels); Ht = max(p.height for p in panels); sheet = Image.new("RGB", (Wt, Ht))
xo = 0
for p in panels: sheet.paste(p, (xo, 0)); xo += p.width
sheet.save(out); print("saved", out)
