import sys
import numpy as np
sys.path.insert(0, ".")
import mujoco
from sim.config import load_yaml
from sim.dog_scale import apply_leo
from sim.rollout import Sim
from PIL import Image, ImageDraw

LM = {
    "xr4_L": dict(hip=(1244, 585), stifle=(927, 1256), plateau=(945, 1343), hock=(1343, 2071)),
    "xr6_R": dict(hip=(1065, 647), stifle=(894, 1400), plateau=(918, 1506), hock=(1388, 2200)),
}
VD = {"xr3_L": dict(hip=(1410, 1092), stifle=(1511, 1867)), "xr3_R": dict(hip=(876, 1054), stifle=(724, 1829))}
d = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
rows = []
for k, v in LM.items():
    fem, tib = d(v["hip"], v["stifle"]), d(v["plateau"], v["hock"]); rows.append((k, fem, tib, fem / tib)); print(f"{k}: femur {fem:.0f} px, tibia {tib:.0f} px, femur/tibia {fem / tib:.2f}")
for k, v in VD.items(): print(f"{k} (VD, femur only): {d(v['hip'], v['stifle']):.0f} px")
ratio = np.mean([r[3] for r in rows]); print(f"mean femur/tibia (lateral views): {ratio:.2f}  (range {min(r[3] for r in rows):.2f}-{max(r[3] for r in rows):.2f})")
cfg = apply_leo(load_yaml("configs/gait/trot.yaml")); s = Sim(cfg); m, dd = s.m, s.d; mujoco.mj_forward(m, dd)
P = lambda n: dd.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)]
fem_m = float(np.linalg.norm(P("upper_leg_L") - P("lower_leg_L"))); tib_m = float(np.linalg.norm(P("lower_leg_L") - P("foot_L")))
print(f"model (leo scale): femur {fem_m:.3f} m, tibia {tib_m:.3f} m, ratio {fem_m / tib_m:.2f}")
tib_photo = 0.127
tib_px = np.mean([r[2] for r in rows]); fem_px = np.mean([r[1] for r in rows]); mpp = tib_photo / tib_px
print(f"scale if tibia = {tib_photo} m: {1000 * mpp:.3f} mm/px; implied femur {fem_px * mpp:.3f} m vs model {fem_m:.3f} m ({100 * (fem_px * mpp / fem_m - 1):+.0f} %)")
for k, name in (("xr4_L", "xr4"), ("xr6_R", "xr6")):
    im = Image.open(f"data/real/leo_xray/{name}.jpg").convert("RGB"); dr = ImageDraw.Draw(im); v = LM[k]
    for a, b, col in (("hip", "stifle", (255, 80, 80)), ("plateau", "hock", (80, 255, 120))): dr.line([v[a], v[b]], fill=col, width=8)
    for p in v.values(): dr.ellipse([p[0] - 14, p[1] - 14, p[0] + 14, p[1] + 14], outline=(255, 255, 0), width=5)
    im.thumbnail((900, 1100)); im.save(f"results/figures/xray_landmarks_{name}.png")
