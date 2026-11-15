import numpy as np
from PIL import Image, ImageFilter, ImageDraw

im = Image.open("data/real/leo_xray/xr3.jpg").convert("L"); g = np.asarray(im.filter(ImageFilter.GaussianBlur(6)), np.float32)
hist, edges = np.histogram(g, bins=256, range=(0, 255)); p = hist / hist.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
sb = (mt * w - mu) ** 2 / (w * (1 - w) + 1e-12); thr = float(np.argmax(sb))
print("Otsu threshold", thr)
VD = {"left (image right)": dict(hip=(1410, 1092), stifle=(1511, 1867), out=+1), "right (image left)": dict(hip=(876, 1054), stifle=(724, 1829), out=-1)}
vis = Image.open("data/real/leo_xray/xr3.jpg").convert("RGB"); dr = ImageDraw.Draw(vis); res = {}
for k, v in VD.items():
    hip, st = np.array(v["hip"], float), np.array(v["stifle"], float); flen = float(np.linalg.norm(hip - st))
    out = []
    for frac in (0.4, 0.5, 0.6):
        c = hip + frac * (st - hip); x0, y = int(round(c[0])), int(round(c[1]))
        x = x0
        while 0 < x < g.shape[1] - 1 and g[y, x] > thr * 0.9: x += v["out"]
        out.append(abs(x - x0)); dr.line([(x0, y), (x, y)], fill=(255, 60, 60), width=6); dr.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 255, 0), width=4)
    res[k] = (flen, out); print(k, "femur len %.0f px; lateral soft-tissue thickness at 40/50/60 %% of the femur: %s px; ratio %s" % (flen, out, [round(o / flen, 2) for o in out]))
vis.thumbnail((900, 1100)); vis.save("results/figures/xray_thigh_index.png")
