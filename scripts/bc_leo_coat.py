import numpy as np
from PIL import Image, ImageFilter

Z = np.load("results/bc_mesh_uv.npz"); V = Z["V"]; tri_v = Z["tri_v"]; tri_l = Z["tri_l"]; uv = Z["uv"]
x, y, z = V[:, 0], V[:, 1], V[:, 2]
sig = lambda t, w: 1 / (1 + np.exp(-t / w))
def noise(a, b, c): return 0.014 * np.sin(38 * a + 11) + 0.012 * np.sin(27 * b + 40 * c) + 0.010 * np.sin(55 * c + 17 * a)
yy = y + noise(x, z, y); zz = z + noise(y, x, z)
white = np.zeros(len(V))
saddle_line = 0.40
body = sig(yy + 0.17, 0.008) * sig(0.17 - yy, 0.008)
white = np.maximum(white, body * sig(saddle_line - zz, 0.012))
white = np.maximum(white, sig(-0.17 - yy, 0.012) * sig(0.43 - zz, 0.015) * sig(yy + 0.42, 0.02))
white = np.maximum(white, sig(-0.19 - yy, 0.012) * sig(yy + 0.37, 0.015) * sig(0.56 - zz, 0.015) * sig(np.abs(x) - 0.012, 0.006))
white = np.maximum(white, sig(-0.08 - yy, 0.012) * sig(0.38 - zz, 0.012))
white = np.maximum(white, sig(yy - 0.05, 0.012) * sig(0.23 - zz, 0.012))
white = np.maximum(white, sig(yy - 0.05, 0.012) * sig(0.20 - yy, 0.012) * sig(0.40 - zz, 0.012) * sig(zz - 0.22, 0.012))
white = np.maximum(white, sig(yy - 0.38, 0.02))
head = sig(-0.40 - y, 0.01); white = white * (1 - head)
white = np.clip(white, 0, 1)

N = 2048; mask = np.zeros((N, N), np.float32)
U = uv * [N - 1, N - 1]; U[:, 1] = (N - 1) - U[:, 1]
for tv, tl in zip(tri_v, tri_l):
    p = U[tl]; w = white[tv]
    x0, y0 = np.floor(p.min(0)).astype(int) - 1; x1, y1 = np.ceil(p.max(0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, N - 1), min(y1, N - 1)
    if x1 <= x0 or y1 <= y0: continue
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)); q = np.stack([gx, gy], -1).astype(np.float32)
    a, b, c = p[0], p[1], p[2]; den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(den) < 1e-9: continue
    l1 = ((b[1] - c[1]) * (q[..., 0] - c[0]) + (c[0] - b[0]) * (q[..., 1] - c[1])) / den
    l2 = ((c[1] - a[1]) * (q[..., 0] - c[0]) + (a[0] - c[0]) * (q[..., 1] - c[1])) / den; l3 = 1 - l1 - l2
    inside = (l1 >= -0.05) & (l2 >= -0.05) & (l3 >= -0.05)
    val = l1 * w[0] + l2 * w[1] + l3 * w[2]
    sub = mask[y0:y1 + 1, x0:x1 + 1]; sub[inside] = val[inside]
m_img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5)); mask = np.clip((np.asarray(m_img, np.float32) / 255 - 0.5) * 2.4 + 0.5, 0, 1)
old = np.asarray(Image.open("data/models/bordercollie_textures/texture/BorderCollie_Albedo_1.png").convert("RGBA"), np.float32) / 255
lum = old[..., :3].mean(-1); bright = lum > 0.62
wmean = old[..., :3][bright].mean(0) if bright.sum() > 1000 else np.array([0.86, 0.85, 0.82])
rng = np.random.default_rng(3); n1 = np.asarray(Image.fromarray((rng.random((N, N)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
n2 = np.asarray(Image.fromarray((rng.random((N, N)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), np.float32) / 255
detail = 0.9 + 0.22 * (n1 - 0.5) / 0.15 * 0.25 + 0.2 * (n2 - 0.5)
whitefur = np.clip(wmean[None, None, :] * 1.36 * detail[..., None], 0, 1)
m3 = mask[..., None]; new = old.copy(); new[..., :3] = old[..., :3] * (1 - m3) + whitefur * m3
Image.fromarray((np.clip(new, 0, 1) * 255).astype(np.uint8), "RGBA").save("results/bc_leo_albedo.png")
print("wmean", wmean.round(2), "white fraction (verts)", round(float((white > 0.5).mean()), 2), "saved results/bc_leo_albedo.png")
