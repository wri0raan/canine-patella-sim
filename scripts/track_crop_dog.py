import argparse
import numpy as np, imageio, torch, torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--seed-frame", type=int, default=0); ap.add_argument("--seed-xy", default=None); ap.add_argument("--pad", type=float, default=1.9); ap.add_argument("--size", type=int, default=768); ap.add_argument("--thr", type=float, default=0.4)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
W = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT; det = fasterrcnn_resnet50_fpn_v2(weights=W).to(dev).eval()
r = imageio.get_reader(a.inp); meta = r.get_meta_data(); fps = meta["fps"]; frames = [np.asarray(f) for f in r]; n = len(frames); H, Wd = frames[0].shape[:2]


def detect(fr):
    with torch.no_grad():
        o = det([torch.from_numpy(fr).permute(2, 0, 1).float().div(255).to(dev)])[0]
    k = (o["labels"] == 18) & (o["scores"] > a.thr)
    return o["boxes"][k].cpu().numpy(), o["scores"][k].cpu().numpy()


def hist(fr, b):
    x0, y0, x1, y1 = [int(v) for v in b]; c = fr[max(y0, 0):y1, max(x0, 0):x1]
    h = np.concatenate([np.histogram(c[..., i], 8, (0, 255))[0] for i in range(3)]).astype(float); return h / (h.sum() + 1e-9)


def iou(p, q):
    x0, y0 = max(p[0], q[0]), max(p[1], q[1]); x1, y1 = min(p[2], q[2]), min(p[3], q[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


boxes = [None] * n; ref_hist = None; prev = None
order = list(range(a.seed_frame, n)) + list(range(a.seed_frame - 1, -1, -1))
for k in order:
    if k == a.seed_frame - 1: prev = boxes[a.seed_frame]
    B, S = detect(frames[k])
    if len(B) == 0: continue
    if prev is None:
        if a.seed_xy:
            sx, sy = [float(v) for v in a.seed_xy.split(",")]; ins = [i for i, b in enumerate(B) if b[0] <= sx <= b[2] and b[1] <= sy <= b[3]]; i = ins[0] if ins else int(np.argmax(S))
        else: i = int(np.argmax((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])))
        boxes[k] = B[i]; prev = B[i]; ref_hist = hist(frames[k], B[i]); continue
    sc = []
    for b in B:
        ctr = np.hypot((b[0] + b[2] - prev[0] - prev[2]) / 2, (b[1] + b[3] - prev[1] - prev[3]) / 2) / max(prev[2] - prev[0], prev[3] - prev[1])
        sc.append(iou(b, prev) - 0.5 * ctr + 0.5 * float(np.minimum(hist(frames[k], b), ref_hist).sum()))
    i = int(np.argmax(sc))
    if sc[i] > 0.35: boxes[k] = B[i]; prev = B[i]
found = np.array([b is not None for b in boxes]); print(a.inp, "frames", n, "dog found in", int(found.sum()))
ctr = np.full((n, 2), np.nan); side = np.full(n, np.nan)
for k, b in enumerate(boxes):
    if b is not None: ctr[k] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]; side[k] = max(b[2] - b[0], b[3] - b[1]) * a.pad
ok = np.where(found)[0]
if len(ok) < 5: raise SystemExit("dog not tracked")
for arr in (ctr[:, 0], ctr[:, 1], side): arr[:] = np.interp(np.arange(n), ok, arr[ok])
from scipy.signal import savgol_filter
w = min(n if n % 2 else n - 1, 15)
cx, cy = savgol_filter(ctr[:, 0], w, 2), savgol_filter(ctr[:, 1], w, 2)
sd = np.full(n, float(np.median(side)) * 1.15)
x0 = np.clip(cx - sd / 2, 0, None); y0 = np.clip(cy - sd / 2, 0, None)
out = []
for k in range(n):
    s = int(round(sd[k])); xi, yi = int(round(x0[k])), int(round(y0[k])); c = np.zeros((s, s, 3), np.uint8); sub = frames[k][yi:yi + s, xi:xi + s]; c[:sub.shape[0], :sub.shape[1]] = sub
    out.append(np.asarray(Image.fromarray(c).resize((a.size, a.size))))
imageio.mimsave(a.out + ".mp4", out, fps=fps, macro_block_size=1)
np.savez(a.out + "_crop.npz", x0=x0, y0=y0, side=sd, found=found, fps=fps)
print("wrote", a.out + ".mp4", "crop side(px) %.0f" % sd[0])
