import argparse, subprocess, sys
import numpy as np, torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--size", type=int, default=1024); ap.add_argument("--side-mult", type=float, default=1.35); ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
import re
_i = subprocess.run([FF, "-i", a.inp], capture_output=True, text=True).stderr; SW, SH = [int(v) for v in re.search(r"Video:.*?, (\d{3,5})x(\d{3,5})", _i).groups()]; DS = max(1, SW // 960); DW, DH = SW // DS, SH // DS; dev = "cuda" if torch.cuda.is_available() else "cpu"
det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(dev).eval()
p = subprocess.Popen([FF, "-v", "error", "-i", a.inp, "-an", "-vf", f"scale={DW}:{DH},format=rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
boxes = []
while True:
    b = p.stdout.read(DW * DH * 3)
    if len(b) < DW * DH * 3: break
    fr = np.frombuffer(b, np.uint8).reshape(DH, DW, 3)
    with torch.no_grad(): o = det([torch.from_numpy(fr.copy()).permute(2, 0, 1).float().div(255).to(dev)])[0]
    k = (o["labels"] == 18) & (o["scores"] > a.thr); bx = o["boxes"][k].cpu().numpy() * DS; sc = o["scores"][k].cpu().numpy()
    boxes.append(bx[np.argmax(sc)] if len(sc) else np.full(4, np.nan))
B = np.array(boxes); n = len(B); found = ~np.isnan(B[:, 0]); print(n, "frames, dog found in", int(found.sum()), flush=True)
idx = np.arange(n)
for j in range(4): B[~found, j] = np.interp(idx[~found], idx[found], B[found, j])
cx = gaussian_filter1d(0.5 * (B[:, 0] + B[:, 2]), 3); cy = gaussian_filter1d(0.5 * (B[:, 1] + B[:, 3]), 3)
side = int(min(SH, a.side_mult * np.percentile(B[:, 2] - B[:, 0], 95))); side += side % 2
x0 = np.clip(np.round(cx - side / 2), 0, SW - side).astype(int); y0 = np.clip(np.round(cy - side / 2), 0, SH - side).astype(int)
i0, i1 = max(0, int(np.nonzero(found)[0][0]) - 10), min(n, int(np.nonzero(found)[0][-1]) + 11)
print("crop side", side, "px (source); frames kept", i0, "-", i1, flush=True)
p1 = subprocess.Popen([FF, "-v", "error", "-i", a.inp, "-an", "-vf", f"format=rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
p2 = subprocess.Popen([FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.size}x{a.size}", "-r", "120", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", a.out + ".mp4"], stdin=subprocess.PIPE)
for i in range(n):
    b = p1.stdout.read(SW * SH * 3)
    if len(b) < SW * SH * 3: print("short read at", i); n = i; break
    if i < i0 or i >= i1 or not found[i]: continue
    fr = np.frombuffer(b, np.uint8).reshape(SH, SW, 3)[y0[i]:y0[i] + side, x0[i]:x0[i] + side]
    p2.stdin.write(np.asarray(Image.fromarray(fr).resize((a.size, a.size), Image.LANCZOS)).tobytes())
p2.stdin.close(); p2.wait()
n1 = min(n, i1); sel = np.array([i for i in range(i0, n1) if found[i]]); np.savez(a.out + "_crop.npz", x0=x0[sel], y0=y0[sel], side=np.full(len(sel), side), found=found[sel], size=a.size, src=np.array([SW, SH]), box=B[sel], frame_idx=sel)
print("wrote", a.out + ".mp4", len(sel), "frames (source frames", i0, "-", n1, ", gaps where the dog was not detected are skipped)")
