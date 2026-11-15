import argparse, glob, subprocess, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("prefix"); ap.add_argument("items", nargs="+"); ap.add_argument("--fps", type=int, default=15); ap.add_argument("--snap", default=None); a = ap.parse_args()
FT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 28); FS = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 20)
panels = []
for it in a.items:
    lab, run = it.rsplit("=", 1); fr = sorted(glob.glob(f"{a.prefix}{run}/*.png")); panels.append((lab, fr))
n = min(len(f) for _, f in panels); W, H = Image.open(panels[0][1][0]).size; FF = imageio_ffmpeg.get_ffmpeg_exe()
enc = subprocess.Popen([FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W * len(panels)}x{H}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
snap_idx = [int(n * q) for q in (0.15, 0.45, 0.75)]; snaps = []
for i in range(n):
    row = Image.new("RGB", (W * len(panels), H))
    for k, (lab, fr) in enumerate(panels):
        im = Image.open(fr[i]).convert("RGB"); d = ImageDraw.Draw(im); d.rectangle([0, 0, W, 44], fill=(0, 0, 0)); d.text((12, 6), lab, fill=(255, 255, 255), font=FT); d.text((W - 150, H - 30), "0.5배 속도", fill=(240, 220, 80), font=FS); row.paste(im, (k * W, 0))
    if i in snap_idx: snaps.append(row.copy())
    enc.stdin.write(row.tobytes())
enc.stdin.close(); enc.wait(); print("wrote", a.out, n, "frames")
if a.snap and snaps:
    S = Image.new("RGB", (snaps[0].width, snaps[0].height * len(snaps))); [S.paste(s, (0, k * s.height)) for k, s in enumerate(snaps)]; S.thumbnail((2000, 2400)); S.save(a.snap); print("snap", a.snap)
