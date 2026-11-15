import sys, json, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
clip, frames = sys.argv[1], [int(x) for x in sys.argv[2:]]
FF = imageio_ffmpeg.get_ffmpeg_exe(); os.makedirs("data/real/leo5/label", exist_ok=True)
z = np.load(f"data/real/leo5/work/hrnet_w32/{clip}_kp.npz"); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
cf = "data/real/leo5/label/centres.json"; C = json.load(open(cf)) if os.path.exists(cf) else {}
T, Z = 520, 1.92; tiles = []; FNT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 15)
for f in frames:
    pts = np.array([kp[f, ix[n], :2] for n in ("back_end", "back_left_paw", "back_right_paw", "back_left_knee", "back_right_knee")]); cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
    x0 = int(np.clip(cx - T / 2, 0, 1024 - T)); y0 = int(np.clip(cy - T / 2 + 40, 0, 1024 - T)); C[f"{clip}_{f}"] = [x0, y0]
    r = subprocess.run([FF, "-v", "error", "-i", f"data/real/leo5/work/c{clip}.mp4", "-vf", f"select=eq(n\\,{f})", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    im = Image.fromarray(np.frombuffer(r.stdout, np.uint8).reshape(1024, 1024, 3).copy()).crop((x0, y0, x0 + T, y0 + T)).resize((int(T * Z), int(T * Z)), Image.LANCZOS); d = ImageDraw.Draw(im)
    for gx in range(-(-x0 // 20) * 20, x0 + T + 1, 20):
        X = (gx - x0) * Z; major = gx % 40 == 0; d.line([(X, 0), (X, T * Z)], fill=(255, 120, 0) if major else (255, 200, 130), width=1)
        if major: d.text((X + 2, 2), str(gx), fill=(200, 0, 0), font=FNT); d.text((X + 2, T * Z - 40), str(gx), fill=(200, 0, 0), font=FNT)
    for gy in range(-(-y0 // 20) * 20, y0 + T + 1, 20):
        Y = (gy - y0) * Z; major = gy % 40 == 0; d.line([(0, Y), (T * Z, Y)], fill=(255, 120, 0) if major else (255, 200, 130), width=1)
        if major: d.text((2, Y + 2), str(gy), fill=(200, 0, 0), font=FNT); d.text((T * Z - 40, Y + 2), str(gy), fill=(200, 0, 0), font=FNT)
    d.rectangle([0, T * Z - 22, 260, T * Z], fill=(255, 255, 255)); d.text((6, T * Z - 18), f"clip {clip} frame {f}", fill=(0, 0, 0)); tiles.append(im)
json.dump(C, open(cf, "w"))
for i in range(0, len(tiles), 2):
    pair = tiles[i:i + 2]; sheet = Image.new("RGB", (sum(t.width for t in pair), pair[0].height), (255, 255, 255)); xo = 0
    for t in pair: sheet.paste(t, (xo, 0)); xo += t.width
    sheet.save(f"data/real/leo5/label/tiles_{clip}_{frames[i]}.jpg", quality=90); print("saved", f"tiles_{clip}_{frames[i]}.jpg", frames[i:i + 2])
