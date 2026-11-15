import json, os, subprocess, base64, io
import numpy as np
from PIL import Image
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe(); T, OUT = 520, 800; Z = OUT / T
os.makedirs("data/real/leo5/label/pack", exist_ok=True)
SEL = {"a": list(range(200, 486, 15)), "b": list(range(140, 405, 24))}
order = []
for i in range(max(len(v) for v in SEL.values())):
    for c in ("a", "b"):
        if i < len(SEL[c]): order.append((c, SEL[c][i]))
out = []
for clip, f in order:
    z = np.load(f"data/real/leo5/work/hrnet_w32/{clip}_kp.npz"); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
    pts = np.array([kp[f, ix[n], :2] for n in ("back_end", "back_left_paw", "back_right_paw", "back_left_knee", "back_right_knee")]); cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
    x0 = int(np.clip(cx - T / 2, 0, 1024 - T)); y0 = int(np.clip(cy - T / 2 + 40, 0, 1024 - T))
    r = subprocess.run([FF, "-v", "error", "-i", f"data/real/leo5/work/c{clip}.mp4", "-vf", f"select=eq(n\\,{f})", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    im = Image.fromarray(np.frombuffer(r.stdout, np.uint8).reshape(1024, 1024, 3).copy()).crop((x0, y0, x0 + T, y0 + T)).resize((OUT, OUT), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=80); out.append(dict(id=f"{clip}_{f}", clip=clip, frame=f, x0=x0, y0=y0, z=Z, w=OUT, jpg_b64=base64.b64encode(buf.getvalue()).decode()))
json.dump(out, open("data/real/leo5/label/pack/frames.json", "w")); print(len(out), "frames,", round(sum(len(o["jpg_b64"]) for o in out) / 1e6, 1), "MB base64")
