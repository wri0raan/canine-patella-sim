import sys, subprocess
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg
model, clip, frames = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
z = np.load(f"data/real/leo5/work/{model}/{clip}_kp.npz"); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
FF = imageio_ffmpeg.get_ffmpeg_exe(); S = 1024; tiles = []
for f in frames:
    r = subprocess.run([FF, "-v", "error", "-i", f"data/real/leo5/work/c{clip}.mp4", "-vf", f"select=eq(n\\,{f})", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    im = Image.fromarray(np.frombuffer(r.stdout, np.uint8).reshape(S, S, 3).copy()); d = ImageDraw.Draw(im)
    for side, col in (("left", (255, 0, 0)), ("right", (0, 90, 255))):
        for pre in ("back", "front"):
            pts = [kp[f, ix[f"{pre}_{side}_{k}"], :2] for k in ("thai", "knee", "paw")]; d.line([tuple(p) for p in pts], fill=col, width=3 if pre == "back" else 2)
            for p in pts: d.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], outline=col, width=2)
    d.text((10, 10), f"{model} {clip} f{f}  red=left blue=right  (thick = hind)", fill=(0, 0, 0)); tiles.append(im.crop((0, 250, S, 900)).resize((680, 430)))
cols = 3; rows = (len(tiles) + cols - 1) // cols; sheet = Image.new("RGB", (cols * 680, rows * 430), (255, 255, 255))
for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * 680, (i // cols) * 430))
sheet.save(f"results/figures/leo5_overlay_{model}_{clip}.jpg", quality=88); print("saved")
