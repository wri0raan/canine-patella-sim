import sys
from PIL import Image, ImageDraw, ImageFont
xr, name = sys.argv[1], sys.argv[2]; x0, y0, x1, y1 = [int(v) for v in sys.argv[3:7]]; Z = float(sys.argv[7]) if len(sys.argv) > 7 else 2.0
im = Image.open(f"data/real/leo_xray/{xr}.jpg").convert("RGB").crop((x0, y0, x1, y1)); im = im.resize((int(im.width * Z), int(im.height * Z)), Image.LANCZOS); d = ImageDraw.Draw(im); F = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 15)
for gx in range(-(-x0 // 25) * 25, x1 + 1, 25):
    X = (gx - x0) * Z; major = gx % 50 == 0; d.line([(X, 0), (X, im.height)], fill=(255, 200, 0) if major else (120, 120, 60), width=1)
    if major: d.text((X + 2, 2), str(gx), fill=(255, 255, 0), font=F); d.text((X + 2, im.height - 20), str(gx), fill=(255, 255, 0), font=F)
for gy in range(-(-y0 // 25) * 25, y1 + 1, 25):
    Y = (gy - y0) * Z; major = gy % 50 == 0; d.line([(0, Y), (im.width, Y)], fill=(255, 200, 0) if major else (120, 120, 60), width=1)
    if major: d.text((2, Y + 2), str(gy), fill=(255, 255, 0), font=F); d.text((im.width - 44, Y + 2), str(gy), fill=(255, 255, 0), font=F)
im.save(f"results/figures/xgrid_{xr}_{name}.png"); print(im.size)
