import sys, numpy as np
from pathlib import Path
src, dst, f_lo = sys.argv[1], sys.argv[2], float(sys.argv[3])
z = dict(np.load(Path("results") / src / "best.npz", allow_pickle=True)); x = z["x"].copy(); x[0] = float(sys.argv[4]) if len(sys.argv) > 4 else max(x[0], f_lo); z["x"] = x
Path("results", dst).mkdir(parents=True, exist_ok=True); np.savez(Path("results") / dst / "best.npz", **z); import shutil
for f in ("trot.yaml", "patella.yaml"):
    if (Path("results") / src / f).exists(): shutil.copy(Path("results") / src / f, Path("results") / dst / f)
print("freq", z["x"][0])
