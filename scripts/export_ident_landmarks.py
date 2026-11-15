import subprocess, sys
from pathlib import Path
PY = sys.executable
for r in sys.argv[1:]:
    out = Path("results/ident") / f"{r}.npz"
    if out.exists() or not (Path("results") / r / "best.npz").exists(): continue
    subprocess.run([PY, "scripts/export_motion_for_blender.py", r, str(out), "--seconds", "14", "--fps", "60", "--t0", "2"], check=False)
    print(r, out.exists(), flush=True)
