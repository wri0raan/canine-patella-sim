import re, sys
import numpy as np, yaml

best = None
for tag in sys.argv[1:]:
    try:
        z = np.load(f"results/{tag}/best.npz", allow_pickle=True); s = str(z["summary"]); tgt = yaml.safe_load(open(f"results/{tag}/trot.yaml"))["gait"]["target_speed"]
        fell = re.search(r"'fell': (True|False)", s).group(1) == "True"; v = float(re.search(r"'v_mean': ([0-9.eE+-]+)", s).group(1))
    except Exception:
        continue
    print(tag, "target", tgt, "v", round(v, 2), "fell", fell, file=sys.stderr)
    if not fell and v >= 0.85 * tgt and (best is None or tgt > best[0]): best = (tgt, tag)
print(best[1] if best else "NONE")
