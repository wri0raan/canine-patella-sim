import sys, yaml
import numpy as np
for r in sys.argv[1:]:
    c = yaml.safe_load(open(f"results/{r}/trot.yaml", encoding="utf-8")); print(r, "cost:", {k: v for k, v in c["cost"].items() if k.startswith("w_") or "target" in k or "win" in k or "calib" in k or "style" in k})
    print("  gait:", {k: c["gait"][k] for k in ("target_speed", "eval_duration") if k in c["gait"]})
    z = np.load(f"results/{r}/best.npz", allow_pickle=True); print("  summary:", str(z["summary"])[:500])
