import glob, os
import numpy as np, pandas as pd
for h in glob.glob("data/real/leo5/work/*/c*_superanimal_quadruped_*.h5"):
    d = pd.read_hdf(h); names = list(dict.fromkeys([c[-2] for c in d.columns]))
    arr = np.stack([np.stack([d.xs((b, k), axis=1, level=(-2, -1)).values[:, 0] for k in ("x", "y", "likelihood")], axis=1) for b in names], axis=1)
    out = os.path.join(os.path.dirname(h), os.path.basename(h).split("_superanimal")[0][1:] + "_kp.npz"); np.savez(out, kp=arr, names=np.array(names)); print(out, arr.shape)
