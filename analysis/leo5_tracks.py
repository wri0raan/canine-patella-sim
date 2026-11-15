import os
import numpy as np


def to_source(model, clip):
    z = np.load(f"data/real/leo5/work/{model}/{clip}_kp.npz"); c = np.load(f"data/real/leo5/work/c{clip}_crop.npz")
    if len(z["kp"]) != len(c["x0"]): raise SystemExit(f"pose output has {len(z['kp'])} frames but the crop video has {len(c['x0'])} ({model}/{clip}): the pose model dropped frames without a detection, the frame index would shift")
    kp = z["kp"].copy(); n = len(kp); kp = kp[:n]; sc = c["side"][:n] / float(c["size"])
    kp[:, :, 0] = kp[:, :, 0] * sc[:, None] + c["x0"][:n, None]; kp[:, :, 1] = kp[:, :, 1] * sc[:, None] + c["y0"][:n, None]
    box = c["box"][:n]; inside = (box[:, 0] > 30) & (box[:, 2] < float(c["src"][0]) - 30)
    if "found" in c.files: inside &= c["found"][:n].astype(bool)
    return kp, list(z["names"]), inside


def resolve(kp, ix, pre):
    A = np.stack([kp[:, ix[f"{pre}_left_{k}"]] for k in ("thai", "knee", "paw")], 1); B = np.stack([kp[:, ix[f"{pre}_right_{k}"]] for k in ("thai", "knee", "paw")], 1)
    A, B = A.copy(), B.copy(); swaps = 0
    for t in range(1, len(A)):
        w = lambda X: X[:, 2:3]
        cost = lambda P, Q: float(np.sum(np.minimum(w(P), w(Q)) * np.sum((P[:, :2] - Q[:, :2]) ** 2, 1, keepdims=True)))
        keep = cost(A[t], A[t - 1]) + cost(B[t], B[t - 1]); swap = cost(B[t], A[t - 1]) + cost(A[t], B[t - 1])
        if swap < keep: A[t], B[t] = B[t].copy(), A[t].copy(); swaps += 1
    return A, B, swaps


if __name__ == "__main__":
    for model in sorted(os.listdir("data/real/leo5/work")):
        for clip in ("a", "b"):
            if not os.path.exists(f"data/real/leo5/work/{model}/{clip}_kp.npz"): continue
            kp, names, inside = to_source(model, clip); ix = {n: i for i, n in enumerate(names)}
            hA, hB, sh = resolve(kp, ix, "back"); fA, fB, sf = resolve(kp, ix, "front")
            np.savez(f"data/real/leo5/work/{model}/{clip}_tracks.npz", hindA=hA, hindB=hB, foreA=fA, foreB=fB, body=kp, names=np.array(names), inside=inside, swaps_hind=sh, swaps_fore=sf)
            print(model, clip, "frames", len(kp), "hind swaps", sh, "fore swaps", sf)
