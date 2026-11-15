import os
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from analysis.leo5_tracks import to_source

CAP_F, AGREE_F = 0.12, 0.05


def viterbi_swap(A, B, cap):
    T = len(A); w = lambda X: X[:, 2:3]
    def tr(P, Q): return float(np.sum(np.minimum(w(P), w(Q)) * np.minimum(np.sum((P[:, :2] - Q[:, :2]) ** 2, 1, keepdims=True), cap ** 2)))
    cost = np.zeros((T, 2)); back = np.zeros((T, 2), int)
    for t in range(1, T):
        for s in (0, 1):
            Pt, Qt = (A[t], B[t]) if s == 0 else (B[t], A[t])
            best = None
            for sp in (0, 1):
                Pp, Qp = (A[t - 1], B[t - 1]) if sp == 0 else (B[t - 1], A[t - 1])
                c = cost[t - 1, sp] + tr(Pt, Pp) + tr(Qt, Qp)
                if best is None or c < best[0]: best = (c, sp)
            cost[t, s], back[t, s] = best
    s = int(np.argmin(cost[-1])); st = np.zeros(T, int)
    for t in range(T - 1, -1, -1): st[t] = s; s = back[t, s]
    A2, B2 = A.copy(), B.copy(); sw = st == 1; A2[sw], B2[sw] = B[sw], A[sw]; return A2, B2, int(np.sum(np.diff(st) != 0))


def clean(X, fps=120.0, hz=12.0):
    X = X.copy(); T = len(X); ok = X[:, 2] > 0.2
    p = X[:, :2]; sm = np.stack([medfilt(p[:, 0], 9), medfilt(p[:, 1], 9)], 1); dev = np.linalg.norm(p - sm, axis=1)
    ok &= dev < max(4 * np.median(dev) + 1e-6, 12.0)
    idx = np.arange(T)
    for d in (0, 1):
        if ok.sum() > 10: X[:, d] = np.interp(idx, idx[ok], X[ok, d])
    b, a = butter(3, hz / (fps / 2)); X[:, 0] = filtfilt(b, a, X[:, 0]); X[:, 1] = filtfilt(b, a, X[:, 1]); X[:, 2] = np.where(ok, X[:, 2], 0.0); return X, ~ok


if __name__ == "__main__":
    os.makedirs("data/real/leo5/work/fused", exist_ok=True)
    import sys
    for clip in (sys.argv[1:] or ["a", "b"]):
        res = {}
        for model in ("hrnet_w32", "resnet_50"):
            kp, names, inside = to_source(model, clip); ix = {n: i for i, n in enumerate(names)}
            dog = np.median(np.linalg.norm(kp[:, ix["back_base"], :2] - kp[:, ix["back_end"], :2], axis=1)[inside]); out = {}
            for pre in ("back", "front"):
                A = np.stack([kp[:, ix[f"{pre}_left_{k}"]] for k in ("thai", "knee", "paw")], 1); B = np.stack([kp[:, ix[f"{pre}_right_{k}"]] for k in ("thai", "knee", "paw")], 1)
                A, B, ns = viterbi_swap(A, B, CAP_F * dog); out[pre] = (A, B); print(model, clip, pre, "identity switches", ns)
            res[model] = (out, dog, kp, names, inside)
        (o1, dog, kp, names, inside), (o2, _, _, _, _) = res["hrnet_w32"], res["resnet_50"]
        fused, flags = {}, {}
        for pre, key in (("back", "hind"), ("front", "fore")):
            A1, B1 = o1[pre]; A2, B2 = o2[pre]; T = len(A1); A2, B2 = A2.copy(), B2.copy()
            d = lambda P, Q: np.sum(np.minimum(P[:, :, 2], Q[:, :, 2]) * np.linalg.norm(P[:, :, :2] - Q[:, :, :2], axis=2), 1)
            sw = (d(A1, B2) + d(B1, A2)) < (d(A1, A2) + d(B1, B2)); A2[sw], B2[sw] = B2[sw], A2[sw]
            for nm, P, Q in ((f"{key}A", A1, A2), (f"{key}B", B1, B2)):
                F = np.zeros_like(P); fl = np.zeros(P.shape[:2], bool)
                for j in range(3):
                    p, q = P[:, j], Q[:, j]; agree = np.linalg.norm(p[:, :2] - q[:, :2], axis=1) < AGREE_F * dog
                    wsum = p[:, 2:3] + q[:, 2:3] + 1e-9; mean = (p * p[:, 2:3] + q * q[:, 2:3]) / wsum; best = np.where((p[:, 2] >= q[:, 2])[:, None], p, q)
                    F[:, j] = np.where(agree[:, None], mean, best); F[:, j, 2] = np.where(agree, np.minimum(p[:, 2], q[:, 2]), 0.5 * np.minimum(p[:, 2], q[:, 2])); fl[:, j] = ~agree
                for j in range(3): F[:, j], bad = clean(F[:, j]); fl[:, j] |= bad
                fused[nm], flags[nm] = F, fl
        np.savez(f"data/real/leo5/work/fused/{clip}_tracks.npz", **fused, flags_hindA=flags["hindA"], flags_hindB=flags["hindB"], body=kp, names=np.array(names), inside=inside, back_px=dog)
        print(clip, "fused; flagged fraction hind A/B (stifle,hock,paw):", flags["hindA"][inside].mean(0).round(2), flags["hindB"][inside].mean(0).round(2))
