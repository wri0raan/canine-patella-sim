from __future__ import annotations

from pathlib import Path

import numpy as np

NB = 32
_REF: dict | None = None
ROOT = Path(__file__).resolve().parents[1]


def _znorm(x: np.ndarray, floor_deg: float) -> np.ndarray:
    return (x - x.mean()) / max(float(x.std()), floor_deg)


def load_reference(clip: str = "Trot_F_IP", floor_deg: float = 3.0) -> dict:
    global _REF
    if _REF is not None:
        return _REF
    d = np.load(ROOT / "data" / "ref" / "bordercollie_gaits.npz")
    ref = {}
    for kind, leg in (("hind", "LH"), ("fore", "LF")):
        A = d[f"{clip}/{leg}"][:-1, :3]
        n = len(A)
        ph = np.arange(n) / n
        grid = np.arange(NB) / NB
        series = [np.interp(grid, np.r_[ph, 1.0], np.r_[A[:, j], A[0, j]]) for j in range(3)]
        ref[kind] = np.stack([_znorm(s, floor_deg) for s in series])
    _REF = ref
    return ref


def shape_error(q_deg: np.ndarray, t: np.ndarray, freq: float, offsets: dict[str, float], floor_deg: float = 3.0) -> float:
    ref = load_reference(floor_deg=floor_deg)
    legs = {"LF": (0, "fore"), "RH": (3, "hind"), "RF": (6, "fore"), "LH": (9, "hind")}
    errs = []
    for leg, (c0, kind) in legs.items():
        phase = (freq * t + offsets[leg]) % 1.0
        idx = np.minimum((phase * NB).astype(int), NB - 1)
        W = np.zeros((3, NB)); cnt = np.bincount(idx, minlength=NB)
        if (cnt == 0).any():
            return 1.5
        for j in range(3):
            W[j] = np.bincount(idx, weights=q_deg[:, c0 + j], minlength=NB) / cnt
        Z = np.stack([_znorm(W[j], floor_deg) for j in range(3)])
        best = min(np.sqrt(np.mean((Z - np.roll(ref[kind], s, axis=1)) ** 2)) for s in range(NB))
        errs.append(best)
    return float(np.mean(errs))
