import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
from scipy.signal import savgol_filter, find_peaks
from analysis.video_style_features import video_landmarks, angle2

SRC = {"또리 수술 전": ("data/real/dogs/cropwork/d1_ttori_pre_kp.npz", 30, 55, 139, 1), "또리 수술 후": ("data/real/dogs/cropwork/d2_ttori_post_kp.npz", 30, 0, 72, 2),
       "또리 수술 후 (60 fps 원본, 참고)": ("data/real/dogs/cropwork/d2_ttori_post_kp.npz", 60, 0, 144, 1),
       "leo A (참고)": ("data/real/leo2/work/leo_kp.npz", 30, int(43.7 * 30), int(46.8 * 30) - 1, 4), "leo B (참고)": ("data/real/leo2/work/leo_kp.npz", 30, int(33.8 * 30), int(36.8 * 30) - 1, 4)}
rows, curves = [], {}
for nm, (path, fps, a, b, step) in SRC.items():
    z = np.load(path, allow_pickle=True); kp, names = z["kp"][::step], list(z["names"]); L = video_landmarks(kp, names, a, b)
    ctr = 0.5 * (L["back_a"] + L["back_b"]); body = float(np.median(np.linalg.norm(L["back_a"] - L["back_b"], axis=1)))
    u = L["hpaw"][:, 0] - ctr[:, 0]; w = max(5, int(0.1 * fps) | 1); uf = savgol_filter(u, w, 2)
    pk, _ = find_peaks(uf, distance=int(0.14 * fps), prominence=0.12 * np.ptp(uf))
    win = max(5, int(0.05 * fps) | 1)
    hock = savgol_filter(angle2(L["stifle"], L["hock"], L["hpaw"]), win, 2); carp = savgol_filter(angle2(L["elbow"], L["carpus"], L["fpaw"]), win, 2)
    for i in range(len(pk) - 1):
        s, e = pk[i], pk[i + 1]; T = (e - s) / fps; f = 1 / T
        if not (0.25 < T < 0.7): continue
        h, c = hock[s:e + 1], carp[s:e + 1]; ph = np.linspace(0, 1, e - s + 1); g = np.linspace(0, 1, 50)
        rows.append(dict(clip=nm, cycle=i, start_s=(a + s) / fps, stride_hz=f, hock_rom=np.ptp(h), hock_min=h.min(), carpus_rom=np.ptp(c), carpus_min=c.min(),
                         paw_excursion=float(np.ptp(uf[s:e + 1]) / body)))
        curves.setdefault(nm, []).append((f, np.interp(g, ph, h), np.interp(g, ph, c)))
R = pd.DataFrame(rows); R.to_csv("results/ttori_cycles.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 200); print(R.round(2).to_string())
post = R[R["clip"] == "또리 수술 후"]; pre = R[R["clip"] == "또리 수술 전"]
print("\npre cycles", len(pre), "stride %.2f Hz" % pre.stride_hz.mean(), "| post cycles with stride <= 2.5 Hz:", int((post.stride_hz <= 2.5).sum()))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2)); ph = np.linspace(0, 100, 50)
sel = {"또리 수술 전": ("#ff7f0e", lambda f: True), "또리 수술 후 (30 fps로 맞춤, 2.5 Hz 이하)": ("#9467bd", lambda f: f <= 2.5), "또리 수술 후 (30 fps로 맞춤, 3 Hz 이상)": ("#cab2d6", lambda f: f >= 3.0), "leo (정상, 참고)": ("#1f77b4", lambda f: True)}
for lab, (col, ok) in sel.items():
    key = "또리 수술 전" if lab == "또리 수술 전" else "또리 수술 후" if lab.startswith("또리 수술 후") else None
    cs = curves.get(key, []) if key else curves.get("leo A (참고)", []) + curves.get("leo B (참고)", [])
    cs = [c for c in cs if ok(c[0])]
    if not cs: continue
    for j, (yl, a_) in enumerate((("hock", ax[0]), ("carpus", ax[1]))):
        M = np.array([c[1 + j] for c in cs]); a_.plot(ph, M.mean(0), c=col, lw=2, label=f"{lab} (n={len(cs)})"); a_.fill_between(ph, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=col, alpha=.15)
for a_, t in ((ax[0], "비절 각도 (2D, 보행 주기 0-100%)"), (ax[1], "손목 각도 (2D)")): a_.set_title(t, fontsize=10); a_.set_xlabel("보행 주기 (%)"); a_.set_ylabel("도"); a_.grid(alpha=.3)
ax[0].legend(fontsize=7)
for lab, d, col in (("수술 전", pre, "#ff7f0e"), ("수술 후", post, "#9467bd")): ax[2].scatter(d.stride_hz, d.hock_rom, c=col, label=lab + " 비절"); ax[2].scatter(d.stride_hz, d.carpus_rom, c=col, marker="s", label=lab + " 손목")
ax[2].set_xlabel("걸음 주기 (Hz)"); ax[2].set_ylabel("주기별 각도 범위 (도)"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3); ax[2].set_title("주기마다의 각도 범위", fontsize=10)
fig.tight_layout(); fig.savefig("results/figures/ttori_compare.png", dpi=150); print("saved")
