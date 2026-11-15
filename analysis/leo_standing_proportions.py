import sys
import numpy as np
z = np.load(sys.argv[1]); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
fpp = int(sys.argv[2]) if len(sys.argv) > 2 else 2
labels = ["photo_6", "photo_8", "photo_9", "photo_11", "photo_12", "photo_13", "photo_4", "photo_5"]
rows = []
for k, lab in enumerate(labels):
    f = kp[k * fpp]; g = lambda n: f[ix[n], :2]; c = lambda n: f[ix[n], 2]
    for side_h, side_f in (("back_left", "front_left"), ("back_right", "front_right")):
        need = [side_h + "_paw", side_h + "_knee", side_h + "_thai", side_f + "_thai", side_f + "_paw"]
        if min(c(n) for n in need) < 0.5: continue
        ground = max(g(side_h + "_paw")[1], g(side_f + "_paw")[1])
        h_st = ground - g(side_h + "_thai")[1]; h_hk = ground - g(side_h + "_knee")[1]
        if h_st <= 5 or h_hk <= 3: continue
        s1, s2 = 16.0 / h_st, 8.5 / h_hk; s = 0.5 * (s1 + s2)
        el = g(side_f + "_thai"); st = g(side_h + "_thai"); hk = g(side_h + "_knee")
        face = np.sign(g("nose")[0] - g("tail_base")[0])
        dx = (el[0] - st[0]) * face * s; elbow_h = (ground - el[1]) * s
        rows.append((lab, side_h, s1 / s2, dx, elbow_h, (el[0] - hk[0]) * face * s))
        print("%-9s %-10s scale(stifle)/scale(hock) %.2f | elbow height %.1f cm | elbow->stifle x %.1f cm | elbow->hock x %.1f cm" % (lab, side_h, s1 / s2, elbow_h, dx, (el[0] - hk[0]) * face * s))
if rows:
    print("median elbow->stifle %.1f cm (n=%d), elbow height %.1f cm" % (np.median([r[3] for r in rows]), len(rows), np.median([r[4] for r in rows])))
