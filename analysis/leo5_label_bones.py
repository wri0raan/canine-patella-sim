import glob, json
import numpy as np
L = {p.replace("\\", "/").split("/")[-1][:-5]: json.load(open(p))["pts"] for p in glob.glob("data/real/leo5/label/user/labels/*.json")}
d = lambda a, b: float(np.linalg.norm(np.array(a) - np.array(b)))
for s, nm in (("n", "near"), ("f", "far")):
    for lab, (p, q) in {"tibia (stifle-hock)": (s + "stif", s + "hock"), "metatarsus (hock-paw)": (s + "hock", s + "paw"), "femur (hip-stifle)": ("hip", s + "stif")}.items():
        v = [d(P[p], P[q]) for P in L.values() if P.get(p) and P.get(q)]
        print(f"{nm:5s} {lab:24s} mean {np.mean(v):6.1f} px  sd {np.std(v):5.1f}  CV {100 * np.std(v) / np.mean(v):5.1f} %  n {len(v)}")
v = [d(P["hip"], P["tail"]) for P in L.values() if P.get("hip") and P.get("tail")]; print("hip-tail root:", round(np.mean(v), 1), "px sd", round(np.std(v), 1))
