import sys
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
import mujoco, numpy as np
from sim.config import load_yaml
from sim.rollout import Sim

COL = {"hind": "tab:blue", "hindfoot": "tab:cyan", "fore": "tab:red", "digit": "tab:orange", "neck": "tab:green", "tail": "tab:purple", "trunk": "tab:gray"}


def region(name):
    from sim.dog_scale import region as rg
    return rg(name)


def settle(model_cfg, T=2.0):
    cfg = load_yaml("configs/gait/trot.yaml"); cfg["model"] = model_cfg
    sim = Sim(cfg); m, d, sv = sim.m, sim.d, sim.sv
    imp = load_yaml("configs/impairment/patella.yaml"); sim.limiter.set_grade(imp, 0, "functional", ("LH", "RH"))
    mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); q = d.qpos[sv["qadr"]].copy()
    for i in range(int(T / m.opt.timestep)):
        d.ctrl[sv["act_ids"]] = q
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
    return m, d


def draw(ax, m, d, dx=0.0):
    x0 = d.xpos[1][0] if m.nbody > 1 else 0.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        t = m.geom_type[g]; c = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3); sz = m.geom_size[g]
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "torso"
        col = COL.get(region(bname), "gray"); P = lambda p: np.array([p[0] - x0 + dx, p[2]])
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ax.add_patch(Circle(P(c), sz[0], fc=col, ec="k", lw=0.3, alpha=0.55))
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ax_ = R[:, 2]; a = P(c - ax_ * sz[1]); b = P(c + ax_ * sz[1]); v = b - a; L = np.linalg.norm(v)
            ax.add_patch(Circle(a, sz[0], fc=col, ec="none", alpha=0.55)); ax.add_patch(Circle(b, sz[0], fc=col, ec="none", alpha=0.55))
            if L > 1e-6:
                n = np.array([-v[1], v[0]]) / L * sz[0]; ax.add_patch(Polygon([a + n, b + n, b - n, a - n], fc=col, ec="none", alpha=0.55))
        elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            S = (R * sz) @ (R * sz).T; A = S[np.ix_([0, 2], [0, 2])]; w, V = np.linalg.eigh(A)
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1])); ax.add_patch(Ellipse(P(c), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, fc=col, ec="k", lw=0.3, alpha=0.55))
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            pts = np.array([[sx, sy, sz_] for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)]) * sz
            q2 = np.array([P(c + R @ p) for p in pts]); from scipy.spatial import ConvexHull
            h = ConvexHull(q2); ax.add_patch(Polygon(q2[h.vertices], fc=col, ec="k", lw=0.3, alpha=0.45))


if __name__ == "__main__":
    leo = load_yaml("configs/model/leo.yaml")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for label, mc, dx in (("original Dog 10.2 kg", None, 0.0), ("Leo-proportioned 7 kg", leo, 1.25)):
        m, d = settle(mc); draw(ax, m, d, dx)
        ax.text(dx - 0.1, 0.62, label, fontsize=9)
    ax.axhline(0, color="k", lw=1); ax.set_aspect("equal"); ax.set_xlim(-0.75, 1.9); ax.set_ylim(-0.02, 0.7)
    ax.set_xlabel("m"); ax.set_ylabel("height (m)")
    for y, lab in ((0.26, "hip 26"), (0.16, "stifle 16"), (0.085, "hock 8.5"), (0.35, "withers 35")): ax.axhline(y, color="r", lw=0.5, ls=":"); ax.text(1.75, y + 0.005, lab + " cm (Leo)", fontsize=7, color="r")
    fig.tight_layout(); fig.savefig(sys.argv[1], dpi=150); print("saved", sys.argv[1])
