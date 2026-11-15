from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "data" / "models" / "dog_xml" / "dog.xml"

DEFAULT = dict(hind=0.55, hindfoot=0.55, fore=0.59, digit=0.70, neck=0.72, head=0.85, skull_x=1.0, skull_y=1.0, skull_z=1.0, scapula=0.77, tail=0.425, trunk_x=0.83, trunk_yz=0.60, chest_yz=0.50, chest_len=0.82, chest_dx=-0.015, neck_r=0.50, mass=7.0)
TARGETS_CM = dict(hip=26.0, stifle=16.0, hock=8.5, shoulder=25.0, withers=35.0, back=45.0)


def _ensure_xml():
    if XML.exists():
        return
    from dm_control.suite import dog
    xml, assets = dog.get_model_and_assets()
    for k, v in assets.items():
        kk = k.replace("./", "")
        rel = kk if kk.startswith("common/") else "dog_assets/" + kk.split("/")[-1]
        p = XML.parent / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(v if isinstance(v, bytes) else v.encode())
    XML.write_bytes(xml)


def region(name: str) -> str:
    if name.startswith("foot_"):
        return "hindfoot"
    if name.startswith("upper_leg_"):
        return "hindup"
    if name.startswith("lower_leg_"):
        return "hind"
    if name.startswith("scapula_"):
        return "scapula"
    if name.startswith(("upper_arm_", "lower_arm_", "hand_")):
        return "fore"
    if name.startswith(("toe_", "finger_")):
        return "digit"
    if name.startswith("C_"):
        return "neck"
    if name.startswith(("skull", "jaw")):
        return "head"
    if name.startswith("Ca_"):
        return "tail"
    return "trunk"


def vec_scale(reg: str, s: dict) -> np.ndarray:
    if reg == "head":
        return s["head"] * np.array([s["skull_x"], s["skull_y"], s["skull_z"]])
    if reg == "trunk":
        return np.array([s["trunk_x"], s["trunk_yz"], s["trunk_yz"]])
    if reg == "hindup":
        return np.full(3, s.get("hind_upper", s["hind"]))
    return np.full(3, s[reg])


def _read_stl(path):
    import struct
    b = open(path, "rb").read(); n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float), a


def _write_stl(path, v, a):
    import struct
    tri = v.reshape(-1, 3, 3).astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nl = np.linalg.norm(nrm, axis=1, keepdims=True); nrm = np.where(nl > 0, nrm / np.maximum(nl, 1e-12), 0).astype(np.float32)
    rec = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])); rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as f:
        f.write(b"toy skull".ljust(80, b"\0")); f.write(struct.pack("<I", len(tri))); f.write(rec.tobytes())


def make_toy_skull_assets(stop_x=0.030, muzzle_k=0.42, dome=0.12):
    src = XML.parent / "dog_assets"; dst = XML.parent / "dog_assets_toy"; dst.mkdir(exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(XML))
    geom = {}
    for b in spec.bodies:
        if b.name in ("skull", "jaw"):
            for g in b.geoms:
                if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                    geom[g.meshname] = (b.name, np.array(g.pos), np.array(b.pos) if b.name == "jaw" else np.zeros(3))
    def warp(p):
        p = p.copy(); x = p[:, 0].copy()
        blend = np.clip(1.0 - (x - (stop_x - 0.01)) / 0.04, 0.0, 1.0)
        p[:, 1] *= 1.0 + dome * blend; p[:, 2] *= 1.0 + dome * blend
        p[:, 0] = np.where(x > stop_x, stop_x + (x - stop_x) * muzzle_k, x)
        return p
    for me in spec.meshes:
        if me.name not in geom: continue
        body, gpos, bpos = geom[me.name]
        v, a = _read_stl(src / me.file.split("/")[-1]); off = gpos + bpos
        if me.name.startswith("eye"):
            c = (v + off).mean(0); sh = warp(c[None, :])[0] - c; vn = v + sh
        else:
            vn = warp(v + off) - off
        _write_stl(dst / me.file.split("/")[-1], vn, a)
    return dst


def _body_scale_for_skin(name: str, s: dict) -> np.ndarray:
    if name == "torso":
        return np.array([s["trunk_x"], s["chest_yz"] * s.get("chest_y_skin", 1.0), s["chest_yz"]])
    reg = region(name)
    if reg == "neck":
        return np.full(3, 0.5 * (s["neck"] + s["neck_r"]))
    if reg == "head":
        return np.full(3, float(np.mean(vec_scale("head", s))))
    return vec_scale(reg, s)


def _transfer_skin(spec, m_orig, m_new, s: dict):
    d0 = mujoco.MjData(m_orig); mujoco.mj_forward(m_orig, d0); d1 = mujoco.MjData(m_new); mujoco.mj_forward(m_new, d1)
    V = m_orig.skin_vert.reshape(-1, 3).copy(); newV = np.zeros_like(V); wsum = np.zeros(len(V))
    nb = m_orig.nskinbone; names = []; vid, vw = [], []
    for b in range(nb):
        body = int(m_orig.skin_bonebodyid[b]); name = mujoco.mj_id2name(m_orig, mujoco.mjtObj.mjOBJ_BODY, body); names.append(name)
        a0, n0 = int(m_orig.skin_bonevertadr[b]), int(m_orig.skin_bonevertnum[b])
        ids = m_orig.skin_bonevertid[a0:a0 + n0].astype(int); w = m_orig.skin_bonevertweight[a0:a0 + n0].astype(float)
        p_old = m_orig.skin_bonebindpos[b]; Sb = _body_scale_for_skin(name, s)
        p_new = d1.xpos[mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_BODY, name)]
        newV[ids] += w[:, None] * (p_new + Sb * (V[ids] - p_old)); wsum[ids] += w
        vid.append(ids.tolist()); vw.append(w.tolist())
    newV = np.where(wsum[:, None] > 0, newV / np.maximum(wsum[:, None], 1e-9), V)
    old = spec.skins[0] if len(spec.skins) else None
    sk = old if old is not None else spec.add_skin()
    sk.file = ""; sk.vert = newV.ravel().tolist()
    sk.texcoord = m_orig.skin_texcoord.ravel().tolist(); sk.face = m_orig.skin_face.ravel().astype(int).tolist()
    sk.bodyname = names
    sk.bindpos = np.concatenate([d1.xpos[mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_BODY, n)] for n in names]).tolist()
    sk.bindquat = m_orig.skin_bonebindquat.ravel().tolist()
    sk.vertid = vid; sk.vertweight = vw
    return sk


def build_leo_model(scale: dict | None = None):
    s = dict(DEFAULT); s.update(scale or {})
    _ensure_xml()
    spec = mujoco.MjSpec.from_file(str(XML))
    if s.get("toy_skull", 1):
        make_toy_skull_assets(s.get("toy_stop_x", 0.030), s.get("toy_muzzle_k", 0.42), s.get("toy_dome", 0.12))
        for me in spec.meshes:
            if me.name in ("MergedSkull", "Jaw", "eye_L", "eye_R"):
                me.file = "../dog_assets_toy/" + me.file.split("/")[-1]
    body_reg = {b.name: region(b.name) for b in spec.bodies if b.name and b.name != "world"}
    skin_info = None
    if s.get("skin", 1):
        spec_o = mujoco.MjSpec.from_file(str(XML)); skin_info = (spec_o, spec_o.compile())
    for sk in list(spec.skins):
        spec.delete(sk)
    for b in spec.bodies:
        if not b.name or b.name == "world":
            continue
        par = b.parent.name if b.parent is not None and b.parent.name else "torso"
        pv = vec_scale(body_reg.get(par, "trunk"), s)
        b.pos = np.array(b.pos) * pv
        v = vec_scale(body_reg[b.name], s); iso = float(np.mean(v))
        gv = v.copy()
        if b.name == "torso":
            gv = np.array([s["trunk_x"] * s["chest_len"], s["chest_yz"], s["chest_yz"]])
        elif body_reg[b.name] == "neck":
            gv = np.full(3, s["neck_r"])
        for g in b.geoms:
            g.pos = np.array(g.pos) * v
            if b.name == "torso" and g.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                g.pos = g.pos + np.array([s["chest_dx"], 0.0, 0.0])
            sz = np.array(g.size)
            if g.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                sz[:3] = sz[:3] * gv
            else:
                sz = sz * float(np.mean(gv))
            g.size = sz
            ft = np.array(g.fromto)
            if not np.isnan(ft[0]):
                g.fromto = np.r_[ft[:3] * v, ft[3:] * v]
        for j in b.joints:
            j.pos = np.array(j.pos) * v
        for st in b.sites:
            st.pos = np.array(st.pos) * v
    mesh_reg = {}
    for b in spec.bodies:
        for g in b.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname:
                mesh_reg.setdefault(g.meshname, body_reg.get(b.name, "trunk"))
    for me in spec.meshes:
        reg = mesh_reg.get(me.name, "trunk")
        me.scale = vec_scale(reg, s) if reg in ("trunk", "head") else np.full(3, float(np.mean(vec_scale(reg, s))))
    m = spec.compile()
    if skin_info is not None:
        try:
            tmp = spec.copy() if hasattr(spec, "copy") else spec
            _transfer_skin(spec, skin_info[1], m, s)
            m = spec.compile()
        except Exception as e:
            print("skin transfer failed:", repr(e)[:200]); m = spec.compile()
    dof_scale = np.zeros(m.nv)
    for dof in range(m.nv):
        b = int(m.dof_bodyid[dof]); nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "torso"
        dof_scale[dof] = float(np.mean(vec_scale(region(nm), s)))
    c = s["mass"] / float(m.body_mass.sum())
    m.body_mass[:] *= c; m.body_inertia[:] *= c
    ml = c * dof_scale ** 2
    m.dof_damping[:] *= ml; m.dof_armature[:] *= ml; m.dof_frictionloss[:] *= ml
    for j in range(m.njnt):
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            m.jnt_stiffness[j] *= float(ml[m.jnt_dofadr[j]])
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            j = int(m.actuator_trnid[a][0]); f = c * dof_scale[m.jnt_dofadr[j]]
        else:
            f = c * s["neck"]
        m.actuator_gear[a, :] *= f
        m.actuator_gainprm[a, 0] *= f
        m.actuator_biasprm[a, :3] *= f if m.actuator_biastype[a] != 0 else 1.0
    sk = float(s.get("spine_k", 1.0))
    if sk != 1.0:
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if nm.startswith("L_"):
                m.jnt_stiffness[j] *= sk
    mujoco.mj_setConst(m, mujoco.MjData(m))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    low = lowest_z(m, d)
    m.qpos0[2] -= low - 0.002
    return m


def lowest_z(m, d) -> float:
    zs = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g][2]; sz = m.geom_size[g]; t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            zs.append(c - sz[0])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            zs.append(c - abs(R[2, 2]) * sz[1] - sz[0])
        elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            zs.append(c - float(np.sqrt(np.sum((R[2, :] * sz) ** 2))))
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            zs.append(c - float(np.sum(np.abs(R[2, :]) * sz)))
        else:
            zs.append(c - m.geom_rbound[g])
    return min(zs)


def standing_dims(m, steps: float = 2.0):
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    q = d.qpos.copy()
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    A = lambda n: d.xanchor[jid(n)]
    out = dict(hip=A("hip_L_extend")[2], stifle=A("knee_L")[2], hock=A("ankle_L")[2], shoulder=A("shoulder_L_extend")[2],
               withers=d.xpos[bid("scapula_L")][2], back=abs(d.xpos[bid("C_7")][0] - d.xpos[bid("Ca_1")][0]))
    return {k: 100 * float(v) for k, v in out.items()}, float(m.body_mass.sum())


if __name__ == "__main__":
    m = build_leo_model()
    dims, mass = standing_dims(m)
    print(f"mass {mass:.2f} kg")
    for k, tgt in TARGETS_CM.items():
        print(f"  {k:9s} model {dims[k]:5.1f} cm  Leo {tgt:5.1f} cm  ratio {dims[k] / tgt:.2f}")


def apply_leo(cfg: dict, name: str = "leo") -> dict:
    import copy
    from sim.config import load_yaml
    leo = load_yaml(f"configs/model/{name}.yaml"); c = copy.deepcopy(cfg)
    c["model"] = {k: v for k, v in leo.items() if k != "overrides"}
    for sec, vals in leo.get("overrides", {}).items():
        c.setdefault(sec, {})
        for k, v in vals.items():
            c[sec][k] = v
    return c


HIND_JOINTS = ("hip_", "knee_", "ankle_")
FORE_JOINTS = ("shoulder_", "elbow_", "wrist_", "scapula_")


def apply_composition(m, d, sv, body: dict | None):
    body = body or {}
    ids = sv["act_ids"]; jid = sv["joint_ids"]
    for a, j in zip(ids, jid):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or ""
        k = 1.0
        if jn.startswith(HIND_JOINTS): k = float(body.get("muscle_hind", 1.0))
        elif jn.startswith(FORE_JOINTS): k = float(body.get("muscle_fore", 1.0))
        if k != 1.0:
            m.actuator_gainprm[a, 0] *= k; m.actuator_biasprm[a, 1] *= k; m.actuator_biasprm[a, 2] *= k
            m.actuator_forcerange[a] *= k
    f = float(body.get("fat", 0.0))
    if f > 0:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        groups = {"trunk": [i for i, n in enumerate(names) if n and (n in ("torso", "pelvis") or n.startswith(("L_", "T_")))],
                  "hindp": [i for i, n in enumerate(names) if n.startswith("upper_leg_")],
                  "forep": [i for i, n in enumerate(names) if n.startswith(("upper_arm_", "scapula_"))],
                  "neck": [i for i, n in enumerate(names) if n.startswith("C_")]}
        share = {"trunk": 0.70, "hindp": 0.15, "forep": 0.10, "neck": 0.05}
        total = float(m.body_mass.sum()) * f
        for g, idx in groups.items():
            if not idx: continue
            base = float(m.body_mass[idx].sum()); fac = 1.0 + total * share[g] / max(base, 1e-9)
            m.body_mass[idx] *= fac; m.body_inertia[idx] *= fac
        mujoco.mj_setConst(m, d)
    return dict(mass=float(m.body_mass.sum()))
