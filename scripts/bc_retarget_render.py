import argparse, math, os, sys
import bpy
import numpy as np
from mathutils import Matrix, Vector

ap = argparse.ArgumentParser(); ap.add_argument("motion"); ap.add_argument("out")
ap.add_argument("--frames", type=int, nargs="*", default=None); ap.add_argument("--step", type=int, default=1)
ap.add_argument("--scale", type=float, default=0.66); ap.add_argument("--w", type=int, default=960); ap.add_argument("--h", type=int, default=540)
ap.add_argument("--fbx", default="data/models/bordercollie_fbx/BorderCollie_No_Anim.fbx"); ap.add_argument("--tex", default="data/models/bordercollie_textures/texture")
ap.add_argument("--mode", choices=["delta", "abs", "hybrid"], default="hybrid", help="hybrid (default): limbs copy the absolute simulated segment directions (joint angles are exact), spine/neck/head use delta (tail copies the simulated direction); delta: rotate each bone by the simulated change relative to the mean pose (keeps the model's own silhouette); abs: copy absolute directions")
ap.add_argument("--analyze", action="store_true", help="print joint-angle ranges (sim vs retargeted) and exit")
ap.add_argument("--ears", choices=["up", "fold"], default="fold", help="fold: drop the ears (leo-like) by pointing the ear bones outward/down in the head frame")
ap.add_argument("--albedo", default=None, help="replacement for BorderCollie_Albedo_1.png (e.g. results/bc_leo_albedo.png: leo-like coat)")
ap.add_argument("--cam", type=float, default=1.9, help="camera distance (m)")
ap.add_argument("--joints", default=None, help="write the projected 2D positions (pixels) of key joints per rendered frame to this npz")
a = ap.parse_args()
a.out = os.path.abspath(a.out); a.motion = os.path.abspath(a.motion)
os.makedirs(a.out, exist_ok=True)
S = a.scale

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=os.path.abspath(a.fbx))
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
arm.scale = (S, S, S)
tex_dir = os.path.abspath(a.tex)
for img in bpy.data.images:
    base = os.path.basename(img.filepath.replace("\\", "/"))
    p = os.path.join(tex_dir, base)
    if a.albedo and base == "BorderCollie_Albedo_1.png": p = os.path.abspath(a.albedo)
    if base and os.path.exists(p):
        img.filepath = p; img.reload()
bones = list(arm.data.bones)
rest = {b.name: np.array(b.matrix_local) for b in bones}
parent = {b.name: (b.parent.name if b.parent else None) for b in bones}
length = {b.name: b.length for b in bones}
order = []
def visit(b):
    order.append(b.name)
    for c in b.children: visit(c)
for b in bones:
    if b.parent is None: visit(b)

Z = np.load(a.motion, allow_pickle=True); names = list(Z["names"]); P = Z["pos"]; fps = int(Z["fps"])
ntot = len(names)
ix = {n: i for i, n in enumerate(names[:ntot - 4])}
tobc = lambda v: np.array([v[1], -v[0], v[2]])
def lm(f, n): return tobc(P[f, ix[n]])

SEG = {}
for s, sd in (("L", "L"), ("R", "R")):
    SEG[f"thigh_b.{s}"] = (f"upper_leg_{sd}", f"lower_leg_{sd}"); SEG[f"leg_b.{s}"] = (f"lower_leg_{sd}", f"foot_{sd}")
    SEG[f"shin_b.{s}"] = (f"foot_{sd}", f"foot_anchor_{sd}"); SEG[f"foot_b.{s}"] = (f"foot_anchor_{sd}", f"toe_{sd}")
    SEG[f"thigh_f.{s}"] = (f"upper_arm_{sd}", f"lower_arm_{sd}"); SEG[f"leg_f.{s}"] = (f"lower_arm_{sd}", f"hand_anchor_{sd}")
    SEG[f"shin_f.{s}"] = (f"hand_anchor_{sd}", f"finger_{sd}"); SEG[f"foot_f.{s}"] = (f"finger_{sd}", f"finger_tip_{sd}")
SEG.update({"Spine_02": ("pelvis", "L_5"), "Spine_03": ("L_5", "L_1"), "Spine_04": ("L_1", "torso"), "Spine_05": ("torso", "C_7"),
            "neck": ("C_7", "C_1"), "head": ("C_1", "skull"),
            "Tail_01": ("Ca_1", "Ca_5"), "Tail_02": ("Ca_5", "Ca_10"), "Tail_03": ("Ca_10", "Ca_15"), "Tail_04": ("Ca_15", "Ca_21"), "Tail_05": ("Ca_15", "Ca_21")})
tip_idx = {"finger_tip_L": ntot - 4, "finger_tip_R": ntot - 3, "toe_L": ntot - 2, "toe_R": ntot - 1}
def pt(f, n):
    if n in tip_idx: return tobc(P[f, tip_idx[n]])
    return lm(f, n)

def T4(v): M = np.eye(4); M[:3, 3] = v; return M
def rot_between(u, v):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v); c = float(np.dot(u, v))
    if c > 0.999999: return np.eye(3)
    if c < -0.999999:
        ax_ = np.cross(u, [1, 0, 0]); ax_ = ax_ / np.linalg.norm(ax_) if np.linalg.norm(ax_) > 1e-6 else np.array([0, 1, 0]); return 2 * np.outer(ax_, ax_) - np.eye(3)
    k = np.cross(u, v); s_ = np.linalg.norm(k); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s_ ** 2))

hipL, hipR = rest["thigh_b.L"][:3, 3], rest["thigh_b.R"][:3, 3]
P0 = 0.5 * (hipL + hipR)
f_rest = np.array([0, -1.0, 0]); l_rest = np.array([1.0, 0, 0]); u_rest = np.cross(f_rest, l_rest); B_rest = np.stack([l_rest, f_rest, u_rest], 1)

def sim_basis(f):
    ff = lm(f, "L_5") - lm(f, "pelvis"); ff /= np.linalg.norm(ff)
    ll = lm(f, "upper_leg_L") - lm(f, "upper_leg_R"); ll = ll - np.dot(ll, ff) * ff; ll /= np.linalg.norm(ll)
    return np.stack([ll, ff, np.cross(ff, ll)], 1)
_Bm = sum(sim_basis(f) for f in range(len(P))) / len(P); _U, _, _Vt = np.linalg.svd(_Bm); B_ref = _U @ _Vt
B_rest = B_rest @ np.eye(3)

def _unit(v): return v / np.linalg.norm(v)
DREF = {}
for _n, (_a, _b) in SEG.items():
    DREF[_n] = _unit(sum(_unit(pt(f, _b) - pt(f, _a)) for f in range(len(P))))

def pose_frame(f, zshift=0.0):
    ff = lm(f, "L_5") - lm(f, "pelvis"); ff /= np.linalg.norm(ff)
    ll = lm(f, "upper_leg_L") - lm(f, "upper_leg_R"); ll = ll - np.dot(ll, ff) * ff; ll /= np.linalg.norm(ll)
    uu = np.cross(ff, ll); B_sim = np.stack([ll, ff, uu], 1)
    Rm = B_sim @ B_ref.T
    mid = 0.5 * (lm(f, "upper_leg_L") + lm(f, "upper_leg_R"))
    R4 = np.eye(4); R4[:3, :3] = Rm
    X = T4(mid / S + np.array([0, 0, zshift / S])) @ R4 @ T4(-P0)
    M = {}
    for n in order:
        par = parent[n]
        if par is None:
            M[n] = X @ rest[n]; B = np.linalg.inv(rest[n]) @ M[n]
        else:
            Lb = np.linalg.inv(rest[par]) @ rest[n]
            M0 = M[par] @ Lb
            if a.ears == "fold" and n.startswith("Ear_0"):
                sgn = 1.0 if n.endswith(".L") else -1.0
                dfold = np.array([sgn * 0.85, 0.20, -0.45]) if n.startswith("Ear_01") else np.array([sgn * 0.30, 0.25, -1.0])
                Rrel = M["head"][:3, :3] @ np.linalg.inv(rest["head"][:3, :3])
                head = M0[:3, 3]; dcur = M0[:3, 1] / np.linalg.norm(M0[:3, 1]); R = rot_between(dcur, Rrel @ (dfold / np.linalg.norm(dfold)))
                Rt = np.eye(4); Rt[:3, :3] = R; M[n] = T4(head) @ Rt @ T4(-head) @ M0
            elif n in SEG:
                head = M0[:3, 3]; dcur = M0[:3, 1] / np.linalg.norm(M0[:3, 1])
                a_, b_ = SEG[n]; dtar = pt(f, b_) - pt(f, a_)
                if a.mode == "delta" or (a.mode == "hybrid" and n.startswith(("Spine", "neck", "head"))):
                    dtar = rot_between(DREF[n], dtar) @ (rest[n][:3, 1] / np.linalg.norm(rest[n][:3, 1]))
                R = rot_between(dcur, dtar)
                Rt = np.eye(4); Rt[:3, :3] = R
                M[n] = T4(head) @ Rt @ T4(-head) @ M0
            else:
                M[n] = M0
            B = np.linalg.inv(Lb) @ np.linalg.inv(M[par]) @ M[n]
        arm.pose.bones[n].matrix_basis = Matrix(B.tolist())
    return M

def claw_min_z(M):
    zs = []
    for n in ("claws_f.L", "claws_f.R", "claws_b.L", "claws_b.R"):
        tail = M[n] @ np.array([0, length[n], 0, 1.0]); zs.append(tail[2] * S)
    return min(zs)

def ang(u, v): return math.degrees(math.acos(max(-1, min(1, float(np.dot(_unit(u), _unit(v)))))))
if a.analyze:
    J = {"elbow": ("thigh_f", "leg_f"), "carpus": ("leg_f", "shin_f"), "stifle": ("thigh_b", "leg_b"), "hock": ("leg_b", "shin_b"), "shoulder-spine": ("Spine_05", "thigh_f")}
    out = {}
    for f in range(0, len(P), 2):
        M = pose_frame(f)
        for jn, (b1, b2) in J.items():
            for s in ("L",):
                n1, n2 = (f"{b1}.{s}", f"{b2}.{s}") if "." not in b1 and not b1.startswith("Spine") else (b1, f"{b2}.{s}")
                d1s = pt(f, SEG[n1][1]) - pt(f, SEG[n1][0]); d2s = pt(f, SEG[n2][1]) - pt(f, SEG[n2][0])
                d1b = M[n1][:3, 1]; d2b = M[n2][:3, 1]
                out.setdefault(jn, {"sim": [], "bc": []})["sim"].append(180 - ang(d1s, d2s)); out[jn]["bc"].append(180 - ang(d1b, d2b))
    print("MODE", a.mode)
    for jn, d in out.items():
        for k in ("sim", "bc"):
            v = np.array(d[k]); print(f"{jn:15s} {k:3s} min {v.min():6.1f} max {v.max():6.1f} ROM {np.ptp(v):5.1f} mean {v.mean():6.1f}")
    sys.exit(0)

frames = a.frames if a.frames else list(range(0, len(P), a.step))
mins = [claw_min_z(pose_frame(f)) for f in range(0, len(P), 3)]
zshift = -float(np.median(mins)) + 0.006
print("zshift", round(zshift, 4), "claw min z range", round(min(mins), 3), round(max(mins), 3))

img = bpy.data.images.new("checker", 512, 512); px = np.ones((512, 512, 4), dtype=np.float32)
yy, xx = np.mgrid[0:512, 0:512]; chk = ((xx // 256 + yy // 256) % 2).astype(np.float32)
px[..., :3] = (0.55 + 0.25 * chk)[..., None] * np.array([0.60, 0.68, 0.78]); img.pixels = px.ravel(); img.pack()
bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0)); floor = bpy.context.active_object
uvl = floor.data.uv_layers.active
for loop in uvl.data: loop.uv = (loop.uv[0] * 60, loop.uv[1] * 60)
mat = bpy.data.materials.new("floor"); mat.use_nodes = True
nt = mat.node_tree; tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img; tex.interpolation = "Closest"
nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"]); floor.data.materials.append(mat)
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"; scene.render.resolution_x = a.w; scene.render.resolution_y = a.h
sh = scene.display.shading; sh.light = "STUDIO"; sh.color_type = "TEXTURE"; sh.show_shadows = False; sh.show_cavity = False; sh.show_specular_highlight = False
scene.display.render_aa = "8"; scene.view_settings.view_transform = "Standard"
scene.world = bpy.data.worlds.new("w"); scene.world.color = (0.62, 0.74, 0.90)
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 50; cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam); scene.camera = cam
target = bpy.data.objects.new("target", None); scene.collection.objects.link(target)
con = cam.constraints.new("TRACK_TO"); con.target = target; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"

JN = ["Tail_01", "Spine_02", "thigh_b.L", "leg_b.L", "shin_b.L", "foot_b.L", "thigh_b.R", "leg_b.R", "shin_b.R", "foot_b.R", "thigh_f.R", "leg_f.R", "shin_f.R"]; J2D = []
for f in frames:
    M = pose_frame(f, zshift)
    ctr = S * np.array(M["Spine_04"][:3, 3]); target.location = (ctr[0], ctr[1], 0.22)
    cam.location = (ctr[0] - a.cam, ctr[1], 0.28)
    bpy.context.view_layer.update()
    scene.render.filepath = os.path.join(a.out, f"f{f:04d}.png"); bpy.ops.render.render(write_still=True)
    if a.joints:
        from bpy_extras.object_utils import world_to_camera_view
        row = []
        for n in JN:
            w = S * np.array(M[n][:3, 3]); c = world_to_camera_view(scene, cam, Vector(w.tolist())); row.append((c.x * a.w, (1 - c.y) * a.h))
            if n == "foot_b.R" or n == "foot_b.L":
                w2 = S * (M[n] @ np.array([0, length[n], 0, 1.0]))[:3]; c2 = world_to_camera_view(scene, cam, Vector(w2.tolist())); row.append((c2.x * a.w, (1 - c2.y) * a.h))
        J2D.append(row)
if a.joints: np.savez(a.joints, joints=np.array(J2D), names=np.array(JN + ["foot_b.L_tip", "foot_b.R_tip"]), frames=np.array(frames))
print("rendered", len(frames), "frames ->", a.out)
