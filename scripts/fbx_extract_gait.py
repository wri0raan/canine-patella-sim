import sys

import bpy
import numpy as np

fbx, out = sys.argv[1], sys.argv[2]
clips = sys.argv[3:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
fps = bpy.context.scene.render.fps


def ang(u, v):
    u, v = np.asarray(u)[1:], np.asarray(v)[1:]
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


res = {"fps": fps}
for clip in clips:
    act = [a for a in bpy.data.actions if a.name.endswith("|" + clip)][0]
    arm.animation_data_create()
    arm.animation_data.action = act
    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    rows = {}
    for f in range(f0, f1 + 1):
        bpy.context.scene.frame_set(f)
        P = lambda n: np.array(arm.pose.bones[n].head)
        trunk = P("Spine_05") - P("Spine_base")
        for leg, side, kind in (("LH", "L", "b"), ("RH", "R", "b"), ("LF", "L", "f"), ("RF", "R", "f")):
            sh, kn, hk, pw = (P(f"{n}_{kind}.{side}") for n in ("thigh", "leg", "shin", "foot"))
            r = rows.setdefault(leg, [])
            r.append([ang(trunk, kn - sh),
                      ang(sh - kn, hk - kn),
                      ang(kn - hk, pw - hk),
                      pw[2],
                      pw[1]])
    for leg, r in rows.items():
        res[f"{clip}/{leg}"] = np.array(r)
    print(clip, "frames", f0, f1, "fps", fps)
np.savez(out, **res)
print("saved", out)
