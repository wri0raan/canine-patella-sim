import sys
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=sys.argv[1])
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
print("armature scale", tuple(arm.scale), "dimensions", tuple(round(x, 2) for x in arm.dimensions))
for b in arm.data.bones:
    if any(k in b.name for k in ("eyelid", "Ear", "tongue", "Mouth", "eye", "nose", "claws")) and not b.name.startswith("claws"):
        continue
    h, t = b.head_local, b.tail_local
    print(f"{b.name:14s} parent={(b.parent.name if b.parent else '-'):12s} head=({h.x:6.3f},{h.y:6.3f},{h.z:6.3f}) tail=({t.x:6.3f},{t.y:6.3f},{t.z:6.3f})")
