import sys
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=sys.argv[1])
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
print("armatures:", [(a.name, len(a.data.bones)) for a in arms])
if arms:
    print("bones:", [b.name for b in arms[0].data.bones])
acts = list(bpy.data.actions)
print("n actions:", len(acts))
for a in acts:
    fr = a.frame_range
    print(f"  {a.name:32s} frames {fr[0]:.0f}-{fr[1]:.0f}")
print("fps:", bpy.context.scene.render.fps)
