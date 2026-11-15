import argparse, os
import bpy
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("out")
ap.add_argument("--frames", type=int, nargs="*", default=None); ap.add_argument("--step", type=int, default=1)
ap.add_argument("--w", type=int, default=960); ap.add_argument("--h", type=int, default=540); ap.add_argument("--cam", type=float, default=1.35); ap.add_argument("--fur", type=float, default=1.0)
a = ap.parse_args()
a.prefix = os.path.abspath(a.prefix); a.out = os.path.abspath(a.out); os.makedirs(a.out, exist_ok=True)
V = np.load(a.prefix + "_verts.npy"); Mz = np.load(a.prefix + "_mesh.npz", allow_pickle=True)
faces = Mz["faces"]; col = Mz["colors"]; txs = Mz["torso_x"]; fur = Mz["fur"] if "fur" in Mz.files else np.zeros(len(col), np.float32)

bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("leo"); mesh.from_pydata(V[0].tolist(), [], faces.tolist()); mesh.update()
obj = bpy.data.objects.new("leo", mesh); bpy.context.scene.collection.objects.link(obj)
mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
ca = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
rgba = np.concatenate([col, np.ones((len(col), 1), np.float32)], 1).astype(np.float32)
ca.data.foreach_set("color", rgba.ravel()); mesh.color_attributes.active_color = ca

bpy.ops.mesh.primitive_grid_add(x_subdivisions=120, y_subdivisions=120, size=60, location=(0, 0, 0)); floor = bpy.context.active_object
fm = floor.data; fca = fm.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
cols = np.zeros((len(fm.loops), 4), np.float32); cols[:, 3] = 1
for pi, poly in enumerate(fm.polygons):
    r_, c_ = divmod(pi, 120); v = 0.55 + 0.25 * ((r_ + c_) % 2); cols[poly.loop_start:poly.loop_start + poly.loop_total, :3] = v * np.array([0.60, 0.68, 0.78])
fca.data.foreach_set("color", cols.ravel()); fm.color_attributes.active_color = fca
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"; scene.render.resolution_x = a.w; scene.render.resolution_y = a.h
sh = scene.display.shading; sh.light = "STUDIO"; sh.show_shadows = False; sh.show_cavity = False; sh.show_specular_highlight = False
scene.display.render_aa = "8"; scene.view_settings.view_transform = "Standard"
scene.world = bpy.data.worlds.new("w"); scene.world.color = (0.62, 0.74, 0.90)
sh.color_type = "VERTEX"
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 50; cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam); scene.camera = cam
target = bpy.data.objects.new("target", None); scene.collection.objects.link(target)
con = cam.constraints.new("TRACK_TO"); con.target = target; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"

_, grp = np.unique(np.round(Mz["rest"], 4), axis=0, return_inverse=True); grp = grp.ravel(); ngrp = grp.max() + 1
cnt = np.bincount(grp, minlength=ngrp).astype(np.float32)
frames = a.frames if a.frames else list(range(0, len(V), a.step))
for f in frames:
    mesh.vertices.foreach_set("co", V[f].ravel()); mesh.update()
    nrm = np.empty(len(V[f]) * 3, np.float32); mesh.vertex_normals.foreach_get("vector", nrm)
    nrm = nrm.reshape(-1, 3); acc = np.zeros((ngrp, 3), np.float32); np.add.at(acc, grp, nrm); nrm = acc[grp] / cnt[grp][:, None]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    mesh.vertices.foreach_set("co", (V[f] + nrm * fur[:, None] * a.fur).ravel()); mesh.update()
    target.location = (txs[f][0], txs[f][1], 0.17); cam.location = (txs[f][0], txs[f][1] - a.cam, 0.22)
    scene.render.filepath = os.path.join(a.out, f"f{f:04d}.png"); bpy.ops.render.render(write_still=True)
print("rendered", len(frames), "->", a.out)
