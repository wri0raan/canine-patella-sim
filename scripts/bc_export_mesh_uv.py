import os, sys
import bpy
import numpy as np
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=os.path.abspath("data/models/bordercollie_fbx/BorderCollie_No_Anim.fbx"))
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]; ob = [o for o in bpy.data.objects if o.type == "MESH"][0]
me = ob.data; Mw = np.array(ob.matrix_world); Ma = np.linalg.inv(np.array(arm.matrix_world))
M = Ma @ Mw
V = np.array([v.co[:] for v in me.vertices], np.float64); V = (np.c_[V, np.ones(len(V))] @ M.T)[:, :3]
me.calc_loop_triangles()
tri_v = np.array([t.vertices[:] for t in me.loop_triangles]); tri_l = np.array([t.loops[:] for t in me.loop_triangles])
uv = np.array([l.uv[:] for l in me.uv_layers.active.data])
np.savez("results/bc_mesh_uv.npz", V=V, tri_v=tri_v, tri_l=tri_l, uv=uv)
print("verts", V.shape, "bbox", V.min(0).round(3), V.max(0).round(3), "tris", len(tri_v))
