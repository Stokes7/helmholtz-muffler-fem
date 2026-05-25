# %% ── 1. Initializing Libraries ──────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import gmsh

output_path = "Projects/helmholtz-muffler-fem/gmsh/mesh/simple_duct.msh"

gmsh.initialize()
gmsh.open(output_path)

# %% ── Plot the mesh ──────────────────────────────────────────────────
from matplotlib.collections import PolyCollection

node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
xy = node_coords.reshape(-1, 3)[:, :2]
node_to_local = {int(t): i for i, t in enumerate(node_tags)}

# Extract boundary nodes while gmsh is still open
_, inlet_nodes  = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
_, outlet_nodes = gmsh.model.mesh.getNodesForPhysicalGroup(1, 2)
_, wall_nodes   = gmsh.model.mesh.getNodesForPhysicalGroup(1, 3)

inlet_xy  = xy[[node_to_local[int(n)] for n in inlet_nodes  if int(n) in node_to_local]]
outlet_xy = xy[[node_to_local[int(n)] for n in outlet_nodes if int(n) in node_to_local]]
wall_xy   = xy[[node_to_local[int(n)] for n in wall_nodes   if int(n) in node_to_local]]

# Collect quads and leftover triangles
conn_quads = []
triangles  = []
for _, entity_tag in gmsh.model.getEntities(2):
    etypes, _, enodes = gmsh.model.mesh.getElements(2, entity_tag)
    for etype, enodes_ in zip(etypes, enodes):
        if etype == 3:    # linear quad — 4 nodes
            conn = np.array(enodes_, dtype=np.int64).reshape(-1, 4)
            conn_quads.append(np.vectorize(node_to_local.get)(conn))
        elif etype == 2:  # leftover linear triangle
            conn = np.array(enodes_, dtype=np.int64).reshape(-1, 3)
            triangles.append(np.vectorize(node_to_local.get)(conn))
        elif etype == 9:  # leftover quadratic triangle — keep corners
            conn = np.array(enodes_, dtype=np.int64).reshape(-1, 6)[:, :3]
            triangles.append(np.vectorize(node_to_local.get)(conn))

conn_quads = np.vstack(conn_quads) if conn_quads else np.empty((0, 4), dtype=int)

# Plot
fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)

# Quads via PolyCollection
if len(conn_quads) > 0:
    col = PolyCollection(xy[conn_quads], facecolors="none", edgecolors="0.3", linewidths=0.4)
    ax.add_collection(col)

# Leftover triangles (usually none with RecombineAll)
if triangles:
    tri_array = np.vstack(triangles)
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tri_array)
    ax.triplot(triang, lw=0.4, color="0.3")

ax.scatter(inlet_xy[:, 0],  inlet_xy[:, 1],  s=8, c="crimson",   zorder=3, label="tag 1 · inlet")
ax.scatter(outlet_xy[:, 0], outlet_xy[:, 1], s=8, c="royalblue", zorder=3, label="tag 2 · outlet")
ax.scatter(wall_xy[:, 0],   wall_xy[:, 1],   s=8, c="gray",      zorder=3, label="tag 3 · walls")

ax.autoscale()
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"Simple duct mesh  (h = {h} m,  {len(xy)} nodes,  {len(conn_quads)} quads)")
ax.legend(loc="upper right", fontsize=9)
plt.show()

# %%
gmsh.finalize()
