# %% ── 1. Initializing Libraries ──────────────────────────────────────────────────
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import gmsh

try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd()

filename = "simple_duct"
# filename = "extended_duct"

output_path = project_root / "gmsh" / "mesh" / (filename + ".msh") 
# output_path = project_root / "gmsh" / "mesh" / "extended_duct.msh"


gmsh.initialize()
gmsh.open(str(output_path))

# %% ── Plot the mesh ──────────────────────────────────────────────────
from matplotlib.collections import LineCollection, PolyCollection

node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
xy = node_coords.reshape(-1, 3)[:, :2]
node_to_local = {int(t): i for i, t in enumerate(node_tags)}

def _get_boundary_segments(phys_tag):
    segs = []
    for etag in gmsh.model.getEntitiesForPhysicalGroup(1, phys_tag):
        etypes, _, enodes = gmsh.model.mesh.getElements(1, etag)
        for etype, enodes_ in zip(etypes, enodes):
            if etype == 1:  # 2-node line
                for e in np.array(enodes_, dtype=np.int64).reshape(-1, 2):
                    idx = [node_to_local[n] for n in e if n in node_to_local]
                    if len(idx) == 2:
                        segs.append([xy[idx[0]], xy[idx[1]]])
    return segs

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

# Read mesh size from file info
h_size = gmsh.model.mesh.getElementQualities(
    gmsh.model.mesh.getElementsByType(3)[0][:1], "minSJ"
) if len(gmsh.model.mesh.getElementsByType(3)[0]) > 0 else ["?"]

# Plot
fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

# Quads via PolyCollection
if len(conn_quads) > 0:
    col = PolyCollection(xy[conn_quads], facecolors="none", edgecolors="0.5", linewidths=0.3)
    ax.add_collection(col)

# Leftover triangles (usually none with RecombineAll)
if triangles:
    tri_array = np.vstack(triangles)
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tri_array)
    ax.triplot(triang, lw=0.3, color="0.5")

# Boundary tags as colored line segments
ax.add_collection(LineCollection(_get_boundary_segments(1), colors="crimson",   linewidths=2.5, zorder=4, label="tag 1 · inlet"))
ax.add_collection(LineCollection(_get_boundary_segments(2), colors="royalblue", linewidths=2.5, zorder=4, label="tag 2 · outlet"))
ax.add_collection(LineCollection(_get_boundary_segments(3), colors="gray",      linewidths=1.5, zorder=3, label="tag 3 · walls"))

ax.autoscale()
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
# ax.set_title(f"{filename}  —  {len(xy)} nodes,  {len(conn_quads)} quads")
ax.legend(loc="upper right", fontsize=9)

figures_dir = project_root / "results" / "figures"
out = Path(figures_dir) / f"{filename}.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")                         
print(f"Mesh image saved to: {out}")
plt.show()

# %%
gmsh.finalize()
