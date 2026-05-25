# %% ── 1. Initializing Model ─────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import gmsh

# Physical parameters
L_in  = 0.10   # inlet pipe length  [m]
L_ch   = 0.20   # chamber length     [m]
L_out = 0.10   # outlet pipe length [m]
r     = 0.025  # pipe half-height   [m]
R     = 0.05   # chamber half-height [m]
h     = 0.005   # mesh size
 
gmsh.initialize()
gmsh.model.add("simple_duct")
 


# %% ── 2. Create the geometry using the OCC kernel ───────────────────────
rect_inlet = gmsh.model.occ.addRectangle(0.0, -r, 0.0, L_in, 2*r)
rect_chamber = gmsh.model.occ.addRectangle(L_in, -R, 0.0, L_ch, 2*R)
rect_outlet = gmsh.model.occ.addRectangle(L_in + L_ch, -r, 0.0, L_out, 2*r)

# fuse(object, tool) — merges tool into object, removes internal edges
out, _ = gmsh.model.occ.fuse(
    [(2, rect_inlet), (2, rect_chamber)],  # object
    [(2, rect_outlet)],                    # tool
    removeObject=True,
    removeTool=True,
)

# Synchronize the geometry with the mesh
gmsh.model.occ.synchronize()

domain_tag = out[0][1]
print(f"Domain tag after fuse: {domain_tag}")

# %% ── 3. Find and classify the boundary edges ───────────────────────────

# getBoundary returns a list of (dim, tag) pairs for all entities on
# the boundary of our surface.  dim=1 means a curve (edge).

curves = gmsh.model.getBoundary([(2, domain_tag)], oriented=False)
print(f"Found {len(curves)} boundary curves: {curves}")
# For a rectangle you should see exactly 4 curves.
 
inlet_tags  = []
outlet_tags = []
wall_tags   = []
tol = 1e-9   # geometric tolerance for comparisons
x_total = L_in + L_ch + L_out
 
for dim, tag in curves:
    # getCenterOfMass returns (x, y, z) of the midpoint of the curve.
    xc, yc, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
    print(f"  curve tag={tag:3d}  centre=({xc:.4f}, {yc:.4f})")
 
    if abs(xc - 0.0) < tol:
        inlet_tags.append(tag)
    elif abs(xc - x_total) < tol:
        outlet_tags.append(tag)
    else:
        wall_tags.append(tag)
 
print(f"inlet_tags  = {inlet_tags}")
print(f"outlet_tags = {outlet_tags}")
print(f"wall_tags   = {wall_tags}")

# %% ── 4. Define physical groups ─────────────────────────────────────────
# addPhysicalGroup(dim, [list of entity tags], tag=N)
#   dim  : 2 for surfaces, 1 for curves, 0 for points
#   tag  : the integer label FEniCSx will see as a facet marker
 
# The 2-D domain (the fluid region)
gmsh.model.addPhysicalGroup(2, [domain_tag], tag=100)
gmsh.model.setPhysicalName(2, 100, "Omega")
 
# The inlet edge
gmsh.model.addPhysicalGroup(1, inlet_tags, tag=1)
gmsh.model.setPhysicalName(1, 1, "Gamma_inlet")
 
# The outlet edge
gmsh.model.addPhysicalGroup(1, outlet_tags, tag=2)
gmsh.model.setPhysicalName(1, 2, "Gamma_outlet")
 
# The rigid walls (top + bottom collected into one group)
gmsh.model.addPhysicalGroup(1, wall_tags, tag=3)
gmsh.model.setPhysicalName(1, 3, "Gamma_wall")

# %% ── 5. Set mesh size and generate ────────────────────────────────────
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
 
# generate(2) creates a 2-D triangular mesh.
# Activar algoritmo de cuadriláteros
gmsh.option.setNumber("Mesh.Algorithm", 8)          # Frontal-Quad
gmsh.option.setNumber("Mesh.RecombineAll", 1)       
gmsh.model.mesh.generate(2)
 
# Write the mesh to disk in gmsh format (.msh).
# FEniCSx can read this file directly.
output_file = "Projects/helmholtz-muffler-fem/gmsh/mesh/simple_duct.msh"
gmsh.write(output_file)
print(f"\nMesh written to: {output_file}")



# %% ── 7. Plot the mesh ──────────────────────────────────────────────────
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




# %% ── 7. Clean up ────────────────────────────────────────────────────────

# gmsh.finalize()
