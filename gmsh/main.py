# %% ################################################################
# Minimal Geometry Pattern
#####################################################################

import gmsh

gmsh.initialize()
gmsh.model.add("demo")

# Rectangle
# Arguments: (x, y, z, dx, dy)
rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 2.0, 1.0)
# Synchronize the model with gmsh
gmsh.model.occ.synchronize()

# Physical groups
# Arguments: (dimension,[list of entities], indentifies])
gmsh.model.addPhysicalGroup(2, [rect], tag=100)
# Arguments: (dimension, tag, name)
gmsh.model.setPhysicalName(2, 100, "domain")

# Meshing
# Arguments: (dim of the meshing)
gmsh.model.mesh.generate(2)
gmsh.write("./project_01/results/demo_rect.msh")
gmsh.finalize()


# %% ################################################################
# %% Visual Boundary-Tag Sketch
#####################################################################
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

fig, ax = plt.subplots(figsize=(9, 3))

# Plate
plate = Rectangle((0.0, 0.0), 2.8, 1.0, fill=False, linewidth=2.0, edgecolor="black")
ax.add_patch(plate)

# Holes (perforated variant)
for xc in [0.9, 1.4, 1.9]:
    ax.add_patch(Circle((xc, 0.5), 0.14, fill=False, linewidth=1.5, edgecolor="dimgray"))

# Boundary labels and arrows
ax.annotate("tag 1: Gamma_D_hot", xy=(0.0, 0.5), xytext=(-0.6, 0.82),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="crimson"),
            color="crimson", fontsize=11)
ax.annotate("tag 2: Gamma_D_cold", xy=(2.8, 0.5), xytext=(2.95, 0.82),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="royalblue"),
            color="royalblue", fontsize=11)
ax.annotate("tag 3: Gamma_N_insulated", xy=(1.4, 1.0), xytext=(0.85, 1.18),
            arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)
ax.annotate("tag 3 on bottom + hole walls", xy=(1.4, 0.0), xytext=(0.75, -0.20),
            arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10)
ax.text(1.12, 0.46, "tag 100: Omega", fontsize=11)

ax.set_aspect("equal")
ax.set_xlim(-0.8, 3.5)
ax.set_ylim(-0.35, 1.35)
ax.set_title("Boundary tags for the thermal bridge study")
ax.axis("off")
plt.show()


# %% ################################################################
# %% Solid Bridge
#####################################################################
import gmsh


def build_bridge_solid(filename="bridge_solid.msh", L=2.8, H=1.0, h=0.08):
    gmsh.initialize()
    gmsh.model.add("bridge_solid")

    plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, H)
    gmsh.model.occ.synchronize()

    # Collect boundaries for tagging.
    curves = gmsh.model.getBoundary([(2, plate)], oriented=False)
    print(f"Curves (dim, tag): {curves}")            

    left, right, top, bottom = [], [], [], []
    for dim, tag in curves:
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(x - 0.0) < 1e-9:
            left.append(tag)
        elif abs(x - L) < 1e-9:
            right.append(tag)
        elif abs(y - H) < 1e-9:
            top.append(tag)
        elif abs(y - 0.0) < 1e-9:
            bottom.append(tag)

    print(f"left: {left}, right: {right}, top: {top}, bottom: {bottom}")
   
    # Add physical group for 2D domain
    gmsh.model.addPhysicalGroup(2, [plate], tag=100)
    gmsh.model.setPhysicalName(2, 100, "Omega")
    
    # Add physical groups for 1D boundarie left
    gmsh.model.addPhysicalGroup(1, left, tag=1)
    gmsh.model.setPhysicalName(1, 1, "Gamma_D_hot")
    
    # Add physical groups for 1D boundarie right
    gmsh.model.addPhysicalGroup(1, right, tag=2)
    gmsh.model.setPhysicalName(1, 2, "Gamma_D_cold")
    
    # Add physical groups for 1D boundarie top and bottom#
    gmsh.model.addPhysicalGroup(1, top + bottom, tag=3)
    gmsh.model.setPhysicalName(1, 3, "Gamma_N_insulated")

    # Add physical groups for 1D boundarie top and bottom
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    gmsh.model.mesh.generate(2)
    gmsh.write("./project_01/results/" + filename)
    gmsh.finalize()

build_bridge_solid()

# %% ################################################################
# %% Perforated bridge (holes increase thermal resistance)
#####################################################################

import gmsh


def build_bridge_holes(filename="bridge_holes.msh", L=2.8, H=1.0, h=0.07):
    gmsh.initialize()
    gmsh.model.add("bridge_holes")

    plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, H)

    # Three circular cut-outs along the centerline.
    r = 0.14
    holes = [
        gmsh.model.occ.addDisk(0.9, 0.5, 0.0, r, r),
        gmsh.model.occ.addDisk(1.4, 0.5, 0.0, r, r),
        gmsh.model.occ.addDisk(1.9, 0.5, 0.0, r, r),
    ]

    out, _ = gmsh.model.occ.cut([(2, plate)], [(2, tag) for tag in holes], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    # Expect one remaining 2D entity after cut.
    domain_tag = out[0][1]
    curves = gmsh.model.getBoundary([(2, domain_tag)], oriented=False)

    left, right, outer_rest, hole_walls = [], [], [], []
    for dim, tag in curves:
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(x - 0.0) < 1e-9:
            left.append(tag)
        elif abs(x - L) < 1e-9:
            right.append(tag)
        elif y < -1e-9 or y > H + 1e-9:
            hole_walls.append(tag)
        else:
            outer_rest.append(tag)

    # Robust split: any curve that is neither left nor right is insulated.
    insulated = [tag for _, tag in curves if tag not in left + right]

    gmsh.model.addPhysicalGroup(2, [domain_tag], tag=100)
    gmsh.model.setPhysicalName(2, 100, "Omega")

    gmsh.model.addPhysicalGroup(1, left, tag=1)
    gmsh.model.setPhysicalName(1, 1, "Gamma_D_hot")

    gmsh.model.addPhysicalGroup(1, right, tag=2)
    gmsh.model.setPhysicalName(1, 2, "Gamma_D_cold")

    gmsh.model.addPhysicalGroup(1, insulated, tag=3)
    gmsh.model.setPhysicalName(1, 3, "Gamma_N_insulated")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(2)
    gmsh.write("./project_01/results/" + filename)
    gmsh.finalize()

build_bridge_holes()

# %% ################################################################
# %% Visualize the Actual Meshes (Solid vs Perforated)
#####################################################################


import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def read_triangles_from_msh(filename):
    if gmsh.isInitialized():
        gmsh.finalize()

    gmsh.initialize()
    gmsh.model.add("view_mesh")
    gmsh.merge(filename)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    xy = node_coords.reshape(-1, 3)[:, :2]
    node_to_local = {int(tag): i for i, tag in enumerate(node_tags)}

    triangles = []
    for _, entity_tag in gmsh.model.getEntities(2):
        element_types, _, element_nodes = gmsh.model.mesh.getElements(2, entity_tag)
        for e_type, e_nodes in zip(element_types, element_nodes):
            # type 2: linear triangles, type 9: quadratic triangles
            if e_type == 2:
                conn = np.array(e_nodes, dtype=np.int64).reshape(-1, 3)
            elif e_type == 9:
                conn = np.array(e_nodes, dtype=np.int64).reshape(-1, 6)[:, :3]
            else:
                continue
            triangles.append(np.vectorize(node_to_local.get)(conn))

    gmsh.finalize()

    if len(triangles) == 0:
        raise RuntimeError(f"No triangular elements found in {filename}")

    tri = np.vstack(triangles)
    return xy, tri


def plot_msh(ax, filename, title):
    xy, tri = read_triangles_from_msh(filename)
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tri)
    ax.triplot(triang, lw=0.35, color="0.2")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
plot_msh(axes[0], "./project_01/results/bridge_solid.msh", "Solid bridge mesh")
plot_msh(axes[1], "./project_01/results/bridge_holes.msh", "Perforated bridge mesh")
plt.show()



# %% ################################################################
# %% Coarse vs fine mesh
#####################################################################
# Regenerate solid bridge with two mesh sizes
build_bridge_solid(filename="bridge_solid_coarse.msh", h=0.16)
build_bridge_solid(filename="bridge_solid_fine.msh", h=0.05)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
plot_msh(axes[0], "./project_01/results/bridge_solid_coarse.msh", "Solid bridge (coarse mesh, h=0.16)")
plot_msh(axes[1], "./project_01/results/bridge_solid_fine.msh", "Solid bridge (fine mesh, h=0.05)")
plt.show()



# %%
