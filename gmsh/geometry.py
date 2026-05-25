# %% ── 1. Initializing Model ─────────────────────────────────────────────
import gmsh

# Physical parameters
L_in  = 0.10   # inlet pipe length  [m]
L_ch   = 0.20   # chamber length     [m]
L_out = 0.10   # outlet pipe length [m]
r     = 0.025  # pipe half-height   [m]
R     = 0.05   # chamber half-height [m]
 
from dolfinx.io import gmsh as dolfinx_gmsh
from mpi4py import MPI


def generate_muffler_mesh(h_size, output_file=None, recombine_quads=True):
    """
    Programmatically generates the geometry and mesh for a simple expansion chamber.
    Returns (domain, cell_tags, facet_tags) directly.
    """
    if gmsh.isInitialized():
        gmsh.finalize()
        
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Silence all Gmsh console output
    gmsh.model.add("simple_duct")
 
    # ── 2. Create the geometry using the OCC kernel ───────────────────────
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
    # print(f"Domain tag after fuse: {domain_tag}")

    # ── 3. Find and classify the boundary edges ───────────────────────────

    # getBoundary returns a list of (dim, tag) pairs for all entities on
    # the boundary of our surface.  dim=1 means a curve (edge).

    curves = gmsh.model.getBoundary([(2, domain_tag)], oriented=False)
    # print(f"Found {len(curves)} boundary curves: {curves}")
    # For a rectangle you should see exactly 4 curves.
     
    inlet_tags  = []
    outlet_tags = []
    wall_tags   = []
    tol = 1e-9   # geometric tolerance for comparisons
    x_total = L_in + L_ch + L_out
     
    for dim, tag in curves:
        # getCenterOfMass returns (x, y, z) of the midpoint of the curve.
        xc, yc, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        # print(f"  curve tag={tag:3d}  centre=({xc:.4f}, {yc:.4f})")
     
        if abs(xc - 0.0) < tol:
            inlet_tags.append(tag)
        elif abs(xc - x_total) < tol:
            outlet_tags.append(tag)
        else:
            wall_tags.append(tag)
     
    # print(f"inlet_tags  = {inlet_tags}")
    # print(f"outlet_tags = {outlet_tags}")
    # print(f"wall_tags   = {wall_tags}")

    # ── 4. Define physical groups ─────────────────────────────────────────
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

    # ── 5. Set mesh size and generate ────────────────────────────────────
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_size)
     
    # generate(2) creates a 2-D triangular mesh.
    if recombine_quads:
        gmsh.option.setNumber("Mesh.Algorithm", 8)          # Frontal-Quad
        gmsh.option.setNumber("Mesh.RecombineAll", 1)       
        # Subdivision prevents mixed cell types and unsupported hybrids
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        
    gmsh.model.mesh.generate(2)
     
    # Write the mesh to disk in gmsh format (.msh).
    if output_file:
        gmsh.write(output_file)
        print(f"\nMesh written to: {output_file}")
        
    domain, cell_tags, facet_tags, _, _, _ = dolfinx_gmsh.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )
    gmsh.finalize()
    return domain, cell_tags, facet_tags

# %% ── 6. Run ─────────────────────────────────────────────
if __name__ == "__main__":
    h     = 0.002   # mesh size
    output_path = "Projects/helmholtz-muffler-fem/gmsh/mesh/simple_duct.msh"
    print(f"Generating mesh with h = {h} m...")
    generate_muffler_mesh(h, output_file=output_path, recombine_quads=True)
    print(f"Mesh saved successfully to: {output_path}")
    # %%
