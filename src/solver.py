# %% ── 1. Initializing Solver ─────────────────────────────────────────────

from dolfinx.io import gmsh as dolfinx_gmsh
from mpi4py import MPI

import gmsh

if gmsh.isInitialized():
    gmsh.finalize()

gmsh.initialize()          # ← falta esta línea
gmsh.open("Projects/helmholtz-muffler-fem/gmsh/mesh/simple_duct.msh")

groups = gmsh.model.getPhysicalGroups()
print(f"Physical groups found: {groups}")

domain, cell_tags, facet_tags, _, _, _ = dolfinx_gmsh.model_to_mesh(
    gmsh.model,
    MPI.COMM_WORLD,
    rank=0,
    gdim=2
)

gmsh.finalize()

print(f"Mesh loaded: {domain.topology.dim}D")
print(f"Number of cells   : {domain.topology.index_map(2).size_local}")
print(f"Number of vertices: {domain.topology.index_map(0).size_local}")
print(f"Cell tags    : {cell_tags}")
print(f"Facet tags   : {facet_tags}")


# %% ── 2. Physical parameters and function space ──────────────────────────
import numpy as np
import ufl
from dolfinx import fem

# Air properties
rho0 = 1.21       # density          [kg/m³]
c0   = 343.0      # speed of sound   [m/s]

# Frequency to solve (single frequency first — loop comes later)
f     = 500.0              # Hz
omega = 2 * np.pi * f      # angular frequency [rad/s]
k     = omega / c0         # wavenumber        [1/m]
vn    = 1e-3               # inlet velocity    [m/s]
Z     = rho0 * c0          # anechoic impedance [Pa·s/m]

print(f"Frequency : {f} Hz")
print(f"Wavenumber: {k:.4f} 1/m")
print(f"Wavelength: {2*np.pi/k:.4f} m")
print(f"Impedance : {Z:.2f} Pa·s/m")

# Function space — Lagrange P1, complex-valued via PETSc
# "Lagrange" = continuous piecewise polynomial
# 1           = polynomial order (linear)
V = fem.functionspace(domain, ("Lagrange", 1))
print(f"Function space: {V.element.signature}")
print(f"Number of DOFs: {V.dofmap.index_map.size_local}")





# %% ── 3. Integration measures and trial/test functions ───────────────────

# dx integrates over the 2D domain
# ds integrates over boundary facets — ds(1)=inlet, ds(2)=outlet, ds(3)=wall
dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Trial function — the unknown p_hat we are solving for
p = ufl.TrialFunction(V)

# Test function — the q that multiplies the weak form
v = ufl.TestFunction(V)

print("Measures and functions defined")
print(f"  dx integrates over: {domain.topology.dim}D cells")
print("  ds(1) = inlet   (tag 1)")
print("  ds(2) = outlet  (tag 2)")
print("  ds(3) = wall    (tag 3) -- will vanish naturally")




# %% ── 4. Weak form ───────────────────────────────────────────────────────

# Bilinear form a(p, v)
# Term 1: ∫ ∇p·∇v dΩ        — from integration by parts
# Term 2: −k² ∫ p·v dΩ      — Helmholtz term
# Term 3: jωρ₀/Z ∫ p·v dS   — Robin BC at outlet (anechoic)
a = (
      ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
    - k**2 * ufl.inner(p, v) * dx
    + 1j * omega * rho0 / Z * ufl.inner(p, v) * ds(2)
)

# Linear form L(v)
# Term 1: −jωρ₀·vn ∫ v dS   — Neumann BC at inlet (velocity source)
L = -1j * omega * rho0 * vn * ufl.conj(v) * ds(1)

print("Weak form defined")
print("  a(p,v) : bilinear form assembled")
print("  L(v)   : linear form assembled")



# %% Verification ─────────────────────────────────────────────────────────
from petsc4py import PETSc

print(f"PETSc scalar type: {PETSc.ScalarType}")


# %% ── 5. Solver ──────────────────────────────────────────────────────────
from dolfinx.fem.petsc import LinearProblem

# LinearProblem assembles A and b, then solves A·x = b
# petsc_options select the linear solver:
#   ksp_type = preonly  — direct solver (no iterative method)
#   pc_type  = lu       — LU factorization
#   mumps    — parallel sparse direct solver, best for complex systems
p_h = fem.Function(V)
p_h.name = "pressure"

problem = LinearProblem(
    a,
    L,
    u=p_h,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="helmholtz",
)

problem.solve()
print("Solved successfully")
print(f"  max |Re(p)| = {np.max(np.abs(p_h.x.array.real)):.4e} Pa")
print(f"  max |Im(p)| = {np.max(np.abs(p_h.x.array.imag)):.4e} Pa")
print(f"  max |p|     = {np.max(np.abs(p_h.x.array)):.4e} Pa")





# %% ── 6. Postprocessing with PyVista ─────────────────────────────────────
import pyvista as pv
from dolfinx import plot

# Build PyVista mesh from FEniCSx mesh
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# Attach solution to mesh
grid["Re(p)"] = p_h.x.array.real
grid["Im(p)"] = p_h.x.array.imag
grid["|p|"]   = np.abs(p_h.x.array)

# Plot
plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 400))

plotter.subplot(0, 0)
plotter.add_mesh(grid.copy(), scalars="Re(p)", cmap="RdBu_r", show_edges=False)
plotter.add_text(f"Re(p)  {f:.0f} Hz", font_size=10)
plotter.view_xy()

plotter.subplot(0, 1)
plotter.add_mesh(grid.copy(), scalars="|p|", cmap="viridis", show_edges=False)
plotter.add_text(f"|p|  {f:.0f} Hz", font_size=10)
plotter.view_xy()

plotter.subplot(0, 2)
plotter.add_mesh(grid.copy(), scalars="Im(p)", cmap="RdBu_r", show_edges=False)
plotter.add_text(f"Im(p)  {f:.0f} Hz", font_size=10)
plotter.view_xy()

plotter.show()





# %% ── 7. Cleanup ────────────────────────────────────────────────────────
print("First 3 quads node indices:")
print(quad_conn[:3])
print("Their coordinates:")
for q in quad_conn[:3]:
    print(coords[q])


# %% ── 7. Cleanup ────────────────────────────────────────────────────────
# gmsh.finalize()
