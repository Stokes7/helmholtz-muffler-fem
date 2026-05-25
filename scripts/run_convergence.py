import os
import matplotlib
matplotlib.use("Agg")  # Run headless
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as dolfinx_gmsh
from mpi4py import MPI
import gmsh

# Ensure results folder exists
os.makedirs("Projects/helmholtz-muffler-fem/results/figures", exist_ok=True)


# %% ── 1. Helper Function: Programmatic Mesh Generator ─────────────────────
def generate_muffler_mesh(h_size):
    """
    Generates a quadrilateral mesh for the expansion chamber geometry
    with characteristic length h_size, returning the dolfinx mesh,
    cell tags, and facet tags.
    """
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.model.add(f"muffler_h_{h_size:.4f}")
    
    # Physical parameters (must match geometry.py)
    L_in  = 0.10   # inlet pipe length  [m]
    L_ch   = 0.20   # chamber length     [m]
    L_out = 0.10   # outlet pipe length [m]
    r     = 0.025  # pipe half-height   [m]
    R     = 0.05   # chamber half-height [m]
    x_total = L_in + L_ch + L_out
    
    # 1. Create the geometry using the OCC kernel
    rect_inlet = gmsh.model.occ.addRectangle(0.0, -r, 0.0, L_in, 2*r)
    rect_chamber = gmsh.model.occ.addRectangle(L_in, -R, 0.0, L_ch, 2*R)
    rect_outlet = gmsh.model.occ.addRectangle(L_in + L_ch, -r, 0.0, L_out, 2*r)
    
    # Fuse them into a single continuous surface
    out, _ = gmsh.model.occ.fuse(
        [(2, rect_inlet), (2, rect_chamber)],
        [(2, rect_outlet)],
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()
    
    domain_tag = out[0][1]
    
    # 2. Identify boundary edges (curves)
    curves = gmsh.model.getBoundary([(2, domain_tag)], oriented=False)
    
    inlet_tags  = []
    outlet_tags = []
    wall_tags   = []
    tol = 1e-8
    
    for dim, tag in curves:
        xc, yc, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(xc - 0.0) < tol:
            inlet_tags.append(tag)
        elif abs(xc - x_total) < tol:
            outlet_tags.append(tag)
        else:
            wall_tags.append(tag)
            
    # 3. Create physical groups
    gmsh.model.addPhysicalGroup(2, [domain_tag], tag=100)
    gmsh.model.setPhysicalName(2, 100, "Omega")
    
    gmsh.model.addPhysicalGroup(1, inlet_tags, tag=1)
    gmsh.model.setPhysicalName(1, 1, "Gamma_inlet")
    
    gmsh.model.addPhysicalGroup(1, outlet_tags, tag=2)
    gmsh.model.setPhysicalName(1, 2, "Gamma_outlet")
    
    gmsh.model.addPhysicalGroup(1, wall_tags, tag=3)
    gmsh.model.setPhysicalName(1, 3, "Gamma_wall")
    
    # 4. Generate structured-like Frontal-Quad mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_size)
    gmsh.option.setNumber("Mesh.Algorithm", 8)          # Frontal-Quad
    gmsh.option.setNumber("Mesh.RecombineAll", 1)       # Recombine triangles to quads
    
    gmsh.model.mesh.generate(2)
    
    # Convert to DOLFINx mesh
    domain, cell_tags, facet_tags, _, _, _ = dolfinx_gmsh.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )
    gmsh.finalize()
    return domain, cell_tags, facet_tags


# %% ── 2. Solver Function ──────────────────────────────────────────────────
def solve_helmholtz(domain, facet_tags, f_val):
    """
    Solves the Helmholtz equation at f_val Hz on the given domain
    with tags for boundaries, returning V, p_h, and boundary lengths.
    """
    # Air properties
    rho0 = 1.21
    c0   = 343.0
    vn   = 1e-3
    Z    = rho0 * c0
    
    omega_val = 2 * np.pi * f_val
    k_val     = omega_val / c0
    
    omega = fem.Constant(domain, complex(omega_val))
    k     = fem.Constant(domain, complex(k_val))
    
    # V space
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear form a (anechoic Robin outlet)
    a = (
          ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
        - k**2 * ufl.inner(p, v) * dx
        + 1j * omega * rho0 / Z * ufl.inner(p, v) * ds(2)
    )
    
    # Linear form L (prescribed normal velocity source)
    L = -1j * omega * rho0 * vn * ufl.conj(v) * ds(1)
    
    p_h = fem.Function(V)
    p_h.name = "pressure"
    
    problem = LinearProblem(
        a, L, u=p_h,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helmholtz",
    )
    problem.solve()
    
    one = fem.Constant(domain, complex(1.0))
    inlet_length  = fem.assemble_scalar(fem.form(one * ds(1)))
    outlet_length = fem.assemble_scalar(fem.form(one * ds(2)))
    
    return V, p_h, inlet_length, outlet_length


# %% ── 3. Run Reference Solution (Ultra-Fine Mesh) ────────────────────────
f_test = 500.0
h_ref = 0.00125  # extremely fine reference mesh (~1.3M DOFs)

print(f"Generating reference solution on ultra-fine mesh (h = {h_ref} m)...")
domain_ref, _, facet_tags_ref = generate_muffler_mesh(h_ref)
V_ref, p_ref, inlet_len_ref, outlet_len_ref = solve_helmholtz(domain_ref, facet_tags_ref, f_test)

# Compute Reference TL
ds_ref = ufl.Measure("ds", domain=domain_ref, subdomain_data=facet_tags_ref)
p_inlet_ref  = fem.assemble_scalar(fem.form(p_ref * ds_ref(1))) / inlet_len_ref
p_outlet_ref = fem.assemble_scalar(fem.form(p_ref * ds_ref(2))) / outlet_len_ref
p_inc_ref    = (p_inlet_ref - 1.21 * 343.0 * 1e-3) / 2.0
TL_ref       = 20 * np.log10(np.abs(p_inc_ref) / np.abs(p_outlet_ref))

print(f"Reference mesh DOFs: {V_ref.dofmap.index_map.size_local}")
print(f"Reference TL       : {TL_ref:.6f} dB\n")


# %% ── 4. Mesh Refinement Sweep ───────────────────────────────────────────
h_sizes = [0.02, 0.01, 0.005, 0.0025]
L2_errors  = []
H1_errors  = []
TL_errors  = []
dof_counts = []

print("Running refinement sweep...")
print(f"{'h [m]':<10} | {'DOFs':<8} | {'TL FEM [dB]':<12} | {'TL Error':<10} | {'L2 Error':<10} | {'H1 Error':<10}")
print("-" * 75)

for h in h_sizes:
    # 1. Generate coarse mesh and solve
    domain_h, _, facet_tags_h = generate_muffler_mesh(h)
    V_h, p_h, inlet_len_h, outlet_len_h = solve_helmholtz(domain_h, facet_tags_h, f_test)
    dof_counts.append(V_h.dofmap.index_map.size_local)
    
    # 2. Compute coarse TL and error
    ds_h = ufl.Measure("ds", domain=domain_h, subdomain_data=facet_tags_h)
    p_inlet_h  = fem.assemble_scalar(fem.form(p_h * ds_h(1))) / inlet_len_h
    p_outlet_h = fem.assemble_scalar(fem.form(p_h * ds_h(2))) / outlet_len_h
    p_inc_h    = (p_inlet_h - 1.21 * 343.0 * 1e-3) / 2.0
    TL_fem_h   = 20 * np.log10(np.abs(p_inc_h) / np.abs(p_outlet_h))
    
    tl_err = abs(TL_fem_h - TL_ref)
    TL_errors.append(tl_err)
    
    # 3. Interpolate coarse solution onto the ultra-fine reference mesh
    p_h_interpolated = fem.Function(V_ref)
    p_h_interpolated.interpolate(p_h)
    
    # 4. Formulate UFL error integrals on the reference mesh
    diff_p = p_h_interpolated - p_ref
    dx_ref = ufl.Measure("dx", domain=domain_ref)
    
    # L2 Error
    l2_err = np.sqrt(domain_ref.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff_p, diff_p) * dx_ref)).real, op=MPI.SUM))
    L2_errors.append(l2_err)
    
    # H1 semi-norm Error (gradient difference)
    diff_grad = ufl.grad(p_h_interpolated) - ufl.grad(p_ref)
    h1_err = np.sqrt(domain_ref.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff_grad, diff_grad) * dx_ref)).real, op=MPI.SUM))
    H1_errors.append(h1_err)
    
    print(f"{h:<10.4f} | {dof_counts[-1]:<8d} | {TL_fem_h:<12.5f} | {tl_err:<10.4e} | {l2_err:<10.4e} | {h1_err:<10.4e}")

print("-" * 75)


# %% ── 5. Estimate Convergence Rates ──────────────────────────────────────
# Fit log-log slopes: log(error) = slope * log(h) + constant
slope_L2 = np.polyfit(np.log(h_sizes), np.log(L2_errors), 1)[0]
slope_H1 = np.polyfit(np.log(h_sizes), np.log(H1_errors), 1)[0]
slope_TL = np.polyfit(np.log(h_sizes), np.log(TL_errors), 1)[0]

print(f"\nEstimated Convergence Rates (Slopes on log-log scale):")
print(f"  L2 Error Rate        : {slope_L2:.3f}   (Theoretical P1: ~2.0)")
print(f"  H1 semi-norm Rate    : {slope_H1:.3f}   (Theoretical P1: ~1.0)")
print(f"  TL Scalar Error Rate : {slope_TL:.3f}")


# %% ── 6. Plot and Save log-log Error Graph ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Plot measured errors
ax.loglog(h_sizes, L2_errors, "bo-", linewidth=1.5, markersize=6, label=f"L2 Error (slope={slope_L2:.2f})")
ax.loglog(h_sizes, H1_errors, "rs-", linewidth=1.5, markersize=6, label=f"H1 semi-norm Error (slope={slope_H1:.2f})")
ax.loglog(h_sizes, TL_errors, "g^-", linewidth=1.5, markersize=6, label=f"TL Scalar Error (slope={slope_TL:.2f})")

# Plot theoretical references
h_ref_lines = np.array(h_sizes)
# Scale theoretical lines to overlap nicely with plots
L2_theory = L2_errors[-1] * (h_ref_lines / h_ref_lines[-1])**2
H1_theory = H1_errors[-1] * (h_ref_lines / h_ref_lines[-1])**1

ax.loglog(h_sizes, L2_theory, "k--", alpha=0.5, label="Theoretical L2 Rate O(h^2)")
ax.loglog(h_sizes, H1_theory, "k:",  alpha=0.5, label="Theoretical H1 Rate O(h^1)")

ax.set_xlabel("Mesh Size h [m]", fontsize=11)
ax.set_ylabel("Numerical Error", fontsize=11)
ax.set_title("Mesh Convergence Study — Helmholtz Muffler FEM", fontsize=13, fontweight="bold")
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(loc="lower right", frameon=True, fontsize=10)

plt.tight_layout()
fig_output = "Projects/helmholtz-muffler-fem/results/figures/mesh_convergence.png"
plt.savefig(fig_output, dpi=150)
print(f"\nConvergence log-log plot successfully saved to: {fig_output}")
