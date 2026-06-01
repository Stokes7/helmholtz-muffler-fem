# %% ── 1. Import Libraries ────────────────────────
import os

import matplotlib

matplotlib.use("Agg")  # Run headless
import sys

import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

# Import our modular geometry and solver components!
from pathlib import Path
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd()

sys.path.append(str(project_root / "gmsh"))
sys.path.append(str(project_root / "src"))
from geometry import generate_muffler_mesh

from solver import HelmholtzSolver

# Ensure results folder exists
figures_dir = project_root / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# %% ── 2. Run Reference Solution (Ultra-Fine Mesh) ────────────────────────
f_test = 500.0
h_ref = 0.0002  # extremely fine reference mesh (~20k DOFs in 2D)

print(f"Generating reference solution on ultra-fine mesh (h = {h_ref} m)...")
domain_ref, _, facet_tags_ref = generate_muffler_mesh(h_ref, recombine_quads=False)
solver_ref = HelmholtzSolver(domain_ref, facet_tags_ref)
TL_ref, p_inlet_ref, p_outlet_ref, p_inc_ref = solver_ref.solve(f_test)

print(f"Reference mesh DOFs: {solver_ref.V.dofmap.index_map.size_local}")
print(f"Reference TL       : {TL_ref:.6f} dB\n")


# %% ── 3. Mesh Refinement Sweep ───────────────────────────────────────────
h_sizes = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625]
L2_errors  = []
H1_errors  = []
TL_errors  = []
dof_counts = []

print("Running refinement sweep...")
print(f"{'h [m]':<10} | {'DOFs':<8} | {'TL FEM [dB]':<12} | {'TL Error':<10} | {'L2 Error':<10} | {'H1 Error':<10}")
print("-" * 75)

for h in h_sizes:
    # 1. Generate coarse mesh and solve
    domain_h, _, facet_tags_h = generate_muffler_mesh(h, recombine_quads=False)
    solver_h = HelmholtzSolver(domain_h, facet_tags_h)
    TL_fem_h, p_inlet_h, p_outlet_h, p_inc_h = solver_h.solve(f_test)
    dof_counts.append(solver_h.V.dofmap.index_map.size_local)
    
    tl_err = abs(TL_fem_h - TL_ref)
    TL_errors.append(tl_err)
    
    # 2. Interpolate the fine reference solution onto the coarse mesh
    num_cells = domain_h.topology.index_map(domain_h.topology.dim).size_local
    cells_h = np.arange(num_cells, dtype=np.int32)
    
    interp_data = fem.create_interpolation_data(
        solver_h.V,
        solver_ref.V,
        cells_h,
        padding=1e-6
    )
    p_ref_interpolated = fem.Function(solver_h.V)
    p_ref_interpolated.interpolate_nonmatching(solver_ref.p_h, cells_h, interp_data)
    
    # 3. Formulate UFL error integrals on the coarse mesh
    diff_p = solver_h.p_h - p_ref_interpolated
    dx_h = ufl.Measure("dx", domain=domain_h)
    
    # L2 Error
    l2_err = np.sqrt(domain_h.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff_p, diff_p) * dx_h)).real, op=MPI.SUM))
    L2_errors.append(l2_err)
    
    # H1 norm Error
    diff_grad = ufl.grad(solver_h.p_h) - ufl.grad(p_ref_interpolated)
    h1_err = np.sqrt(domain_h.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff_grad, diff_grad) * dx_h)).real, op=MPI.SUM))
    H1_errors.append(h1_err)
    
    print(f"{h:<10.4f} | {dof_counts[-1]:<8d} | {TL_fem_h:<12.5f} | {tl_err:<10.4e} | {l2_err:<10.4e} | {h1_err:<10.4e}")

print("-" * 75)


# %% ── 4. Estimate Convergence Rates ──────────────────────────────────────
# Fit log-log slopes: log(error) = slope * log(h) + constant
slope_L2 = np.polyfit(np.log(h_sizes), np.log(L2_errors), 1)[0]
slope_H1 = np.polyfit(np.log(h_sizes), np.log(H1_errors), 1)[0]
slope_TL = np.polyfit(np.log(h_sizes), np.log(TL_errors), 1)[0]

print("\nEstimated Convergence Rates (Slopes on log-log scale):")
print(f"  L2 Error Rate        : {slope_L2:.3f}   (Theoretical P1: ~2.0)")
print(f"  H1 Full Error Rate   : {slope_H1:.3f}   (Theoretical P1: ~1.0)")
print(f"  TL Scalar Error Rate : {slope_TL:.3f}")


# %% ── 5. Plot and Save log-log Error Graph ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Plot measured errors
ax.loglog(h_sizes, L2_errors, "bo-", linewidth=1.5, markersize=6, label=f"L2 Error (slope={slope_L2:.2f})")
ax.loglog(h_sizes, H1_errors, "rs-", linewidth=1.5, markersize=6, label=f"H1 Error (slope={slope_H1:.2f})")
# ax.loglog(h_sizes, TL_errors, "g^-", linewidth=1.5, markersize=6, label=f"TL Scalar Error (slope={slope_TL:.2f})")

# Plot theoretical references
h_ref_lines = np.array(h_sizes)
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
fig_output = figures_dir / "mesh_convergence.png"
plt.savefig(str(fig_output), dpi=150)
print(f"\nConvergence log-log plot successfully saved to: {fig_output}")
