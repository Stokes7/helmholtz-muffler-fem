# %% ── 1. Import Libraries ──────────────────────────────────


# matplotlib.use("Agg")  # Run headless without GUI windows
import sys

# Import our modular geometry and solver components!
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd()

sys.path.append(str(project_root / "gmsh"))
sys.path.append(str(project_root / "src"))
from geometry import generate_muffler_mesh

from solver import HelmholtzSolver

# Ensure output directories exist
figures_dir = project_root / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# %% ── 2. Generate Mesh programmatically ──────────────────────────────────
h_default = 0.005
mesh_path = project_root / "gmsh" / "mesh" / "simple_duct.msh"
print(f"Generating mesh with h = {h_default} m...")
domain, cell_tags, facet_tags = generate_muffler_mesh(h_default, output_file=str(mesh_path))

# %% ── 3. Initialize Solver ───────────────────────────────────────────────
print("Initializing Helmholtz solver...")
solver = HelmholtzSolver(domain, facet_tags)

# %% ── 4. Run Frequency Sweep ─────────────────────────────────────────────
freqs = np.arange(10, 2001, 5)  # 10 to 2000 Hz, step 5 Hz
TL_fem = np.zeros(len(freqs))
TL_anal = np.zeros(len(freqs))

print(f"\nStarting sweep over {len(freqs)} frequencies...")
for i, f_i in enumerate(freqs):
    # Solve using the modular HelmholtzSolver class!
    TL_fem[i], _, _, _ = solver.solve(f_i)
    
    # Compute analytical plane-wave theory comparison
    k_val = 2 * np.pi * f_i / solver.c0
    TL_anal[i] = 10 * np.log10(1 + 0.25 * (solver.m - 1/solver.m)**2 * np.sin(k_val * solver.L_ch)**2)
    
    if (i + 1) % 50 == 0 or (i + 1) == len(freqs):
        print(f"  Processed {i+1}/{len(freqs)} frequencies ({f_i} Hz) | TL FEM = {TL_fem[i]:.3f} dB")

print("Sweep completed successfully!")

# %% ── 5. Generate and Save Comparison Plot ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(freqs, TL_anal, "k--", linewidth=1.8, label="Analytical (Plane Wave Theory)")
ax.plot(freqs, TL_fem,  "r-",  linewidth=1.5, label="FEM 2D (Lagrange P1 elements)")

ax.set_xlabel("Frequency [Hz]", fontsize=11)
ax.set_ylabel("Transmission Loss (TL) [dB]", fontsize=11)
ax.set_title("Transmission Loss Spectrum — Simple Expansion Muffler", fontsize=13, fontweight="bold")
ax.set_xlim(freqs[0], freqs[-1])
ax.set_ylim(-2, 5)
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()
fig_output = figures_dir / "validation_tl_sweep.png"
plt.savefig(str(fig_output), dpi=150)
print(f"\nSweep validation plot successfully saved to: {fig_output}")

# %% ─── 6. Plot pressure field at 500 Hz ────────────────────────────────────────────────────────
import pyvista as pv
from dolfinx import plot

solver.solve(500.0)

topology, cell_types, geometry = plot.vtk_mesh(solver.V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["|p|"] = np.abs(solver.p_h.x.array)

plotter = pv.Plotter(window_size=(500, 400), off_screen=True)
plotter.add_mesh(grid.copy(), scalars="|p|", cmap="viridis", show_edges=False)
plotter.add_text("|p| at 500 Hz", font_size=10)
plotter.view_xy()

pressure_output = figures_dir / "pressure_field_500Hz.png"
plotter.screenshot(str(pressure_output))
print(f"Pressure field saved to: {pressure_output}")

# %%
