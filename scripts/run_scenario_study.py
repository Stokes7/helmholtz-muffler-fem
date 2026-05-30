# %% ── 1. Import Libraries ──────────────────────────────────
import os
import sys

import matplotlib

matplotlib.use("Agg")  # Run headless without GUI windows
import matplotlib.pyplot as plt
import numpy as np

# Append paths to search local modules
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

sys.path.append(str(project_root / "gmsh"))
sys.path.append(str(project_root / "src"))
from geometry import generate_muffler_mesh
from geometry_extended import generate_extended_mesh

from solver import HelmholtzSolver

# Ensure output directories exist
figures_dir = project_root / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

h_size = 0.005  # Standard mesh size for fast, highly resolved sweep
f_opt = 1200.0  # Targeted optimization frequency (Hz)

# %% ── 2. Run Comparative Frequency Sweeps (1D) ────────────
print("=================================================================")
print("STAGE 1: Running comparative frequency sweeps (10 - 2000 Hz)...")
print("=================================================================")

freqs = np.arange(10, 2001, 5)  # 10 to 2000 Hz, step 5 Hz
cases = {
    "Simple Muffler": {"L_in": 0.00, "L_out": 0.00, "color": "red", "style": "-"},
    "Inlet Ext Only": {"L_in": 0.05, "L_out": 0.00, "color": "orange", "style": "--"},
    "Outlet Ext Only": {"L_in": 0.00, "L_out": 0.05, "color": "green", "style": "-."},
    "Symmetric Ext Both": {"L_in": 0.05, "L_out": 0.05, "color": "blue", "style": "-"}
}

tl_results = {}

for name, params in cases.items():
    print(f"\nProcessing case: {name} (L_ext_in = {params['L_in']}m, L_ext_out = {params['L_out']}m)...")
    
    # 1. Generate appropriate mesh
    if params["L_in"] == 0.00 and params["L_out"] == 0.00:
        domain, _, facet_tags = generate_muffler_mesh(h_size)
    else:
        domain, _, facet_tags = generate_extended_mesh(h_size, params["L_in"], params["L_out"])
        
    # 2. Solve Helmholtz system
    solver = HelmholtzSolver(domain, facet_tags)
    TL_curve = np.zeros(len(freqs))
    
    for i, f_i in enumerate(freqs):
        TL_curve[i] = solver.solve(f_i)[0]
        
    tl_results[name] = TL_curve
    print(f"-> Sweep for '{name}' completed successfully.")

# Plot and save 1D TL Spectrum comparison
fig, ax = plt.subplots(figsize=(12, 6))
for name, params in cases.items():
    ax.plot(freqs, tl_results[name], color=params["color"], linestyle=params["style"], 
            linewidth=1.5, label=name)

# Theoretical quarter-wave resonance at 1715 Hz for a 0.05m protrusion
ax.axvline(1319, color="black", linestyle=":", alpha=0.6, 
           label="Theoretical Quarter-Wave Peak (1715 Hz)")

ax.set_xlabel("Frequency [Hz]", fontsize=11)
ax.set_ylabel("Transmission Loss (TL) [dB]", fontsize=11)
ax.set_title("Transmission Loss Spectrum — Parametric Design Comparison", fontsize=13, fontweight="bold")
ax.set_xlim(freqs[0], freqs[-1])
ax.set_ylim(-2, 70)
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

fig_output = figures_dir / "extended_tl_comparison.png"
plt.tight_layout()
plt.savefig(str(fig_output), dpi=150)
print(f"\n[SUCCESS] Spectrum comparison plot saved to: {fig_output}")


# %% ── 3. Run 2D Protrusion Length Optimization Sweep ──────────
print("\n=================================================================")
print(f"STAGE 2: Running 2D parameter sweep at {f_opt} Hz...")
print("=================================================================")

# 20x20 grid from 0.005 m up to 0.08 m (safe space within L_ch = 0.20 m)
L_vals = np.linspace(0.005, 0.08, 10)
L_in_grid, L_out_grid = np.meshgrid(L_vals, L_vals)
TL_grid = np.zeros_like(L_in_grid)

for i in range(len(L_vals)):
    for j in range(len(L_vals)):
        l_in = L_in_grid[i, j]
        l_out = L_out_grid[i, j]
        
        # Generate mesh for this specific length combination
        domain, _, facet_tags = generate_extended_mesh(h_size, l_in, l_out)
        solver = HelmholtzSolver(domain, facet_tags)
        
        # Calculate TL at f_opt Hz
        TL_grid[i, j] = solver.solve(f_opt)[0]
        
        print(f"Solved L_ext_in = {l_in:.3f} m | L_ext_out = {l_out:.3f} m | TL = {TL_grid[i,j]:.3f} dB")

# Plot and save 2D Contour Optimization Heatmap
fig2, ax2 = plt.subplots(figsize=(8, 6))
cp = ax2.contourf(L_in_grid, L_out_grid, TL_grid, levels=20, cmap="viridis")
fig2.colorbar(cp, label="Transmission Loss (TL) [dB]")

# Mark the maximum TL point (Optimal Design)
max_idx = np.unravel_index(np.argmax(TL_grid), TL_grid.shape)
opt_in = L_in_grid[max_idx]
opt_out = L_out_grid[max_idx]
opt_tl = TL_grid[max_idx]

ax2.plot(opt_in, opt_out, "ro", markersize=8, label=f"Optimal Design ({opt_in:.3f}m, {opt_out:.3f}m)")

ax2.set_xlabel("Inlet Protrusion Length $L_{ext\\_in}$ [m]", fontsize=11)
ax2.set_ylabel("Outlet Protrusion Length $L_{ext\\_out}$ [m]", fontsize=11)
ax2.set_title(f"Acoustic Optimization Heatmap at {f_opt:.0f} Hz", fontsize=13, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.legend(loc="upper right", frameon=True)

opt_fig_output = figures_dir / "length_optimization_2d.png"
plt.tight_layout()
plt.savefig(str(opt_fig_output), dpi=150)
print(f"[SUCCESS] 2D Optimization Heatmap saved to: {opt_fig_output}")
print(f"Optimal Design Found: L_ext_in = {opt_in:.3f} m, L_ext_out = {opt_out:.3f} m yielding TL = {opt_tl:.3f} dB!")


# %% ── 4. Generate 2D PyVista Pressure Field at Resonance ─────
print("\n=================================================================")
print("STAGE 3: Rendering pressure field at resonance...")
print("=================================================================")

import pyvista as pv

# Setup symmetric extended mesh at 1715 Hz
print("Solving symmetric extended case at 1715 Hz for pressure visualization...")
domain_vis, _, facet_tags_vis = generate_extended_mesh(h_size, 0.05, 0.05)
solver_vis = HelmholtzSolver(domain_vis, facet_tags_vis)
solver_vis.solve(1715.0)

# Extract mesh for PyVista
from dolfinx import plot

topology, cell_types, geometry = plot.vtk_mesh(solver_vis.V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# Add complex pressure values as real/imag/mag scalars
p_arr = solver_vis.p_h.x.array
grid.point_data["Re(p)"] = p_arr.real
grid.point_data["|p|"]   = np.abs(p_arr)
grid.point_data["Im(p)"] = p_arr.imag

# Setup Plotter
plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 400), off_screen=True)

plotter.subplot(0, 0)
plotter.add_mesh(grid.copy(), scalars="Re(p)", cmap="RdBu_r", show_edges=False)
plotter.add_text("Re(p) at 1715 Hz", font_size=10)
plotter.view_xy()

plotter.subplot(0, 1)
plotter.add_mesh(grid.copy(), scalars="|p|", cmap="viridis", show_edges=False)
plotter.add_text("|p| at 1715 Hz", font_size=10)
plotter.view_xy()

plotter.subplot(0, 2)
plotter.add_mesh(grid.copy(), scalars="Im(p)", cmap="RdBu_r", show_edges=False)
plotter.add_text("Im(p) at 1715 Hz", font_size=10)
plotter.view_xy()

screenshot_path = figures_dir / "extended_pressure_field.png"
plotter.screenshot(str(screenshot_path))
print(f"[SUCCESS] Resonance pressure field saved to: {screenshot_path}")
print("\nAll tasks in Exercise 4 completed successfully!")
