# %% ── 1. Import Libraries ──────────────────────────────────
import sys

import matplotlib

matplotlib.use("Agg")  # Run headless without GUI windows
# Append paths to search local modules
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
from geometry_extended import generate_extended_mesh

from solver import HelmholtzSolver

# Ensure output directories exist
figures_dir = project_root / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

h_size = 0.005  # Mesh size for highly resolved sweep
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

# c0 = 343.0
# r_pipe    = 0.025   # pipe half-height [m]
# r_chamber = 0.05    # chamber half-height [m]
# L_ext = cases["Symmetric Ext Both"]["L_in"]
#
# # Geometric quarter-wave resonance (no end correction)
# f_qw = c0 / (4 * L_ext)
#
# # End-length correction (Ingard, flanged tube inside concentric chamber)
# delta = 0.6 * r_pipe * (1 - 1.25 * r_pipe / r_chamber)
# f_qw_corr = c0 / (4 * (L_ext + delta))
#
# ax.axvline(f_qw_corr, color="black", linestyle=":",  alpha=0.8,
#            label=f"QW corrected  $c_0/(4(L+\\delta))$ = {f_qw_corr:.0f} Hz  ($\\delta$={delta*1000:.1f} mm)")

ax.set_xlabel("Frequency [Hz]", fontsize=11)
ax.set_ylabel("Transmission Loss (TL) [dB]", fontsize=11)
# ax.set_title("Transmission Loss Spectrum — Parametric Design Comparison", fontsize=13, fontweight="bold")
ax.set_xlim(freqs[0], freqs[-1])
ax.set_ylim(-2, 70)
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(loc="upper left", frameon=True, fontsize=10)

fig_output = figures_dir / "extended_tl_comparison.png"
plt.tight_layout()
plt.savefig(str(fig_output), dpi=150)
print(f"\n[SUCCESS] Spectrum comparison plot saved to: {fig_output}")

# Peak TL frequency of the symmetric extended case
sym_tl = tl_results["Symmetric Ext Both"]
f_peak = freqs[np.argmax(sym_tl)]
print(f"Peak TL of symmetric case: {sym_tl.max():.3f} dB at {f_peak:.0f} Hz")


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
# ax2.set_title(f"Acoustic Optimization Heatmap at {f_opt:.0f} Hz", fontsize=13, fontweight="bold")
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

import matplotlib.image as mpimg
import pyvista as pv
from dolfinx import plot

# Setup symmetric extended mesh at peak TL frequency
print(f"Solving symmetric extended case at {f_peak:.0f} Hz (peak TL) for pressure visualization...")
domain_vis, _, facet_tags_vis = generate_extended_mesh(h_size, 0.05, 0.05)
solver_vis = HelmholtzSolver(domain_vis, facet_tags_vis)
solver_vis.solve(f_peak)

topology, cell_types, geometry = plot.vtk_mesh(solver_vis.V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
p_arr = solver_vis.p_h.x.array
grid.point_data["|p|"] = np.abs(p_arr)

pv.set_plot_theme("document")

scalar_bar_args = dict(
    title_font_size=14, label_font_size=11,
    shadow=True, n_labels=5,
    fmt="%.2f", vertical=False,
    position_x=0.2, position_y=0.02, width=0.6, height=0.08,
)

plotter = pv.Plotter(window_size=(800, 300), off_screen=True)
plotter.set_background("white")
plotter.add_mesh(grid.copy(), scalars="|p|", cmap="plasma",
                 show_edges=False, scalar_bar_args={**scalar_bar_args, "title": "|p| [Pa]"})
plotter.add_mesh(grid.copy(), style="wireframe", color="white", line_width=0.2, opacity=0.1)
plotter.view_xy()
plotter.camera.zoom(2.7)

screenshot_path = figures_dir / "extended_pressure_field.png"
plotter.screenshot(str(screenshot_path))
print(f"[SUCCESS] Resonance pressure field saved to: {screenshot_path}")

img = mpimg.imread(str(screenshot_path))
fig, ax = plt.subplots(figsize=(12, 2.2))
ax.imshow(img)
ax.axis("off")
plt.tight_layout()
plt.show()

print("\nAll tasks in Exercise 4 completed successfully!")



# %%
