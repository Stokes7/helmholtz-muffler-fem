import os
import matplotlib
matplotlib.use("Agg")  # Run headless without GUI windows
import matplotlib.pyplot as plt
import numpy as np
import sys

# Import our modular geometry and solver components!
sys.path.append(os.path.abspath("Projects/helmholtz-muffler-fem/gmsh"))
sys.path.append(os.path.abspath("Projects/helmholtz-muffler-fem/src"))
from geometry import generate_muffler_mesh
from solver import HelmholtzSolver

# Ensure output directories exist
os.makedirs("Projects/helmholtz-muffler-fem/results/figures", exist_ok=True)

# %% ── 1. Generate Mesh programmatically ──────────────────────────────────
h_default = 0.005
mesh_path = "Projects/helmholtz-muffler-fem/gmsh/mesh/simple_duct.msh"
print(f"Generating mesh with h = {h_default} m...")
domain, cell_tags, facet_tags = generate_muffler_mesh(h_default, output_file=mesh_path)

# %% ── 2. Initialize Solver ───────────────────────────────────────────────
print("Initializing Helmholtz solver...")
solver = HelmholtzSolver(domain, facet_tags)

# %% ── 3. Run Frequency Sweep ─────────────────────────────────────────────
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

# %% ── 4. Generate and Save Comparison Plot ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(freqs, TL_anal, "k--", linewidth=1.8, label="Analytical (Plane Wave Theory)")
ax.plot(freqs, TL_fem,  "r-",  linewidth=1.5, label="FEM 2D (Lagrange P1 elements)")

ax.set_xlabel("Frequency [Hz]", fontsize=11)
ax.set_ylabel("Transmission Loss (TL) [dB]", fontsize=11)
ax.set_title("Transmission Loss Spectrum — Simple Expansion Muffler", fontsize=13, fontweight="bold")
ax.set_xlim(freqs[0], freqs[-1])
ax.set_ylim(-2, 25)
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()
fig_output = "Projects/helmholtz-muffler-fem/results/figures/validation_tl_sweep.png"
plt.savefig(fig_output, dpi=150)
print(f"\nSweep validation plot successfully saved to: {fig_output}")
