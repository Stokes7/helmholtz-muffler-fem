# Helmholtz Muffler FEM

This repository contains a modular 2D acoustic simulation tool for an expansion chamber muffler. It solves the complex-valued Helmholtz equation using the **Finite Element Method (FEM)**. The project is built in Python using **FEniCSx/DOLFINx** and **Gmsh**.

---

## 1. Prerequisites & Environment Setup

Acoustic FEM simulations require advanced compiled scientific libraries (such as PETSc parallel linear solvers, MPI bindings, and the OCC geometry kernel). To make setup effortless, the environment must be managed via Conda/Mamba.

> [!IMPORTANT]
> **Prerequisite:** You **must** have a Conda distribution installed on your system before running any commands. 
> * It is highly recommend to install **[Miniforge](https://github.com/conda-forge/miniforge)** (which comes with `mamba` pre-installed and uses the free, community-driven `conda-forge` channel).
> * Alternatively, you can use **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)**.

Choose one of the two options below to configure your environment:

### Clean Automated Installation via Conda/Mamba (Recommended)
This is the most robust, cross-platform method. It automatically installs all compiled dependencies (including FEniCSx and Gmsh) in a clean, isolated virtual environment named `fenicsx-complex`.

1. Open your terminal, clone this repository, and navigate to its root:
   ```bash
   git clone <REPOSITORY_URL>
   cd helmholtz-muffler-fem
   ```
2. Create the virtual environment from the provided `environment.yml` file:
   ```bash
   # If using Miniforge (highly recommended - faster implementation):
   mamba env create -f environment.yml
   
   # If using standard Miniconda/Anaconda:
   conda env create -f environment.yml
   ```
3. Activate the newly created environment:
   ```bash
   conda activate fenicsx-complex
   ```

---

## 2. Codebase Structure

The project is structured modularly to separate geometry modeling, physical solving, and scientific analysis:

* 📂 **`gmsh/`**: Geometry and mesh generation.
  * `geometry.py`: Generates the mesh for a simple expansion chamber.
  * `geometry_extended.py`: Generates the mesh for a muffler with internal inlet/outlet protrusions (extended pipes) using boolean operations.
  * `plot_mesh.py`: Interactive 2D mesh visualizer using Matplotlib.
* 📂 **`src/`**: Core FEM solver.
  * `solver.py`: Contains the `HelmholtzSolver` class. It manages the Lagrange P1 function space, builds the complex weak formulation, applies anechoic Robin boundary conditions at the outlet, and computes the Transmission Loss (TL).
* 📂 **`scripts/`**: Parametric sweeps, convergence, and optimization.
  * `run_verfication_case.py`: A comprehensive frequency sweep (10 - 2000 Hz) validating the 2D FEM solution against 1D plane-wave analytical theory.
  * `run_convergence.py`: A formal mesh convergence study checking $L^2$ and $H^1$ error norms against an ultra-fine reference solution to verify theoretical convergence rates.
  * `run_scenario_study.py`: Runs 1D frequency sweeps for multiple designs, performs a 2D parametric length optimization sweep, and saves 2D pyvista pressure field renders at resonance.
* 📂 **`test/`**: Automated unit tests.
  * `test_solver.py`: Automated pytest suite testing physical solver bounds, geometry creation, and collision detection.
* 📂 **`results/`**: Figures and renders are automatically saved here.

---

## 3. Quick-Start Execution Guide

Once your environment is active (`fenicsx-complex`), you can run each script manually or execute the entire project automatically.

### Option A: Automated Execution Wrapper (HPC & Local - Recommended)
To run all verification tests, convergence studies, and scenario sweeps at once with zero manual configuration, execute the wrapper script in the root directory:
```bash
./run_all.sh
```

**What this wrapper script does:**
1. **Dynamic Environment Resolution:** Automatically detects and resolves the active Conda/Mamba environment or falls back to your local path, ensuring all Python dependencies are loaded correctly.
2. **HPC Compiler Override:** Globally sets `export CC=gcc`, which forces FEniCSx's JIT compiler to use the system `gcc` instead of cluster-loaded Intel compilers (like `icx`).
3. **Sequential Automation:** Runs the verification sweeps, calculates convergence norms, executes the 2D optimization grid, and renders PyVista screenshots.

---

### Option B: Manual Step-by-Step Execution
If you prefer running individual components manually:

1. **Generate the muffler mesh**:
   ```bash
   python gmsh/geometry.py
   ```
2. **Visualize the generated mesh**:
   ```bash
   python gmsh/plot_mesh.py
   ```
3. **Run the physical validation sweep against analytical theory**:
   ```bash
   python scripts/run_verfication_case.py
   ```
4. **Run the mesh convergence study**:
   ```bash
   python scripts/run_convergence.py
   ```
5. **Run the design parameter sweeps and resonance visualizations**:
   ```bash
   python scripts/run_scenario_study.py
   ```

All resulting plots and rendering figures will be saved automatically in the `results/figures/` folder.
