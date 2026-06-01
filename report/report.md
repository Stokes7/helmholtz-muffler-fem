# Finite Element Simulation and Design Optimization of a 2D Helmholtz Acoustic Muffler

**Course:** Modern Simulation Software Development (MSSD)  
**Author:** zp252136  
**Institution:** RWTH Aachen University  
**Date:** May 2026  

---

## Abstract

---

## 1. Introduction and Physical Setup

Exhaust noise from internal combustion engines is widely recognized as a major environmental pollutant in urban areas. To mitigate this issue, expansion chamber mufflers are extensively employed in automotive exhaust systems to attenuate high-level sound waves [1]. The acoustic attenuation performance of these devices is standardly evaluated using a metric known as Transmission Loss (TL) [2]. While 1D analytical plane-wave models provide exact solutions for evaluating the TL of simple expansion chambers (SEC), they are inherently limited when dealing with complex internal structures. Consequently, numerical techniques such as the Finite Element Method (FEM) have become essential tools for predicting and optimizing muffler performance.

This report evaluates two muffler configurations using a 2D FEM simulation implemented in FEniCSx, both analyzed over a swept frequency range to obtain their respective TL spectra.

The first case, shown in Figure 1, corresponds to a Simple Expansion Chamber (SEC), a straight duct that widens into a larger volume before narrowing back to the outlet. At the inlet boundary $\Gamma_\text{in}$, a uniform acoustic velocity is imposed, representing a plane wave entering the domain. At the outlet boundary $\Gamma_\text{out}$, a non-reflecting radiation condition is applied to avoid spurious reflections. The top and bottom wall sections, $\Gamma_\text{wall}$, are treated as rigid boundaries.

<figure style="text-align: center;">
  <img src="assets/simple_duct_diagram.png" alt="Simple expansion chamber">
  <figcaption>Figure 1: Geometry and boundary conditions of the simple expansion chamber (SEC).</figcaption>
</figure>

The second case, shown in Figure 2, introduces an extended-tube chamber configuration, where the inlet and outlet pipes are extended into the interior of the chamber. The boundary conditions remain identical to the SEC case, imposed velocity at $\Gamma_\text{in}$ and a radiation condition at $\Gamma_\text{out}$, but the internal geometry changes significantly. The protruding pipes with lengths $L_{ext, in}$ and $L_{ext, out}$ act as acoustic resonators that interfere with the standing wave pattern inside the chamber, filling in the frequency gaps where the SEC provides no attenuation and resulting in a broader and more consistent noise reduction across the frequency range [3].

<figure style="text-align: center;">
  <img src="assets/extended_duct_diagram.png" alt="Extended duct muffler">
  <figcaption>Figure 2: Geometry and boundary conditions of the extended-tube chamber configuration.</figcaption>
</figure>

The SEC is first verified against the 1D analytical plane-wave solution over the 10–2000 Hz range, and a mesh convergence study is performed to confirm numerical accuracy. The extended-tube configuration is then used as the practical scenario, where a parametric sweep over the protrusion lengths $L_\text{ext,in}$ and $L_\text{ext,out}$ is carried out to identify designs that maximize attenuation at a target frequency.

---

## 2. Mathematical Formulation & Weak Form

### 2.1 The Helmholtz Equation

Acoustic wave propagation in a compressible, inviscid fluid is governed by the wave equation for the acoustic pressure $p(\mathbf{x}, t)$. Assuming time-harmonic behavior of the form $p(\mathbf{x}, t) = \text{Re}[\hat{p}(\mathbf{x})\, e^{-i\omega t}]$, where $\omega = 2\pi f$ is the angular frequency, the spatial amplitude $\hat{p}(\mathbf{x})$ satisfies the Helmholtz equation:

$$\nabla^2 \hat{p} + k^2 \hat{p} = 0 \quad \text{in } \Omega \subset \mathbb{R}^2,$$

where $k = \omega / c_0$ is the acoustic wavenumber. The domain $\Omega$ represents the 2D cross-section of the muffler cavity. The fluid parameters used throughout this work are summarized in the following table:

<figure style="text-align: center;">

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
| Air density | $\rho_0$ | 1.21 | kg/m³ |
| Speed of sound | $c_0$ | 343 | m/s |
| Characteristic impedance | $Z = \rho_0 c_0$ | 415.03 | Pa·s/m |
| Inlet normal velocity | $v_n$ | $10^{-3}$ | m/s |

<figcaption>Table 1: Acoustic fluid parameters used in the simulation.</figcaption>
</figure>

For clarity, the hat notation is dropped hereafter and $p$ refers to the complex pressure amplitude.

### 2.2 Boundary Conditions

The boundary of the domain decomposes into three disjoint parts:

$$\partial \Omega = \Gamma_\text{in} \cup \Gamma_\text{out} \cup \Gamma_\text{wall},$$

which are described as follows:

**Inlet $\Gamma_\text{in}$: prescribed normal velocity (Neumann):** A uniform inward acoustic velocity $v_n$ is imposed at the inlet, representing an incoming plane wave. From the linearized momentum equation, this translates to a Neumann condition on the pressure gradient:

$$\nabla p \cdot \mathbf{n} = -i\omega\rho_0 v_n \quad \text{on } \Gamma_\text{in}.$$

**Outlet $\Gamma_\text{out}$: anechoic radiation condition (Robin):** To prevent spurious reflections at the outlet, a first-order absorbing boundary condition is applied. It enforces that the wave exits the domain as if it were propagating into an infinite anechoic duct:

$$\nabla p \cdot \mathbf{n} = \frac{i\omega\rho_0}{Z}\, p \quad \text{on } \Gamma_\text{out}.$$

**Walls $\Gamma_\text{wall}$: rigid wall (homogeneous Neumann):** The chamber walls are modeled as acoustically rigid, meaning no normal particle velocity, which corresponds to a zero normal pressure gradient:

$$\nabla p \cdot \mathbf{n} = 0 \quad \text{on } \Gamma_\text{wall}.$$

### 2.3 Weak Form Derivation

To obtain the weak form, the Helmholtz equation is multiplied by a test function $v \in V$ and integrated over $\Omega$:

$$\int_\Omega \left(\nabla^2 p + k^2 p\right) v \, dx = 0.$$

Applying Green's first identity to the Laplacian term:

$$\int_\Omega \nabla^2 p \cdot v \, dx = -\int_\Omega \nabla p \cdot \nabla v \, dx + \int_{\partial\Omega} (\nabla p \cdot \mathbf{n})\, v \, ds.$$

The boundary integral is split over the three boundary segments and the respective conditions are substituted:

$$\int_{\partial\Omega} (\nabla p \cdot \mathbf{n})\, v \, ds = \underbrace{-i\omega\rho_0 v_n \int_{\Gamma_\text{in}} v \, ds}_{\text{inlet}} + \underbrace{\frac{i\omega\rho_0}{Z}\int_{\Gamma_\text{out}} p\, v \, ds}_{\text{outlet (Robin)}} + \underbrace{0}_{\text{walls}}.$$

Collecting all terms, the weak form reads: find $p \in V$ such that

$$a(p, v) = L(v) \quad \forall\, v \in V,$$

where the bilinear and linear forms are:

$$a(p, v) = \int_\Omega \nabla p \cdot \nabla v \, dx - k^2 \int_\Omega p\, v \, dx - \frac{i\omega\rho_0}{Z} \int_{\Gamma_\text{out}} p\, v \, ds,$$

$$L(v) = i\omega\rho_0 v_n \int_{\Gamma_\text{in}} \bar{v} \, ds.$$

<!-- The Robin term on $\Gamma_\text{out}$ appears naturally in the bilinear form without requiring any special treatment: it is a natural boundary condition. -->

### 2.4 Finite Element Discretization

The function space $V$ is approximated by the finite-dimensional subspace $V_h \subset V$ of continuous, piecewise-linear polynomials over the mesh $\mathcal{T}_h$:

$$V_h = \{ v_h \in C^0(\Omega) \mid v_h|_K \in \mathcal{P}_1(K),\ \forall K \in \mathcal{T}_h \},$$

corresponding to Lagrange $\mathcal{P}_1$ elements. This choice is sufficient for the smooth acoustic fields expected at the frequency range studied (10–2000 Hz), and yields theoretical convergence rates of $\mathcal{O}(h^2)$ in the $L^2$ norm and $\mathcal{O}(h)$ in the $H^1$ seminorm for smooth solutions.

Since the pressure field is complex-valued, the computation uses a complex-valued PETSc/DOLFINx assembly with the MUMPS sparse direct solver.

The discrete system $\mathbf{A}\mathbf{p} = \mathbf{b}$ is assembled and solved at each frequency independently, updating only the constants $\omega$ and $k$ without recompiling the forms.

**Transmission Loss.** The acoustic performance is quantified by the Transmission Loss (TL), defined as the ratio of the incident pressure to the transmitted pressure at the outlet:

$$\text{TL} = 20 \log_{10} \frac{|p_\text{inc}|}{|\bar{p}_\text{out}|} \quad [\text{dB}],$$

where $\bar{p}_\text{out}$ is the spatially averaged pressure over $\Gamma_\text{out}$, and the incident pressure $p_\text{inc}$ is recovered from the inlet via plane-wave decomposition:

$$p_\text{inc} = \frac{\bar{p}_\text{in} - \rho_0 c_0 v_n}{2}.$$

---

## 3. Modular Implementation and Solver Choices

The project is organized into four directories. The `gmsh/` folder handles geometry and mesh generation, `src/` contains the FEM solver, `scripts/` holds the parametric studies, and `test/` contains the automated tests. This separation makes it straightforward to swap geometries without touching the solver, or to run individual studies without executing the full pipeline.

**Mesh generation.** Both geometries (the simple chamber and the extended-tube version) are built in a reproducible way using the Gmsh Python API with the OpenCASCADE (OCC) kernel. Rectangular regions are fused or subtracted via boolean operations, which keeps the geometry definition compact and fully reproducible from a single function call. Boundary subsets are assigned integer tags (1: inlet, 2: outlet, 3: walls) directly in Gmsh and carried through to DOLFINx via `gmshio.model_to_mesh`, so the solver always applies boundary conditions by tag rather than by coordinate lookup.

**Solver design.** The `HelmholtzSolver` class in `src/solver.py` sets up the weak form once at construction time. The frequency-dependent quantities $\omega$ and $k$ are declared as `fem.Constant` objects, so solving at a new frequency only requires updating two numbers: the UFL forms and the sparse matrix sparsity pattern are never recompiled. This makes the frequency sweep across hundreds of points practical without significant overhead.

For the linear system, the MUMPS sparse direct solver is used via PETSc (`pc_type: lu`, `pc_factor_mat_solver_type: mumps`). For the mesh sizes used here (a few thousand DOFs), a direct solver is more reliable and faster than an iterative one, since there is no need to tune preconditioners or worry about convergence for a complex-valued indefinite system.

**Automated testing.** Three pytest tests in `test/test_solver.py` verify some basic test cases: that the simple geometry produces a valid mesh with the correct tags, that the collision check in the extended geometry raises an error when the protrusions exceed the chamber length, and that the solver returns a physically plausible TL value (between −5 and 40 dB) at a test frequency. These checks are intended to be lightweight, rather than complete numerical verification, since this is already covered in Section 4.

**Reproducibility.** The full Conda environment is pinned in `environment.yml`, including the complex-valued PETSc build required by DOLFINx. A single shell script `run_all.sh` runs the tests and all three studies in sequence, and handles the `CC=gcc` override needed on HPC clusters where Intel compilers are loaded by default. A GitHub Actions pipeline runs this script on every push to `main` and deploys the compiled report to GitHub Pages.

## 4. Exercise 2: Verification Sweep

---

## 5. Exercise 3: Mesh Convergence Study

### 5.1 Convergence Summary Table

### 5.2 Estimated Convergence Rates

---

## 6. Exercise 4: Practical Scenario (Extended Protruding Ducts)

### 6.1 Design Concept & Acoustic Resonance Tuning

### 6.2 2D Parametric Design Sweep & Optimization

### 6.3 Visualizing the Resonance Pressure Field

---

## 7. Conclusions & Recommendations

---

## 8. Declaration of Academic Honesty

---

## 9. References

[1] Ezzeddin, M. M., & Jolgaf, M. (2017). *Acoustic Analysis of a Perforated-pipe Muffler Using ANSYS*. University Bulletin-ISSUE, 19(4).

[2] Wagner, N., & Helfrich, R. (2008). *Computation of the transmission loss of acoustic resonators*. Aeroacoustics and Flow Noise, 535-548.

[3] Munjal, M. L. (2014). Acoustics of ducts and mufflers (2nd ed.). Wiley.

