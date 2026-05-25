# %% ── 1. Import Libraries ──────────────────────────────────
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem


# %% ── 2. Helmholtz Solver Class ─────────────────────────────────────────
class HelmholtzSolver:
    def __init__(self, domain, facet_tags, rho0=1.21, c0=343.0, vn=1e-3):
        """
        Encapsulates the 2D acoustic Helmholtz finite element solver in DOLFINx.
        """
        self.domain = domain
        self.facet_tags = facet_tags
        self.rho0 = rho0
        self.c0 = c0
        self.vn = vn
        self.Z = rho0 * c0
        
        # Duct geometry (must match geometry.py)
        self.L_ch = 0.20    # chamber length     [m]
        self.r    = 0.025   # pipe half-height   [m]
        self.R    = 0.05    # chamber half-height [m]
        self.m    = self.R / self.r   # area ratio for analytical TL
        
        # Lagrange P1 function space
        self.V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Measures and Functions
        self.dx = ufl.Measure("dx", domain=domain)
        self.ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        
        self.p = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        
        # Declare Constants for dynamic updates
        self.omega = fem.Constant(domain, complex(0.0))
        self.k     = fem.Constant(domain, complex(0.0))
        
        # Bilinear form a (anechoic Robin BC at outlet Gamma_outlet)
        self.a = (
              ufl.inner(ufl.grad(self.p), ufl.grad(self.v)) * self.dx
            - self.k**2 * ufl.inner(self.p, self.v) * self.dx
            + 1j * self.omega * self.rho0 / self.Z * ufl.inner(self.p, self.v) * self.ds(2)
        )
        
        # Linear form L (inward normal velocity boundary at inlet Gamma_inlet)
        self.L = -1j * self.omega * self.rho0 * self.vn * ufl.conj(self.v) * self.ds(1)
        
        # Solution function
        self.p_h = fem.Function(self.V)
        self.p_h.name = "pressure"
        
        # Linear Problem using MUMPS sparse direct solver
        self.problem = LinearProblem(
            self.a, self.L, u=self.p_h,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
            petsc_options_prefix="helmholtz",
        )
        
        # Precompute boundary lengths
        one = fem.Constant(domain, complex(1.0))
        self.inlet_length  = fem.assemble_scalar(fem.form(one * self.ds(1)))
        self.outlet_length = fem.assemble_scalar(fem.form(one * self.ds(2)))
        
        # Precompile average pressure integration forms
        self._p_inlet_form  = fem.form(self.p_h * self.ds(1))
        self._p_outlet_form = fem.form(self.p_h * self.ds(2))
        
    def solve(self, f_val):
        """
        Solves the Helmholtz equation at frequency f_val [Hz].
        Returns (TL_fem, p_inlet_avg, p_outlet_avg, p_inc).
        """
        omega_val = 2 * np.pi * f_val
        k_val     = omega_val / self.c0
        
        # Update Constant parameters dynamically (no recompilation)
        self.omega.value = complex(omega_val)
        self.k.value     = complex(k_val)
        
        # Solve linear system
        self.problem.solve()
        
        # Evaluate pressures
        p_inlet_avg  = fem.assemble_scalar(self._p_inlet_form)  / self.inlet_length
        p_outlet_avg = fem.assemble_scalar(self._p_outlet_form) / self.outlet_length
        p_inc        = (p_inlet_avg - self.rho0 * self.c0 * self.vn) / 2.0
        
        # Compute Transmission Loss
        TL_fem = 20 * np.log10(np.abs(p_inc) / np.abs(p_outlet_avg))
        return TL_fem, p_inlet_avg, p_outlet_avg, p_inc
        
    def save_visualization(self, f_val, output_path):
        """
        Generates and saves a PyVista rendering of the pressure fields offscreen.
        """
        import pyvista as pv
        from dolfinx import plot
        
        topology, cell_types, geometry = plot.vtk_mesh(self.V)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid["Re(p)"] = self.p_h.x.array.real
        grid["Im(p)"] = self.p_h.x.array.imag
        grid["|p|"]   = np.abs(self.p_h.x.array)
        
        plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 400), off_screen=True)
        
        plotter.subplot(0, 0)
        plotter.add_mesh(grid.copy(), scalars="Re(p)", cmap="RdBu_r", show_edges=False)
        plotter.add_text(f"Re(p)  {f_val:.0f} Hz", font_size=10)
        plotter.view_xy()
        
        plotter.subplot(0, 1)
        plotter.add_mesh(grid.copy(), scalars="|p|", cmap="viridis", show_edges=False)
        plotter.add_text(f"|p|  {f_val:.0f} Hz", font_size=10)
        plotter.view_xy()
        
        plotter.subplot(0, 2)
        plotter.add_mesh(grid.copy(), scalars="Im(p)", cmap="RdBu_r", show_edges=False)
        plotter.add_text(f"Im(p)  {f_val:.0f} Hz", font_size=10)
        plotter.view_xy()
        
        plotter.screenshot(output_path)


if __name__ == "__main__":
    # Standard script execution
    from dolfinx.io import gmsh as dolfinx_gmsh
    from mpi4py import MPI

    import gmsh
    
    if gmsh.isInitialized():
        gmsh.finalize()
        
    gmsh.initialize()
    gmsh.open("Projects/helmholtz-muffler-fem/gmsh/mesh/simple_duct.msh")
    domain, _, facet_tags, _, _, _ = dolfinx_gmsh.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )
    gmsh.finalize()
    
    print(f"Mesh loaded: {domain.topology.dim}D")
    print(f"Number of cells   : {domain.topology.index_map(2).size_local}")
    print(f"Number of vertices: {domain.topology.index_map(0).size_local}")
    
    # Default parameters
    freqs = np.array([500.0])
    
    solver = HelmholtzSolver(domain, facet_tags)
    print(f"\nSolving for {len(freqs)} frequency values...")
    
    TL_fem  = np.zeros(len(freqs))
    TL_anal = np.zeros(len(freqs))
    
    for i, f_i in enumerate(freqs):
        TL_fem[i], p_inlet_avg, p_outlet_avg, p_inc = solver.solve(f_i)
        
        # Analytical comparison
        k_val = 2 * np.pi * f_i / solver.c0
        TL_anal[i] = 10 * np.log10(1 + 0.25 * (solver.m - 1/solver.m)**2 * np.sin(k_val * solver.L_ch)**2)
        
    print("\nSolved successfully!")
    print(f"  Inlet avg p  = {p_inlet_avg:.4e} Pa")
    print(f"  Outlet avg p = {p_outlet_avg:.4e} Pa")
    print(f"  Incident p   = {p_inc:.4e} Pa")
    print(f"  TL FEM       = {TL_fem[0]:.4f} dB")
    print(f"  TL analytical= {TL_anal[0]:.4f} dB")
    print(f"  Error        = {abs(TL_fem[0] - TL_anal[0]):.4f} dB")
    
    os_img = "Projects/helmholtz-muffler-fem/results/pressure_field.png"
    solver.save_visualization(freqs[0], os_img)
    print(f"Pressure field visualization saved to: {os_img}")
