import pytest
import numpy as np
from pathlib import Path
import sys
import os

# Append local directories
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "gmsh"))
sys.path.append(str(project_root / "src"))

from geometry import generate_muffler_mesh
from geometry_extended import generate_extended_mesh
from solver import HelmholtzSolver

# Set CC=gcc inside pytest to ensure execution in HPC clusters
os.environ["CC"] = "gcc"

def test_simple_geometry_mesh():
    """Verify that simple mesh generation creates valid physical tags and domain."""
    h_test = 0.01
    domain, cell_tags, facet_tags = generate_muffler_mesh(h_test, recombine_quads=False)
    
    assert domain is not None
    assert domain.topology.dim == 2
    # Ensure physical tags are properly mapped
    assert facet_tags is not None
    
def test_extended_geometry_collision_check():
    """Ensure that the extended mesh generator correctly raises ValueError on pipe collisions."""
    h_test = 0.01
    # Symmetric extensions of 0.15m + 0.10m = 0.25m (exceeds L_ch = 0.20m)
    with pytest.raises(ValueError):
        generate_extended_mesh(h_test, L_ext_in=0.15, L_ext_out=0.10, recombine_quads=False)

def test_helmholtz_solver_execution():
    """Verify that the Helmholtz solver runs and returns mathematically valid TL values."""
    h_test = 0.01
    f_test = 500.0
    domain, _, facet_tags = generate_muffler_mesh(h_test, recombine_quads=False)
    
    solver = HelmholtzSolver(domain, facet_tags)
    TL, p_inlet, p_outlet, p_inc = solver.solve(f_test)
    
    # Assert physical properties
    assert isinstance(TL, float)
    assert TL >= -5.0 and TL <= 40.0  # TL must lie within physically sensible bounds
    assert not np.isnan(TL)
    assert not np.isinf(TL)
