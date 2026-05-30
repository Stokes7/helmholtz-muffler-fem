#!/bin/bash
# Force exit if any command fails
set -e

echo "============================================================"
echo "      HELMHOLTZ MUFFLER FEM PROJECT WRAPPER RUNNER"
echo "============================================================"

# Force standard GCC to bypass cluster Intel icx compiler linker errors
export CC=gcc

# Get the script directory to run from anywhere
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Resolve the python interpreter dynamically (favoring the active Conda/Mamba env)
PYTHON_EXEC="python3"
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
elif [ -f "$HOME/.miniforge3/envs/fenicsx-complex/bin/python" ]; then
    PYTHON_EXEC="$HOME/.miniforge3/envs/fenicsx-complex/bin/python"
elif [ -f "$HOME/miniforge3/envs/fenicsx-complex/bin/python" ]; then
    PYTHON_EXEC="$HOME/miniforge3/envs/fenicsx-complex/bin/python"
fi

echo "Using Python interpreter: $PYTHON_EXEC"

# 1. Run verification frequency sweep
echo -e "\n--> Running Exercise 2 Verification Sweep..."
"$PYTHON_EXEC" test/test_verification.py

# 2. Run mesh convergence study
echo -e "\n--> Running Exercise 3 Mesh Convergence Study..."
"$PYTHON_EXEC" scripts/run_convergence.py

# 3. Run practical scenario study
echo -e "\n--> Running Exercise 4 Extended Muffler Parametric Study..."
"$PYTHON_EXEC" scripts/run_scenario_study.py

echo -e "\n============================================================"
echo " [SUCCESS] All exercises executed successfully!"
echo " Results and figures saved in 'results/figures/'"
echo "============================================================"
