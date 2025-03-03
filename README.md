
# ME_700_Assignment_2: 3D Frame Solver
![GitHub Actions](https://github.com/DVinals4721/ME_700_Assignment_2/actions/workflows/test.yml/badge.svg)
[![codecov](https://codecov.io/gh/DVinals4721/ME_700_Assignment_2/branch/main/graph/badge.svg)](https://codecov.io/gh/DVinals4721/ME_700_Assignment_2)
![GitHub issues](https://img.shields.io/github/issues/DVinals4721/ME_700_Assignment_2)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/DVinals4721/ME_700_Assignment_2)

This package provides an implementation of a 3D Frame Solver using the Direct Stiffness Method, including:

- 3D beam element formulation
- Geometric nonlinearity consideration
- Local and global stiffness matrix assembly
- Boundary condition application
- Solver for displacements and reactions

## Installation and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/DVinals4721/ME_700_Assignment_2.git
   cd ME_700_Assignment_2
   ```

2. Set up a Conda environment:

   ```bash
   conda create --name frame-solver-env python=3.11
   conda activate frame-solver-env
   ```

   Note: You can also use mamba if you prefer.

3. Verify Python version:

   ```bash
   python --version
   ```

   Ensure it shows version 3.11 or later.

4. Update pip and essential tools:

   ```bash
   pip install --upgrade pip setuptools wheel
   ```

5. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

   Make sure you're in the correct directory (ME_700_Assignment_2) when running this command.

6. Install pytest and pytest-cov for testing:

   ```bash
   pip install pytest pytest-cov
   ```

7. Run tests with coverage:

   ```bash
   pytest -v --cov=frame_solver --cov-report term-missing
   ```

8. Run specific tests:

   ```bash
   pytest tests/test_frame_solver.py
   ```

## Usage Example

After installation, explore the functionality through our example script:

### 3D Frame Analysis

```bash
python examples/frame_analysis_example.py
```

This script demonstrates how to set up a simple 3D frame, solve it, and visualize the results.

## Package Structure

- `src/frame_solver/`: Contains the main implementation of the 3D Frame Solver.
- `tests/`: Contains unit tests for the solver.
- `examples/`: Contains example scripts demonstrating the usage of the solver.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

