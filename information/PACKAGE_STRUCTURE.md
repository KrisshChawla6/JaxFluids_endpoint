# JAXFLUIDS Package Structure

This document outlines the structure of the JAXFLUIDS package. The source code is located in the `src` directory and is organized into the following subdirectories:

- **`jaxfluids`**: This is the core of the package, containing the main components for running CFD simulations.
- **`jaxfluids_nn`**: This directory likely contains modules related to neural networks, possibly for machine learning-based models or closures.
- **`jaxfluids_postprocess`**: This directory probably holds tools for post-processing simulation data, such as visualization and data analysis.
- **`jaxfluids_thirdparty`**: This directory may contain third-party libraries or dependencies used by JAXFLUIDS.

## Core Package (`jaxfluids`)

The `jaxfluids` directory is further organized into the following modules:

- **`simulation_manager.py`**: This appears to be a central script for managing simulations.
- **`solvers`**: This directory contains different numerical solvers for the governing equations.
- **`stencils`**: This likely contains implementations of different numerical stencils for spatial discretization (e.g., WENO).
- **`time_integration`**: This directory probably holds different time-stepping schemes (e.g., Runge-Kutta).
- **`turbulence`**: This directory likely contains turbulence models (e.g., ALDM).
- **`levelset`** and **`diffuse_interface`**: These directories likely contain methods for simulating two-phase flows.
- **`materials`**: This directory probably defines the properties of different fluids.
- **`domain`**: This directory likely handles the computational domain and grid.
- **`initialization`**: This directory probably contains functions for setting the initial conditions of a simulation.
- **`boundary_conditions`**: Although not a directory, the presence of boundary condition logic is inferred from the `README.MD`.
- **`parallel`**: This directory likely contains the logic for parallelizing simulations.
- **`io_utils`**: This directory probably handles input/output operations, like reading configuration files and writing simulation results.
- **`helper_functions`**: A collection of utility functions.
- **`config`**: This likely handles configuration management.
- **`forcing`**: This directory probably contains implementations of different forcing terms that can be added to the governing equations.

This structure is based on the file and directory names. For more detailed information, you should refer to the source code and the official documentation. 