# JAXFLUIDS Tutorials

This document provides a guide to the tutorials and examples available in the JAXFLUIDS package. These resources are designed to help you get started with running simulations and understanding the capabilities of the package.

## Jupyter Notebooks

The `notebooks` directory contains a series of Jupyter notebooks that provide a hands-on introduction to JAXFLUIDS.

### Basics

The `notebooks/basics` directory covers the fundamental concepts of setting up and running simulations.

- **`case_setup.ipynb`**: This notebook demonstrates how to set up a simulation case, including defining the geometry, initial conditions, and boundary conditions.
- **`numerical_setup.ipynb`**: This notebook explains how to configure the numerical methods for a simulation, such as the choice of time-stepping scheme, spatial reconstruction, and Riemann solver.
- **`parallel_simulations.ipynb`**: This notebook shows how to run simulations in parallel on multiple devices (CPU/GPU/TPU).

### Simulations

The `notebooks/simulations` directory provides examples of various types of simulations.

- **`sod_shocktube`**: A classic 1D shock tube problem.
- **`gasliquid_shocktube`**: A 1D shock tube problem with a gas-liquid interface.
- **`air_helium`**: A 2D simulation of a shock wave interacting with a helium bubble in air.
- **`air_water`**: A 2D simulation of a shock wave interacting with a water bubble in air.
- **`cylinderflow`**: A 2D simulation of flow over a cylinder.
- **`laminar_boundarylayer`**: A 2D simulation of a laminar boundary layer.

## Examples

The `examples` directory contains Python scripts that demonstrate how to run simulations from the command line. The examples are organized by dimensionality (1D, 2D, and 3D).

To run an example, you can navigate to the corresponding directory and execute the Python script. For example, to run the 1D Sod shock tube example, you would do the following:

```bash
cd JAXFlUIDS_Package/examples/examples_1D/sod_shock_tube
python3 sod_shock_tube.py
```

These examples are a great starting point for developing your own simulation scripts. 