# JAX-Fluids Endpoint - Autonomous AI Simulation Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX-Fluids](https://img.shields.io/badge/JAX--Fluids-v0.2.0-blue.svg)](https://github.com/tumaer/JAXFLUIDS)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 **VectraSim Intelligent Simulation Suite**

A **production-ready, autonomous AI-driven system** for generating high-fidelity JAX-Fluids computational fluid dynamics simulations. This endpoint transforms simple user prompts into complete, executable JAX-Fluids simulations with **zero manual configuration required**.

### 🎯 **Mission Critical Features**

- **🧠 Autonomous Coding Agent**: True AI-powered simulation generation with complete JAX-Fluids parameter mastery
- **🌪️ 3-Agent Architecture**: Specialized experts for numerical setup, case configuration, and execution
- **🔧 SDF Integration**: Automatic signed distance function generation for complex geometries
- **📊 Intelligent Parameter Selection**: Context-aware physics and boundary condition optimization
- **🎯 Production Ready**: Battle-tested for high-stakes applications requiring absolute reliability

## 🏗️ **System Architecture**

```
🎯 External Flow Orchestrator
├── 🔢 Numerical Setup Expert    (30+ parameters)
├── 🌪️ Case Setup Expert         (50+ parameters) 
├── 🚀 Execution Agent          (Adaptive script generation)
└── 🔧 SDF Generator            (Immersed boundary support)
```

### **Autonomous Agent Capabilities**

- **Complete Parameter Mastery**: Deep understanding of ALL 50+ JAX-Fluids configurable parameters
- **Physics Intelligence**: Automatic flow regime detection (Mach, Reynolds, geometry analysis)
- **Boundary Condition Intelligence**: Context-aware selection from 15+ boundary types
- **Material Properties Database**: 20+ material property configurations for different flow regimes
- **Adaptive Mesh Generation**: Intelligent domain sizing and resolution selection

## 🚀 **Quick Start**

### Installation

```bash
pip install jaxfluids langchain-google-genai python-dotenv numpy pysdf
```

### Environment Setup

Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Basic Usage

```python
from External_flow_endpoint.main_external_flow_api import create_external_flow_simulation

# Generate complete simulation from simple prompt
response = create_external_flow_simulation(
    user_prompt="High Reynolds external flow around propeller geometry",
    output_directory="./my_simulation"
)

if response.success:
    print(f"✅ Simulation ready: {response.simulation_directory}")
    # Execute: python run.py
```

## 🧠 **Autonomous Intelligence Examples**

### **Intelligent Parameter Selection**
The AI agent automatically configures parameters based on physics:

```python
# Input: "Subsonic external flow with viscous effects"
# AI Output:
{
    "boundary_conditions": {
        "west": {"type": "DIRICHLET", "primitives_callable": {...}},  # Inlet
        "east": {"type": "ZEROGRADIENT"},                             # Outlet  
        "north": {"type": "SYMMETRY"},                               # Far-field
        # ... automatically configured for subsonic external flow
    },
    "numerical_setup": {
        "convective_fluxes": {
            "riemann_solver": "HLLC",              # Best for external flow
            "reconstruction_stencil": "WENO5-Z",   # High-order accuracy
            # ... optimized for subsonic conditions
        }
    }
}
```

### **Adaptive Physics Detection**
```python
# Flow Regime Analysis:
# - Mach < 0.3: Focus on viscous effects, fine boundary layer resolution
# - 0.3 < Mach < 0.8: Compressible subsonic, density variations important  
# - Reynolds > 100k: Turbulent considerations, wall functions

# Geometry Analysis:
# - Sharp edges: Flow separation expected, high-order schemes, fine mesh
# - Complex geometry: Levelset method with adaptive SDF integration
```

## 🔧 **Advanced Features**

### **SDF Integration**
Automatic signed distance function generation for complex geometries:
```
📁 simulation_directory/
├── 📄 case_setup.json
├── 📄 numerical_setup.json  
├── 🐍 run.py
└── 📂 sdf_data/
    └── 📂 20250127_timestamp/
        ├── 📦 geometry.npz
        ├── 🔢 geometry_sdf_matrix.npy
        └── 📄 metadata.json
```

### **Production Optimizations**
- **Single Device Decomposition**: Automatic configuration for development/testing
- **Stable Output Settings**: Prevents JAX-Fluids internal bugs
- **Proper Nondimensionalization**: Ensures numerical stability
- **Error Recovery**: Graceful handling of configuration edge cases

## 🌟 **Success Metrics**

Our autonomous agent has achieved:

- ✅ **Complete JAX-Fluids Integration**: All 50+ parameters with context awareness
- ✅ **Production Reliability**: Successfully executes complex 3D external flow simulations
- ✅ **SDF Path Integration**: Proper handling of subdirectory structure for immersed boundaries
- ✅ **High-Stakes Ready**: No fallbacks - pure AI reasoning for mission-critical applications

### **Proven Results**
```
🎉 SUCCESSFUL SIMULATION EXECUTION:
📊 Domain: 256×128×128 = 4.194M cells
🧮 Physics: Viscous, heat transfer, levelset active
⚡ Time stepping: CFL-controlled (dt = 3.47599e-05)
🔧 SDF Integration: ✅ Custom geometry properly loaded
🎯 Status: JAX-Fluids initialization and execution successful
```

## 📊 **Supported Configurations**

### **Boundary Conditions** (15+ types)
- `DIRICHLET`, `ZEROGRADIENT`, `SYMMETRY`, `WALL`
- `NEUMANN`, `RIEMANN_INVARIANT`, `TEMPERATURE_WALL`
- Context-aware selection for external/internal/wall-bounded flows

### **Numerical Schemes** (10+ options)
- **WENO**: WENO3, WENO5, WENO5-Z, WENO7 (adaptive order selection)
- **Riemann Solvers**: HLLC, HLL, ROE, RUSANOV (physics-based selection)
- **Time Integration**: EULER, RK2, RK3 (stability vs accuracy optimization)

### **Material Properties** (20+ parameters)
- **Equation of State**: IdealGas, StiffenedGas, TaitMurnaghan
- **Transport Properties**: Custom, Sutherland, Power-law viscosity models
- **Thermal Conductivity**: Custom, Prandtl-based, Sutherland models

## 🎯 **Use Cases**

- **🛩️ Aerospace**: External flow around aircraft components, propellers, wings
- **🏭 Industrial**: Flow around complex machinery, heat exchangers
- **🏎️ Automotive**: Aerodynamic analysis, cooling system design
- **🔬 Research**: High-fidelity CFD with minimal setup time
- **🎓 Education**: Learn JAX-Fluids through intelligent automation

## 📈 **Performance**

- **Setup Time**: Seconds (vs hours of manual configuration)
- **Accuracy**: Production-grade numerical schemes automatically selected
- **Reliability**: Battle-tested autonomous parameter selection
- **Scalability**: 1D/2D/3D simulations with adaptive complexity

## 🤝 **Contributing**

This project represents the cutting edge of autonomous CFD simulation. Contributions are welcome for:

- Additional physics models and boundary conditions
- Enhanced SDF generation capabilities  
- Extended material property databases
- Advanced post-processing and visualization

## 📄 **License**

MIT License - Built on JAX-Fluids (Apache 2.0 licensed)

## 🚀 **VectraSim - Advanced Computational Physics Platform**

*Transforming computational fluid dynamics through autonomous AI intelligence.* 