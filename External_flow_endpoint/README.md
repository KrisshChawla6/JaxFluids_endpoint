# JAX-Fluids External Flow API 🌪️

**AI-Driven 3D External Flow Simulations with Immersed Boundaries**

A sophisticated **3-agent agentic system** powered by Gemini 2.5 Pro for generating complete JAX-Fluids external flow simulations. Specializes in **subsonic wind tunnel setups** with **immersed boundary conditions** using signed distance functions (SDFs).

## 🎯 **System Overview**

This endpoint provides a complete AI-driven workflow for external flow CFD simulations:

1. **Multimodal Input Processing**: Natural language prompts + optional context
2. **Automatic SDF Integration**: Seamlessly integrates with immersed boundary endpoint
3. **3-Agent Orchestration**: Specialized experts for numerical, case, and execution setup
4. **Production-Ready Output**: Complete JAX-Fluids simulation ready to run

## 🏗️ **Architecture**

```
External_flow_endpoint/
├── main_external_flow_api.py           # 🚪 Main API entry point
├── external_flow_orchestrator.py       # 🎯 Master orchestrator agent  
├── subsonic_windtunnel/                # 🌪️ 3 specialized expert agents
│   ├── numerical_setup_expert.py       # 🔢 30+ numerical parameters
│   ├── case_setup_expert.py           # ⚗️ Case/domain/BC expert
│   └── execution_agent.py             # 🚀 Run script generator
├── test_external_flow_api.py          # 🧪 Comprehensive test suite
└── README.md                          # 📚 This documentation
```

## ✨ **Key Features**

### 🤖 **AI-Powered Parameter Selection**
- **50+ JAX-Fluids Parameters** expertly configured
- **Gemini 2.5 Pro** with comprehensive framework knowledge
- **Context-aware** parameter optimization for subsonic flows

### 🌪️ **Wind Tunnel Expertise**
- **Intelligent Boundary Conditions**: Inlet/outlet/far-field setup
- **Domain Sizing**: Automatic wind tunnel scaling (5-10x object size)
- **Flow Physics**: Mach number, angle of attack, viscous effects

### 🎯 **SDF Integration**
- **Automatic Detection**: Finds latest SDF from immersed boundary endpoint
- **Seamless Integration**: Copies and configures SDF for levelset method
- **3-Slot History**: Works with the history window system

### 📊 **Production-Grade Output**
- **numerical_setup.json**: Complete numerical configuration
- **case_setup.json**: Domain, BCs, materials, initial conditions  
- **run.py**: Production-ready execution script
- **Comprehensive Logging**: Full parameter tracking and timing

## 🚀 **Quick Start**

### 1. **Setup Environment**
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional: JAX configuration
export JAX_PLATFORM_NAME="cpu"  # or "gpu"
export JAX_ENABLE_X64="True"
```

### 2. **Basic Usage**
```python
from main_external_flow_api import create_external_flow_simulation

# Simple external flow simulation
response = create_external_flow_simulation(
    user_prompt="Create a subsonic external flow simulation around a propeller at Mach 0.3, 5 degrees angle of attack, using WENO5-Z reconstruction and HLLC Riemann solver",
    output_directory="propeller_simulation"
)

if response.success:
    print(f"✅ Simulation ready: {response.simulation_directory}")
    # Run: cd propeller_simulation/jaxfluids_external_flow_xxx && python run.py
```

### 3. **Advanced Usage**
```python
from main_external_flow_api import ExternalFlowAPI, ExternalFlowRequest

# Advanced configuration with multimodal context
multimodal_context = {
    'flow_requirements': {
        'accuracy': 'high',
        'physics': ['viscous', 'thermal'],
        'time_constraint': 'moderate'
    },
    'geometry_info': {
        'type': 'airfoil',
        'complexity': 'high'
    }
}

api = ExternalFlowAPI()
request = ExternalFlowRequest(
    user_prompt="High-fidelity viscous flow analysis with heat transfer at Mach 0.2, 8 degrees AOA",
    multimodal_context=multimodal_context,
    output_directory="high_fidelity_analysis"
)

response = api.process_external_flow_request(request)
```

## 🔬 **Expert Agent Capabilities**

### 🔢 **Numerical Setup Expert** (30+ Parameters)
Handles all JAX-Fluids numerical parameters with deep framework knowledge:

- **Time Integration**: EULER, RK2, RK3 with optimal CFL selection
- **Convective Schemes**: GODUNOV, AUSM, LAX_FRIEDRICHS, CENTRAL
- **Riemann Solvers**: HLLC, HLL, ROE, RUSANOV, LAX_FRIEDRICHS  
- **Reconstruction**: WENO3, WENO5, WENO5-Z, WENO7, CENTRAL variants
- **Levelset Settings**: Narrowband, reinitialization, interface interaction
- **Precision Control**: Double/single precision for compute and output

**Example Generated Configuration:**
```json
{
    "conservatives": {
        "halo_cells": 5,
        "time_integration": {
            "integrator": "RK3",
            "CFL": 0.5
        },
        "convective_fluxes": {
            "convective_solver": "GODUNOV",
            "godunov": {
                "riemann_solver": "HLLC",
                "signal_speed": "EINFELDT",
                "reconstruction_stencil": "WENO5-Z",
                "reconstruction_variable": "CHAR-PRIMITIVE"
            }
        }
    },
    "active_physics": {
        "is_convective_flux": true,
        "is_viscous_flux": false,
        "is_levelset": true
    },
    "levelset": {
        "narrowband_computation": {
            "is_narrowband": true,
            "narrowband_size": 10
        }
    }
}
```

### 🌪️ **Case Setup Expert** (20+ Parameters)
Specializes in wind tunnel physics and domain configuration:

- **Wind Tunnel BCs**: Intelligent inlet/outlet/far-field setup
- **Domain Scaling**: Automatic sizing based on SDF bounds
- **Flow Conditions**: Mach number, AOA, altitude effects
- **Material Properties**: Air properties, viscosity, thermal conductivity
- **SDF Integration**: Automatic levelset configuration

**Wind Tunnel Boundary Logic:**
- **INLET (west)**: DIRICHLET with freestream conditions
- **OUTLET (east)**: ZEROGRADIENT (subsonic outflow)  
- **FAR-FIELD**: SYMMETRY or ZEROGRADIENT based on domain size
- **LEVELSET**: Always ZEROGRADIENT for all boundaries

### 🚀 **Execution Agent** (7+ Parameters)
Generates production-ready JAX-Fluids execution scripts:

- **Error Handling**: Robust file checking and JAX device detection
- **Progress Monitoring**: Professional logging and timing
- **Post-Processing**: Basic result handling and visualization setup
- **Execution Management**: Optional automatic simulation launching

## 📊 **Parameter Coverage**

### **Numerical Parameters (27+)**
```python
"conservatives.time_integration.integrator"           # EULER, RK2, RK3
"conservatives.time_integration.CFL"                  # 0.1-0.9
"conservatives.convective_fluxes.convective_solver"   # GODUNOV, AUSM, etc.
"conservatives.convective_fluxes.godunov.riemann_solver"  # HLLC, HLL, ROE, etc.
"conservatives.convective_fluxes.godunov.reconstruction_stencil"  # WENO3-7, CENTRAL
"levelset.narrowband_computation.is_narrowband"       # Efficiency optimization
"levelset.reinitialization.reinitialization_interval" # Levelset quality
"precision.is_double_precision_compute"               # Accuracy control
# ... and 19+ more parameters
```

### **Case Parameters (20+)**
```python
"domain.x.cells"                                      # Grid resolution
"domain.x.range"                                      # Domain bounds
"boundary_conditions.primitives.west.type"           # Inlet BC
"boundary_conditions.primitives.east.type"           # Outlet BC
"initial_condition.primitives.rho"                   # Density
"initial_condition.primitives.u"                     # Velocity components
"material_properties.equation_of_state.specific_heat_ratio"  # Gamma
"material_properties.transport.dynamic_viscosity.value"      # Viscosity
"nondimensionalization_parameters.velocity_reference"        # Reference values
# ... and 11+ more parameters
```

## 🧪 **Testing**

Run the comprehensive test suite:

```bash
python test_external_flow_api.py
```

**Test Coverage:**
- ✅ Basic external flow setups (propeller, airfoil)
- ✅ High-fidelity viscous simulations  
- ✅ Quick analysis configurations
- ✅ Advanced API with multimodal context
- ✅ Parameter coverage verification
- ✅ SDF integration testing

## 🔧 **Configuration Examples**

### **Propeller Analysis**
```python
response = create_external_flow_simulation(
    user_prompt="Subsonic propeller analysis at Mach 0.3, 5° AOA, high accuracy with WENO5-Z and HLLC solver"
)
```

### **Viscous Airfoil**
```python
response = create_external_flow_simulation(
    user_prompt="Viscous flow around airfoil at Mach 0.2, 8° AOA, with heat transfer, enable thermal effects"
)
```

### **Quick Preliminary Design**
```python
response = create_external_flow_simulation(
    user_prompt="Quick inviscid analysis for preliminary design, Mach 0.25, 3° AOA, efficiency priority"
)
```

## 🌐 **Integration with Immersed Boundary Endpoint**

The system automatically integrates with your immersed boundary SDF endpoint:

1. **Auto-Detection**: Finds latest SDF from `../immersed_boundary_endpoint_final/sdf_files/`
2. **3-Slot History**: Works with the history window system
3. **Domain Scaling**: Uses SDF bounds to create appropriate wind tunnel domain
4. **Levelset Setup**: Configures JAX-Fluids levelset method automatically

## 🎯 **Subsonic Wind Tunnel Expertise**

### **Best Practices Implemented**
- **Domain Sizing**: 5-10x object size for proper far-field
- **Boundary Conditions**: Physics-based inlet/outlet/far-field setup
- **Grid Resolution**: Balanced resolution for accuracy vs efficiency
- **Solver Selection**: HLLC Riemann solver for subsonic external flow
- **Reconstruction**: WENO5-Z for smooth flow accuracy
- **Time Integration**: RK3 for high-order temporal accuracy

### **Flow Physics Handling**
- **Mach Number Extraction**: Automatic velocity computation
- **Angle of Attack**: Proper velocity component decomposition  
- **Material Properties**: Standard air properties with altitude effects
- **Viscous Effects**: Automatic viscosity enabling based on prompt
- **Heat Transfer**: Thermal conductivity setup when requested

## 🚀 **Production Readiness**

### **Generated Files**
Each simulation includes:
- `numerical_setup.json` - Complete numerical configuration
- `case_name.json` - Domain, BCs, materials, initial conditions
- `run.py` - Production-ready execution script
- `custom_sdf.npy` - Integrated SDF for levelset method
- `simulation_summary.json` - Complete processing metadata

### **Execution**
```bash
cd simulation_directory/jaxfluids_external_flow_timestamp/
python run.py
```

The generated `run.py` includes:
- JAX device detection and configuration
- Robust error handling and file verification
- Professional progress logging
- Timing and performance metrics
- Basic post-processing setup

## 📈 **Performance**

- **Parameter Generation**: ~2-5 seconds per agent
- **Total Setup Time**: ~10-20 seconds for complete simulation
- **SDF Integration**: Automatic and seamless
- **Memory Efficient**: Uses JAX-Fluids' built-in optimizations

## 🤝 **Integration with Your Software Backend**

The API is designed to integrate seamlessly with your existing software backend:

```python
# Your backend calls
from External_flow_endpoint.main_external_flow_api import create_external_flow_simulation

def handle_external_flow_request(user_prompt, context):
    """Your backend integration point"""
    
    response = create_external_flow_simulation(
        user_prompt=user_prompt,
        multimodal_context=context,
        output_directory=f"simulations/{user_id}/external_flow"
    )
    
    return {
        'success': response.success,
        'simulation_directory': response.simulation_directory,
        'ready_to_run': response.run_script_file is not None
    }
```

## 🔍 **Troubleshooting**

### **Common Issues**
1. **Missing Gemini API Key**: Set `GEMINI_API_KEY` environment variable
2. **No SDF Found**: Ensure immersed boundary endpoint has generated SDFs
3. **Import Errors**: Check Python path and JAX-Fluids installation
4. **Parameter Parsing**: Check JSON format in generated configuration files

### **Debug Mode**
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 **References**

- [JAX-Fluids Documentation](https://jax-fluids.readthedocs.io/)
- [JAX-Fluids GitHub](https://github.com/tumaer/JAXFLUIDS)
- [JAX-Fluids Paper](https://arxiv.org/abs/2402.05193)

---

**🎉 Ready for Production-Grade External Flow Simulations!**

This system provides expert-level JAX-Fluids configuration with the convenience of natural language input and automatic SDF integration. Perfect for aerospace, automotive, and general external flow applications. 