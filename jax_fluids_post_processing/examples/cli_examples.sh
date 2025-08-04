#!/bin/bash
# 
# CLI Usage Examples for JAX-Fluids Post-Processing
# 

echo "JAX-Fluids Post-Processing CLI Examples"
echo "======================================="

# Set paths (adjust these for your data)
RESULTS_PATH="subsonic_wind_tunnel_external_flow-1/domain"
OUTPUT_PATH="cli_output"
MESH_PATH="data/mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh"

echo ""
echo "1. Quick Visualization"
echo "----------------------"
echo "Command: jax-fluids-postprocess quick-viz $RESULTS_PATH --variable velocity_magnitude --mesh-path $MESH_PATH"
# jax-fluids-postprocess quick-viz $RESULTS_PATH --variable velocity_magnitude --mesh-path $MESH_PATH

echo ""
echo "2. Full Processing Workflow"
echo "---------------------------"
echo "Command: jax-fluids-postprocess process $RESULTS_PATH $OUTPUT_PATH --plot --export-vtk --mesh-path $MESH_PATH"
# jax-fluids-postprocess process $RESULTS_PATH $OUTPUT_PATH --plot --export-vtk --mesh-path $MESH_PATH

echo ""
echo "3. Create Animation"
echo "-------------------"
echo "Command: jax-fluids-postprocess animate $RESULTS_PATH $OUTPUT_PATH --variable pressure --plane xy --fps 15"
# jax-fluids-postprocess animate $RESULTS_PATH $OUTPUT_PATH --variable pressure --plane xy --fps 15

echo ""
echo "4. Export Multiple Variables"
echo "-----------------------------"
echo "Command: jax-fluids-postprocess export $RESULTS_PATH $OUTPUT_PATH --variables velocity_magnitude pressure density --mesh-path $MESH_PATH"
# jax-fluids-postprocess export $RESULTS_PATH $OUTPUT_PATH --variables velocity_magnitude pressure density --mesh-path $MESH_PATH

echo ""
echo "5. Process Specific Time Steps"
echo "-------------------------------"
echo "Command: jax-fluids-postprocess export $RESULTS_PATH $OUTPUT_PATH --time-indices 0 5 10 --variables velocity_magnitude"
# jax-fluids-postprocess export $RESULTS_PATH $OUTPUT_PATH --time-indices 0 5 10 --variables velocity_magnitude

echo ""
echo "6. Save Screenshot"
echo "------------------"
echo "Command: jax-fluids-postprocess process $RESULTS_PATH $OUTPUT_PATH --save-screenshot screenshot.png --variable pressure"
# jax-fluids-postprocess process $RESULTS_PATH $OUTPUT_PATH --save-screenshot screenshot.png --variable pressure

echo ""
echo "7. Different Animation Formats"
echo "-------------------------------"
echo "Command: jax-fluids-postprocess animate $RESULTS_PATH $OUTPUT_PATH --format mp4 --variable velocity_magnitude --plane xz"
# jax-fluids-postprocess animate $RESULTS_PATH $OUTPUT_PATH --format mp4 --variable velocity_magnitude --plane xz

echo ""
echo "8. Help and Documentation"
echo "-------------------------"
echo "Command: jax-fluids-postprocess --help"
# jax-fluids-postprocess --help

echo "Command: jax-fluids-postprocess process --help"
# jax-fluids-postprocess process --help

echo ""
echo "To run these commands, uncomment them in this script or copy-paste to terminal."
echo "Make sure to adjust the paths to match your data location."