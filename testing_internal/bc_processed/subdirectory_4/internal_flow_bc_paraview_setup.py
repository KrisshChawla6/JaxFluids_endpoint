#!/usr/bin/env python3
"""
ParaView Python Script for Intelligent Boundary Condition Visualization
Generated automatically by intelligent_BC_final endpoint

Usage:
1. Open ParaView
2. Tools -> Python Shell
3. Run Script -> Select this file
   OR
4. pvpython internal_flow_bc_paraview_setup.py
"""

import paraview.simple as pvs

# Load VTK file
reader = pvs.OpenDataFile('internal_flow_bc_visualization.vtm')

# Get render view
view = pvs.GetActiveViewOrCreate('RenderView')

# Set black background
view.Background = [0, 0, 0]

# Create displays for each component
rocket_display = pvs.Show(reader, view)

# Set up color scheme
rocket_display.ColorArrayName = ['CELLS', 'boundary_type']
rocket_display.LookupTable = pvs.GetColorTransferFunction('boundary_type')

# Configure lookup table for boundary types
lut = rocket_display.LookupTable
lut.RGBPoints = [
    0.0, 1.0, 0.0, 0.0,    # Inlet = Red
    1.0, 0.8, 0.8, 0.8,    # Euler/Slipwall = Light Gray  
    2.0, 0.0, 1.0, 0.0     # Outlet = Green
]
lut.ColorSpace = 'RGB'
lut.NanColor = [0.5, 0.5, 0.5]

# Add color legend
legend = pvs.GetScalarBar(lut, view)
legend.Title = 'Boundary Types'
legend.ComponentTitle = ''
legend.LabelFormat = '%-#6.0f'
legend.RangeLabelFormat = '%-#6.0f'

# Position and style legend
legend.Position = [0.85, 0.2]
legend.ScalarBarLength = 0.6
legend.ScalarBarThickness = 16
legend.TitleColor = [1, 1, 1]
legend.LabelColor = [1, 1, 1]

# Set camera position for good view
view.CameraPosition = [2500, 1500, 1000]
view.CameraFocalPoint = [800, 0, 0]
view.CameraViewUp = [0, 0, 1]

# Reset and render
pvs.ResetCamera()
pvs.Render()

print("ParaView visualization loaded successfully!")
print("Red = Inlet boundary")
print("Gray = Euler/Slipwall boundaries") 
print("Green = Outlet boundary")
print("Domain box and flow arrows included")
