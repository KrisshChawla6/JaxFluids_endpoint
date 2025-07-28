#!/usr/bin/env python3
"""
High-Quality JAX-Fluids Rocket Visualization
Uses proper SDF visualization with marching cubes like the immersed boundary endpoint
"""

import pyvista as pv
import numpy as np
import h5py
from pathlib import Path

try:
    import mcubes
    MCUBES_AVAILABLE = True
except ImportError:
    MCUBES_AVAILABLE = False
    try:
        from skimage import measure
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False

def load_jaxfluids_data(h5_file):
    """Load JAX-Fluids data from proper HDF5 structure"""
    data = {}
    
    with h5py.File(h5_file, 'r') as f:
        print(f"Loading data from: {h5_file}")
        
        # Load primitives
        if 'primitives' in f:
            for var in f['primitives'].keys():
                data[var] = f['primitives'][var][...]
                print(f"  {var}: shape={data[var].shape}, range=[{data[var].min():.6f}, {data[var].max():.6f}]")
        
        # Load miscellaneous
        if 'miscellaneous' in f:
            for var in f['miscellaneous'].keys():
                data[var] = f['miscellaneous'][var][...]
                print(f"  {var}: shape={data[var].shape}, range=[{data[var].min():.6f}, {data[var].max():.6f}]")
    
    return data

def load_rocket_sdf():
    """Load the rocket SDF using proper path resolution"""
    sdf_paths = [
        "intelligent_boundary_conditions/working/hardcoded_run/attempt1/rocket_case/20250728_003202/Rocket Engine_sdf_matrix.npy",
        "intelligent_boundary_conditions/working/rocket_simulation_final/masks/Rocket Engine_sdf_matrix.npy"
    ]
    
    for sdf_path in sdf_paths:
        if Path(sdf_path).exists():
            print(f"Loading rocket SDF from: {sdf_path}")
            sdf = np.load(sdf_path)
            print(f"  SDF shape: {sdf.shape}, range=[{sdf.min():.3f}, {sdf.max():.3f}]")
            return sdf
    
    print("❌ Could not find rocket SDF file")
    return None

def load_inlet_outlet_masks():
    """Load inlet/outlet masks"""
    masks = {}
    
    mask_paths = [
        ("intelligent_boundary_conditions/working/rocket_simulation_final/masks/inlet_boundary_mask.npy", "inlet"),
        ("intelligent_boundary_conditions/working/rocket_simulation_final/masks/outlet_boundary_mask.npy", "outlet")
    ]
    
    for mask_path, mask_type in mask_paths:
        if Path(mask_path).exists():
            print(f"Loading {mask_type} mask from: {mask_path}")
            mask_data = np.load(mask_path)
            masks[mask_type] = mask_data
            print(f"  {mask_type} mask: shape={mask_data.shape}, active points={np.sum(mask_data):,}")
    
    return masks

def extract_rocket_surface_professional(sdf_grid, domain_bounds):
    """Extract rocket surface using proper marching cubes like immersed boundary endpoint"""
    print("🎨 Extracting φ=0 boundary surface using professional method...")
    
    if not MCUBES_AVAILABLE and not SKIMAGE_AVAILABLE:
        print("❌ No marching cubes library available!")
        return None
    
    try:
        if MCUBES_AVAILABLE:
            vertices, triangles = mcubes.marching_cubes(sdf_grid, 0.0)
            print("🔥 Using PyMCubes for high-quality isosurface extraction")
        else:
            vertices, triangles, _, _ = measure.marching_cubes(sdf_grid, 0.0)
            print("⚙️ Using scikit-image for isosurface extraction")
        
        # Transform to world coordinates
        nx, ny, nz = sdf_grid.shape
        x_min, y_min, z_min, x_max, y_max, z_max = domain_bounds
        
        vertices[:, 0] = x_min + (vertices[:, 0] / (nx - 1)) * (x_max - x_min)
        vertices[:, 1] = y_min + (vertices[:, 1] / (ny - 1)) * (y_max - y_min)
        vertices[:, 2] = z_min + (vertices[:, 2] / (nz - 1)) * (z_max - z_min)
        
        print(f"🎯 Extracted {len(vertices):,} vertices, {len(triangles):,} triangles")
        
        # Create PyVista mesh from marching cubes result
        faces = np.c_[np.full(triangles.shape[0], 3), triangles]
        rocket_mesh = pv.PolyData(vertices, faces)
        
        return rocket_mesh
        
    except Exception as e:
        print(f"❌ Failed to extract surface: {e}")
        return None

def create_inlet_outlet_markers(masks, domain_bounds, grid_shape):
    """Create proper inlet/outlet markers as spheres at centroids"""
    markers = {}
    
    # Domain coordinate arrays - masks are in (x, y, z) = (128, 64, 64)
    # But flow grid is (z, y, x) = (64, 64, 128)
    
    for mask_type, mask in masks.items():
        # Masks are (128, 64, 64) in (x, y, z) order
        mask_nx, mask_ny, mask_nz = mask.shape  # (128, 64, 64)
        
        # Create coordinate arrays for mask space
        mask_x = np.linspace(domain_bounds[0], domain_bounds[3], mask_nx)
        mask_y = np.linspace(domain_bounds[1], domain_bounds[4], mask_ny)
        mask_z = np.linspace(domain_bounds[2], domain_bounds[5], mask_nz)
        
        Mask_X, Mask_Y, Mask_Z = np.meshgrid(mask_x, mask_y, mask_z, indexing='ij')
        
        # Find centroid of active mask points in mask coordinate space
        active_indices = np.where(mask)
        if len(active_indices[0]) > 0:
            centroid_x = np.mean(Mask_X[active_indices])
            centroid_y = np.mean(Mask_Y[active_indices])
            centroid_z = np.mean(Mask_Z[active_indices])
            
            # Create sphere marker at centroid
            sphere = pv.Sphere(radius=50.0, center=[centroid_x, centroid_y, centroid_z])
            markers[mask_type] = {
                'mesh': sphere,
                'centroid': [centroid_x, centroid_y, centroid_z],
                'count': len(active_indices[0])
            }
            
            print(f"✅ {mask_type.title()} marker: center=({centroid_x:.1f}, {centroid_y:.1f}, {centroid_z:.1f}), {len(active_indices[0]):,} points")
    
    return markers

def create_high_quality_visualization():
    """Create high-quality rocket visualization"""
    print("🚀 HIGH-QUALITY ROCKET VISUALIZATION")
    print("=" * 60)
    
    # Find latest data file
    data_dir = Path("intelligent_boundary_conditions/working/rocket_simulation_final/output/rocket_nozzle_internal_supersonic_production/domain")
    if not data_dir.exists():
        print("❌ No output directory found!")
        return
    
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        print("❌ No HDF5 files found!")
        return
    
    latest_file = sorted(h5_files)[-1]
    
    # Load all data
    flow_data = load_jaxfluids_data(latest_file)
    rocket_sdf = load_rocket_sdf()
    masks = load_inlet_outlet_masks()
    
    if rocket_sdf is None:
        print("❌ Cannot proceed without rocket SDF")
        return
    
    # Domain and grid setup
    domain_bounds = [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0]
    grid_shape = flow_data['density'].shape  # (64, 64, 128)
    
    print(f"\n🎨 Creating high-quality visualization...")
    print(f"Grid shape: {grid_shape}")
    print(f"Domain bounds: {domain_bounds}")
    
    # Extract rocket surface using professional method
    if rocket_sdf.shape == (128, 64, 64):  # (x, y, z)
        sdf_for_extraction = rocket_sdf.transpose(2, 1, 0)  # -> (z, y, x)
    else:
        sdf_for_extraction = rocket_sdf
    
    rocket_surface = extract_rocket_surface_professional(sdf_for_extraction, domain_bounds)
    
    # Create inlet/outlet markers
    inlet_outlet_markers = create_inlet_outlet_markers(masks, domain_bounds, grid_shape)
    
    # Create plotter with high quality settings
    plotter = pv.Plotter(window_size=[1920, 1080])
    plotter.set_background('white')
    
    print("\n🔧 Adding rocket surface...")
    
    # Add rocket surface with professional styling
    if rocket_surface is not None:
        plotter.add_mesh(rocket_surface,
                        color='lightgray',
                        opacity=0.8,
                        show_edges=False,
                        smooth_shading=True,
                        label='Rocket Nozzle')
        print(f"   ✅ Rocket surface: {rocket_surface.n_points:,} vertices, {rocket_surface.n_cells:,} faces")
    
    # Add inlet marker (red sphere)
    if 'inlet' in inlet_outlet_markers:
        plotter.add_mesh(inlet_outlet_markers['inlet']['mesh'],
                        color='red',
                        opacity=0.8,
                        label='Inlet')
        center = inlet_outlet_markers['inlet']['centroid']
        plotter.add_point_labels([center], ['INLET'], font_size=16, text_color='darkred')
        print(f"   🔴 Inlet marker at ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    
    # Add outlet marker (green sphere)
    if 'outlet' in inlet_outlet_markers:
        plotter.add_mesh(inlet_outlet_markers['outlet']['mesh'],
                        color='green',
                        opacity=0.8,
                        label='Outlet')
        center = inlet_outlet_markers['outlet']['centroid']
        plotter.add_point_labels([center], ['OUTLET'], font_size=16, text_color='darkgreen')
        print(f"   🟢 Outlet marker at ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    
    # Add flow field slice
    print("\n🌊 Adding flow field slice...")
    
    # Create structured grid for flow field
    nz, ny, nx = grid_shape
    x = np.linspace(domain_bounds[0], domain_bounds[3], nx)
    y = np.linspace(domain_bounds[1], domain_bounds[4], ny)
    z = np.linspace(domain_bounds[2], domain_bounds[5], nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    flow_grid = pv.StructuredGrid()
    points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    flow_grid.points = points
    flow_grid.dimensions = [nx, ny, nz]
    
    # Add pressure data
    if 'pressure' in flow_data:
        flow_grid['pressure'] = flow_data['pressure'].transpose(2, 1, 0).ravel()
        
        # Create slice through center
        center_x = (domain_bounds[0] + domain_bounds[3]) / 2
        slice_plane = flow_grid.slice(normal='x', origin=[center_x, 0, 0])
        
        if slice_plane.n_points > 0:
            plotter.add_mesh(slice_plane,
                           scalars='pressure',
                           cmap='viridis',
                           opacity=0.6,
                           show_scalar_bar=True,
                           scalar_bar_args={'title': 'Pressure [Pa]', 'height': 0.8})
    
    # Professional styling
    plotter.add_axes(line_width=4, labels_off=False)
    plotter.add_text("High-Quality JAX-Fluids Rocket Nozzle Visualization\n"
                    "Professional SDF Surface + Inlet/Outlet Markers + Flow Field", 
                    font_size=14, position='upper_left')
    
    # Set camera for optimal view
    plotter.camera_position = [
        (2000, 1000, 800),   # camera position
        (400, 0, 0),         # focal point (nozzle center)
        (0, 0, 1)           # view up
    ]
    
    # Enable better rendering
    plotter.enable_anti_aliasing()
    plotter.enable_depth_peeling(4)
    
    print("\n🎮 Controls:")
    print("   Mouse: Rotate/Pan/Zoom")
    print("   'q': Quit")
    print("   'r': Reset view")
    
    # Show the visualization
    plotter.show()

if __name__ == "__main__":
    try:
        create_high_quality_visualization()
        print("✅ High-quality visualization complete!")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc() 