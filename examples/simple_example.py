"""Simple example demonstrating basic terrain mesh generation.

This example shows the minimal code needed to generate a mesh from a DEM file
with uniform grid spacing.
"""
import terrain_mesh as tm
from pathlib import Path


def main():
    """Generate a simple uniform mesh from DEM data."""
    
    # Define paths (update these for your system)
    dem_path = "path/to/your/terrain.tif"   # Required: path to your GeoTIFF/DAT/NetCDF DEM
    output_dir = "./mesh_output"
    
    # Create simple configuration.
    # center_lat / center_lon are optional: if omitted the centre is
    # auto-detected from the geographic extent of the DEM file.
    terrain_config = tm.TerrainConfig(
        # Uncomment and replace with your site's coordinates to crop a
        # specific sub-region rather than the full DEM extent.
        # center_lat=39.71121111,
        # center_lon=-7.73483333,
        crop_size_km=25,
        rotation_deg=0,
        smoothing_sigma=0  # No smoothing
    )
    
    # Create uniform grid (200x200 cells)
    grid_config = tm.create_uniform_grid_config(nx=200, ny=200)
    
    # Create mesh configuration
    mesh_config = tm.MeshConfig(
        domain_height=3000.0,  # 3 km height
        total_z_cells=50
    )
    
    # Run the pipeline
    pipeline = tm.TerrainMeshPipeline()
    results = pipeline.run(
        dem_path=dem_path,
        terrain_config=terrain_config,
        grid_config=grid_config,
        mesh_config=mesh_config,
        output_dir=output_dir
    )
    
    print(f"\n✓ Mesh generated successfully!")
    print(f"  Terrain map: {results['terrain_map_path']}")
    print(f"  blockMeshDict: {results['blockmesh_path']}")


if __name__ == "__main__":
    main()
