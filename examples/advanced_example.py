"""Advanced example with multi-block grading and boundary treatment.

This example demonstrates:
- Multi-block grading for refined center region
- Vertical grading for near-ground refinement
- Directional boundary treatment
- Roughness map integration
"""
import terrain_mesh as tm


def main():
    """Generate an advanced mesh with grading and boundary treatment."""
    
    # Define paths (update these for your system)
    dem_path = "path/to/your/terrain.tif"
    roughness_path = "path/to/your/roughness.tif"  # Optional
    output_dir = "./mesh_output_advanced"
    
    # Terrain configuration with rotation
    terrain_config = tm.TerrainConfig(
        center_lat=39.71121111,
        center_lon=-7.73483333,
        crop_size_km=25,
        rotation_deg=45,  # Align with prevailing wind direction
        smoothing_sigma=0
    )
    
    # Grid with 3-zone grading (refined center)
    grid_config = tm.GridConfig(
        nx=384,
        ny=384,
        x_grading=[
            (0.35, 0.12, 0.05),   # First 35% of domain, 12% of cells, compress toward center
            (0.30, 0.76, 1.0),    # Middle 30%, 76% of cells, uniform
            (0.35, 0.12, 20.0)    # Last 35%, 12% of cells, expand away from center
        ],
        y_grading=[
            (0.35, 0.12, 0.05),
            (0.30, 0.76, 1.0),
            (0.35, 0.12, 20.0)
        ]
    )
    
    # Mesh with vertical grading for near-ground refinement
    mesh_config = tm.MeshConfig(
        domain_height=3000.0,
        total_z_cells=60,
        z_grading=[
            (0.033, 0.50, 1.0),    # Bottom 100m, 50% of cells, uniform
            (0.967, 0.50, 100.0)   # Remaining height, 50% of cells, expand upward
        ],
        patch_types={
            "ground": "wall",
            "sky": "patch",
            "inlet": "patch",
            "outlet": "patch",
            "sides": "patch"
        }
    )
    
    # Boundary treatment (flatten inlet/outlet)
    boundary_config = tm.BoundaryConfig(
        aoi_fraction=0.3,
        boundary_mode="directional",
        flat_boundary_thickness_fraction=0.08,
        enabled_boundaries=["east", "west"],  # Inlet and outlet
        smoothing_method="mean",
        kernel_progression="exponential",
        base_kernel_size=5,
        progression_rate=10,
        boundary_flatness_mode="blend_target"
    )
    
    # Visualization settings
    viz_config = tm.VisualizationConfig(
        create_plots=True,
        show_grid_lines=True,
        save_high_res=True,
        plot_format="png",
        dpi=150
    )
    
    # Run the pipeline
    pipeline = tm.TerrainMeshPipeline()
    results = pipeline.run(
        dem_path=dem_path,
        rmap_path=roughness_path,
        terrain_config=terrain_config,
        grid_config=grid_config,
        mesh_config=mesh_config,
        boundary_config=boundary_config,
        visualization_config=viz_config,
        output_dir=output_dir
    )
    
    print(f"\n✓ Advanced mesh generated successfully!")
    print(f"  VTK file: {results['vtk_path']}")
    print(f"  blockMeshDict: {results['blockmesh_path']}")
    if results['has_roughness']:
        print(f"  z0 field: {output_dir}/0/include/z0Values")


if __name__ == "__main__":
    main()
