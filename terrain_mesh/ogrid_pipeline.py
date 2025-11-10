"""O-grid pipeline for circular domain terrain meshing"""

from pathlib import Path
from typing import Union, Optional, Dict

from .ogrid_config import OGridConfig
from .ogrid_generator import OGridGenerator
from .ogrid_blockmesh import OGridBlockMeshGenerator


class OGridPipeline:
    """Pipeline for O-grid circular domain mesh generation"""
    
    def __init__(self):
        self.generator = OGridGenerator()
        self.blockmesh_generator = OGridBlockMeshGenerator()
    
    def run(self,
           dem_path: Union[str, Path],
           terrain_config: object,  # TerrainConfig
           ogrid_config: OGridConfig,
           mesh_config: object,  # MeshConfig
           boundary_config: object,  # BoundaryConfig
           visualization_config: Optional[object] = None,
           rmap_path: Optional[Union[str, Path]] = None,
           output_dir: Optional[Union[str, Path]] = None,
           create_blockmesh: bool = True,
           save_metadata: bool = True) -> Dict:
        """
        Run O-grid circular domain pipeline.
        
        Similar to main pipeline but optimized for circular domains.
        """
        
        # Import here to avoid circular dependencies
        from .terrain_processor import TerrainProcessor
        from .boundary_treatment import BoundaryTreatment
        from .visualizer import TerrainVisualizer
        from .utils import write_metadata
        
        if output_dir is None:
            output_dir = Path.cwd() / 'terrain_mesh_output_circular'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Running O-Grid Circular Domain Pipeline")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        
        # Step 1: Extract circular terrain
        print("\n[1/6] Extracting circular terrain...")
        processor = TerrainProcessor()
        
        elevation_data, transform, crs, pixel_res, crop_mask, centre_utm = \
            processor.extract_circular_terrain(dem_path, terrain_config)
        
        roughness_data, roughness_transform = None, None
        if rmap_path:
            print("[1/6] Extracting circular roughness map...")
            roughness_data, roughness_transform = \
                processor.extract_rotated_rmap(rmap_path, terrain_config)
        
        # Step 2: Apply boundary treatment (radial mode)
        print("\n[2/6] Applying radial boundary treatment...")
        boundary_treatment = BoundaryTreatment()
        
        # Force radial mode for circular domain
        boundary_config.boundary_mode = 'radial'
        
        treated_elevation, boundary_elevations, treated_mask, zones = \
            boundary_treatment.process_boundaries(
                elevation_data, crop_mask, boundary_config, 0.0  # No rotation for circular
            )
        
        # Step 3: Generate O-grid
        print("\n[3/6] Generating O-grid structured mesh...")
        grid = self.generator.create_ogrid(
            treated_elevation,
            transform,
            ogrid_config,
            terrain_config,
            boundary_config,
            treated_mask,
            centre_utm
        )
        
        # Step 4: Save VTK output
        print("\n[4/6] Saving VTK mesh...")
        vtk_path = output_dir / 'terrain_ogrid.vtk'
        grid.save(str(vtk_path))
        print(f"VTK mesh saved to: {vtk_path}")

        # Step 5: Generate OpenFOAM outputs
        print("\n[5/6] Generating OpenFOAM files...")

        # Generate blockMeshDict (reads from VTK)
        blockmesh_path = None
        if create_blockmesh:
            blockmesh_path = output_dir / 'system' / 'blockMeshDict'
            
            self.blockmesh_generator.generate_blockMeshDict(
                mesh_config=mesh_config,
                ogrid_config=ogrid_config,
                input_vtk_file=str(vtk_path),  # CHANGED: Pass VTK path
                output_dict_file=str(blockmesh_path)
            )
            
            print(f"  ✓ blockMeshDict saved to: {blockmesh_path}")
        
        # Step 6: Create visualizations
        if visualization_config and visualization_config.create_plots:
            print("\n[6/6] Creating visualization plots...")
            
            visualizer = TerrainVisualizer(visualization_config)
            
            # Use existing visualization methods (they work with radial zones)
            visualizer.create_overview_plots(
                original_dem=elevation_data,
                zones=zones,
                treated_elevation=treated_elevation,
                output_dir=output_dir,
                grid=grid,
                rotation_deg=0.0,  # No rotation for circular
                crop_mask=crop_mask
            )
            
            print("  ✓ Visualization plots created")
        
        # Save metadata
        metadata_path = None
        if save_metadata:
            print("\nSaving pipeline metadata...")
            metadata_path = output_dir / 'pipeline_metadata.json'
            
            # Create metadata dict
            metadata = {
                'domain_type': 'circular_ogrid',
                'dem_path': str(dem_path),
                'output_dir': str(output_dir),
                'vtk_path': str(vtk_path),
                'blockmesh_path': str(blockmesh_path) if blockmesh_path else None,
                'ogrid_config': {
                    'n_sectors': ogrid_config.n_sectors,
                    'nr': ogrid_config.nr,
                    'n_circumferential_per_sector': ogrid_config.n_circumferential_per_sector,
                    'aoi_cell_fraction': ogrid_config.aoi_cell_fraction,
                    'radial_expansion_ratio': ogrid_config.radial_expansion_ratio
                },
                'center_utm': centre_utm,
                'crop_radius_m': terrain_config.crop_size_km * 1000 / 2
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Metadata saved to: {metadata_path}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("O-Grid Pipeline completed successfully!")
        print("=" * 60)
        
        # Return results
        results = {
            'output_dir': str(output_dir),
            'vtk_path': str(vtk_path),
            'blockmesh_path': str(blockmesh_path) if blockmesh_path else None,
            'metadata_path': str(metadata_path) if metadata_path else None,
            'domain_type': 'circular_ogrid',
            'has_roughness': roughness_data is not None
        }
        
        return results