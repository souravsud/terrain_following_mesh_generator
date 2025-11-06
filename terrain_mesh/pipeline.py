"""Main pipeline orchestrating the complete terrain-to-mesh workflow"""

import json
from pathlib import Path
from typing import Union, Optional, Dict
from datetime import datetime
import numpy as np

from .config import TerrainConfig, GridConfig, MeshConfig, BoundaryConfig, VisualizationConfig
from .terrain_processor import TerrainProcessor
from .boundary_treatment import BoundaryTreatment
from .grid_generator import StructuredGridGenerator
from .visualizer import TerrainVisualizer
from .blockmesh_generator import BlockMeshGenerator

class TerrainMeshPipeline:
    """Main pipeline orchestrating the entire terrain-to-mesh process"""
    
    def __init__(self):
        self.processor = TerrainProcessor()
        self.boundary_treatment = BoundaryTreatment()
        self.generator = StructuredGridGenerator()
        self.visualizer = TerrainVisualizer()
        self.blockmesh_generator = BlockMeshGenerator()
        self.metadata = {}
    
    def run(self, 
           dem_path: Union[str, Path],
           terrain_config: TerrainConfig, 
           grid_config: GridConfig,
           mesh_config: Optional[MeshConfig] = None,
           boundary_config: Optional[BoundaryConfig] = None,
           visualization_config: Optional[VisualizationConfig] = None,
           rmap_path: Optional[Union[str, Path]] = None,
           output_dir: Optional[Union[str, Path]] = None, 
           create_blockmesh: bool = True,
           save_metadata: bool = True) -> Dict:
        """Run the complete terrain-to-mesh pipeline"""
        
        # Setup configs and output directory
        mesh_config = mesh_config or MeshConfig()
        boundary_config = boundary_config or BoundaryConfig()
        visualization_config = visualization_config or VisualizationConfig()
        
        if output_dir is None:
            output_dir = Path.cwd() / 'terrain_mesh_output'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Running Terrain Following Mesh Generation Pipeline")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        
        # Step 1: Extract terrain and roughness data
        print("\n[1/6] Extracting terrain elevation...")
        elevation_data, transform, crs, pixel_res, crop_mask = \
            self.processor.extract_rotated_terrain(dem_path, terrain_config)
        
        roughness_data, roughness_transform = None, None
        if rmap_path:
            print("[1/6] Extracting roughness map...")
            roughness_data, roughness_transform = \
                self.processor.extract_rotated_rmap(rmap_path, terrain_config)
        
        # Step 2: Apply boundary treatment
        print("\n[2/6] Applying boundary treatment...")
        treated_elevation, boundary_elevations, treated_mask, zones = \
            self.boundary_treatment.process_boundaries(
                elevation_data, crop_mask, boundary_config, terrain_config.rotation_deg
            )
        
        # Step 3: Generate structured grid
        print("\n[3/6] Generating structured grid...")
        grid = self.generator.create_grid(
            treated_elevation, 
            transform, 
            grid_config, 
            terrain_config, 
            treated_mask
        )
        
        # Step 4: Save VTK output
        print("\n[4/6] Saving VTK mesh...")
        vtk_path = output_dir / 'terrain_structured.vtk'
        grid.save(str(vtk_path))
        print(f"VTK mesh saved to: {vtk_path}")
        
        # Step 5: Generate OpenFOAM outputs
        print("\n[5/6] Generating OpenFOAM files...")
        
        # Generate z0 field if roughness map provided
        z0_stats = None
        if roughness_data is not None:
            z0_file = output_dir / '0' / 'include' / 'z0Values'
            z0_stats = self.blockmesh_generator.generate_z0_field(
                vtk_file=str(vtk_path),
                roughness_data=roughness_data,
                roughness_transform=roughness_transform,
                output_file=str(z0_file),
                default_z0=0.1
            )
            print(f"  ✓ z0 field saved with {z0_stats['n_faces']} faces")
        
        # Generate blockMeshDict if requested
        blockmesh_path = None
        if create_blockmesh:
            blockmesh_path = output_dir / 'system' / 'blockMeshDict'
            inletFaceInfo_path = output_dir / '0' / 'include' / 'inletFaceInfo.txt'
            self.blockmesh_generator.generate_blockMeshDict(
                mesh_config, str(vtk_path), str(blockmesh_path), str(inletFaceInfo_path)
            )
            print(f"  ✓ blockMeshDict saved to: {blockmesh_path}")
            print(f"  ✓ inletFaceInfo saved to: {inletFaceInfo_path}")
        
        # Step 6: Create visualizations
        if visualization_config.create_plots:
            print("\n[6/6] Creating visualization plots...")
            
            self.visualizer = TerrainVisualizer(visualization_config)
            
            # Terrain overview plots
            self.visualizer.create_overview_plots(
                original_dem=elevation_data,
                zones=zones,
                treated_elevation=treated_elevation,
                final_elevation=treated_elevation,
                output_dir=output_dir,
                pv_grid=grid
            )
            
            # Roughness plots if available
            if roughness_data is not None and z0_stats is not None:
                self.visualizer.create_roughness_plots(
                    roughness_data, 
                    roughness_transform, 
                    z0_stats, 
                    output_dir, 
                    str(vtk_path)
                )
            
            print("  ✓ Visualization plots created")
        
        # Save metadata
        metadata_path = None
        if save_metadata:
            print("\nSaving pipeline metadata...")
            metadata_path = output_dir / 'pipeline_metadata.json'
            self._save_metadata(
                dem_path=dem_path,
                rmap_path=rmap_path,
                terrain_config=terrain_config,
                grid_config=grid_config,
                mesh_config=mesh_config,
                boundary_config=boundary_config,
                visualization_config=visualization_config,
                elevation_data=elevation_data,
                treated_elevation=treated_elevation,
                transform=transform,
                crs=crs,
                pixel_res=pixel_res,
                grid=grid,
                vtk_path=vtk_path,
                blockmesh_path=blockmesh_path,
                output_dir=output_dir,
                metadata_path=metadata_path
            )
            print(f"  ✓ Metadata saved to: {metadata_path}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
        # Return results dictionary
        results = {
            'output_dir': str(output_dir),
            'vtk_path': str(vtk_path),
            'blockmesh_path': str(blockmesh_path) if blockmesh_path else None,
            'metadata_path': str(metadata_path) if metadata_path else None,
            'has_roughness': roughness_data is not None
        }
        
        return results
    
    def _save_metadata(self, **kwargs):
        """Save pipeline metadata to JSON file"""
        
        metadata = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "version": "2.0",
                "pipeline_class": self.__class__.__name__
            },
            
            "input_files": {
                "dem_path": str(kwargs['dem_path']),
                "roughness_path": str(kwargs['rmap_path']) if kwargs['rmap_path'] else None
            },
            
            "output_files": {
                "output_directory": str(kwargs['output_dir']),
                "vtk_mesh": str(kwargs['vtk_path']),
                "blockmesh_dict": str(kwargs['blockmesh_path']) if kwargs['blockmesh_path'] else None,
                "metadata_file": str(kwargs['metadata_path'])
            },
            
            "configurations": {
                "terrain": {
                    "center_lat": kwargs['terrain_config'].center_lat,
                    "center_lon": kwargs['terrain_config'].center_lon,
                    "center_utm": kwargs['terrain_config'].center_coordinates,
                    "crop_size_km": kwargs['terrain_config'].crop_size_km,
                    "rotation_deg": kwargs['terrain_config'].rotation_deg,
                    "smoothing_sigma": kwargs['terrain_config'].smoothing_sigma
                },
                
                "grid": {
                    "nx": kwargs['grid_config'].nx,
                    "ny": kwargs['grid_config'].ny,
                    "x_grading": kwargs['grid_config'].x_grading,
                    "y_grading": kwargs['grid_config'].y_grading
                },
                
                "mesh": {
                    "domain_height": kwargs['mesh_config'].domain_height,
                    "total_z_cells": kwargs['mesh_config'].total_z_cells,
                    "z_grading": kwargs['mesh_config'].z_grading,
                    "patch_types": kwargs['mesh_config'].patch_types
                } if kwargs['mesh_config'] else None,
                
                "boundary": {
                    "aoi_fraction": kwargs['boundary_config'].aoi_fraction,
                    "boundary_mode": kwargs['boundary_config'].boundary_mode,
                    "flat_boundary_thickness_fraction": kwargs['boundary_config'].flat_boundary_thickness_fraction,
                    "enabled_boundaries": kwargs['boundary_config'].enabled_boundaries,
                    "smoothing_method": kwargs['boundary_config'].smoothing_method,
                    "kernel_progression": kwargs['boundary_config'].kernel_progression,
                    "base_kernel_size": kwargs['boundary_config'].base_kernel_size,
                    "max_kernel_size": kwargs['boundary_config'].max_kernel_size,
                    "progression_rate": kwargs['boundary_config'].progression_rate,
                    "boundary_flatness_mode": kwargs['boundary_config'].boundary_flatness_mode,
                    "uniform_elevation": kwargs['boundary_config'].uniform_elevation
                },
                
                "visualization": {
                    "create_plots": kwargs['visualization_config'].create_plots,
                    "show_grid_lines": kwargs['visualization_config'].show_grid_lines,
                    "save_high_res": kwargs['visualization_config'].save_high_res,
                    "plot_format": kwargs['visualization_config'].plot_format,
                    "dpi": kwargs['visualization_config'].dpi
                }
            },
            
            "processing_results": {
                "coordinate_system": {
                    "crs": str(kwargs['crs']),
                    "pixel_resolution": kwargs['pixel_res'],
                    "transform": list(kwargs['transform']) if hasattr(kwargs['transform'], '__iter__') else str(kwargs['transform'])
                },
                
                "elevation_statistics": {
                    "original": self._get_array_stats(kwargs['elevation_data']),
                    "treated": self._get_array_stats(kwargs['treated_elevation'])
                },
                
                "grid_statistics": {
                    "number_of_points": kwargs['grid'].GetNumberOfPoints() if hasattr(kwargs['grid'], 'GetNumberOfPoints') else None,
                    "number_of_cells": kwargs['grid'].GetNumberOfCells() if hasattr(kwargs['grid'], 'GetNumberOfCells') else None,
                    "bounds": list(kwargs['grid'].GetBounds()) if hasattr(kwargs['grid'], 'GetBounds') else None
                }
            }
        }
        
        # Save to file
        with open(kwargs['metadata_path'], 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _get_array_stats(self, data: np.ndarray) -> dict:
        """Helper to extract statistics from numpy array"""
        return {
            "shape": list(data.shape),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data))
        }