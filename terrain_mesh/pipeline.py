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
    
    def run(self, dem_path: Union[str, Path], 
           terrain_config: TerrainConfig, 
           grid_config: GridConfig,
           mesh_config: Optional[MeshConfig] = None,
           boundary_config: Optional[BoundaryConfig] = None,
           visualization_config: Optional[VisualizationConfig] = None,
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
        
        # Step 1: Load and prepare DEM
        print("\n[1/5] Loading and preparing DEM...")
        utm_dem_path = self.processor.load_and_prepare_dem(dem_path)

        # Step 2: Extract terrain  
        print("\n[2/5] Extracting rotated terrain...")
        elevation_data, transform, crs, pixel_res, crop_mask = self.processor.extract_rotated_terrain(utm_dem_path, terrain_config)

        # Step 3: Apply boundary treatment
        print("\n[3/5] Applying boundary treatment...")
        treated_elevation, boundary_elevations, treated_mask, zones = self.boundary_treatment.process_boundaries(
            elevation_data, crop_mask, boundary_config, terrain_config.rotation_deg)
        
        # Step 4: Generate structured grid
        print("\n[4/5] Generating structured grid...")
        grid = self.generator.create_grid(
                                            treated_elevation, 
                                            transform, 
                                            grid_config, 
                                            terrain_config, 
                                            treated_mask
                                        )
        
        # Step 5: Save outputs and create visualizations
        print("\n[5/5] Saving outputs...")
        vtk_path = output_dir / 'terrain_structured.vtk'
        grid.save(str(vtk_path))
        
        if visualization_config.create_plots:
            # Store original DEM for visualization
            original_dem = elevation_data.copy()  # Before boundary treatment
            
            self.visualizer = TerrainVisualizer(visualization_config)
            self.visualizer.create_overview_plots(
                original_dem,      # Before processing
                zones,            # Zone masks  
                treated_elevation, # After boundary processing
                treated_elevation, # Final output (same for now)
                output_dir,
                grid
            )
        
        # Generate blockMeshDict if requested
        blockmesh_path = None
        if create_blockmesh and mesh_config:
            blockmesh_path = output_dir / 'system' / 'blockMeshDict'
            inletFaceInfo_path = output_dir / '0' / 'include'/ 'inletFaceInfo.txt'
            self.blockmesh_generator.generate_blockMeshDict(mesh_config, str(vtk_path), str(blockmesh_path), str(inletFaceInfo_path))
        
        # Save metadata
        if save_metadata:
            metadata_path = output_dir / 'pipeline_metadata.json'
            self._save_metadata(
                dem_path=dem_path,
                utm_dem_path=utm_dem_path,
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
            print(f"Metadata saved to: {metadata_path}")
        
        # Return results dictionary
        results = {
            'message': 'Pipeline structure ready - integrate your original functions',
            'output_dir': str(output_dir),
            'vtk_path': str(vtk_path),
            'metadata_path': str(metadata_path) if save_metadata else None
        }
        
        return results
    
    def _save_metadata(self, **kwargs):
        """Save pipeline metadata to JSON file"""
        
        # Extract data for easier reference
        elevation_data = kwargs['elevation_data']
        treated_elevation = kwargs['treated_elevation']
        transform = kwargs['transform']
        crs = kwargs['crs']
        grid = kwargs['grid']
        
        metadata = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",  # Add your pipeline version
                "pipeline_class": self.__class__.__name__
            },
            
            "input_files": {
                "original_dem": str(kwargs['dem_path']),
                "utm_dem": str(kwargs['utm_dem_path'])
            },
            
            "output_files": {
                "output_directory": str(kwargs['output_dir']),
                "vtk_mesh": str(kwargs['vtk_path']),
                "blockmesh_dict": str(kwargs['blockmesh_path']) if kwargs['blockmesh_path'] else None,
                "metadata_file": str(kwargs['metadata_path'])
            },
            
            "terrain_config": {
                "center_lat": kwargs['terrain_config'].center_lat,
                "center_lon": kwargs['terrain_config'].center_lon,
                "crop_size_km": kwargs['terrain_config'].crop_size_km,
                "rotation_deg": kwargs['terrain_config'].rotation_deg,
                "smoothing_sigma": kwargs['terrain_config'].smoothing_sigma,
                "center_coordinates": kwargs['terrain_config'].center_coordinates
            },
            
            "grid_config": {
                "nx": kwargs['grid_config'].nx,
                "ny": kwargs['grid_config'].ny,
                "x_grading": kwargs['grid_config'].x_grading,
                "y_grading": kwargs['grid_config'].y_grading
            },
            
            "mesh_config": {
                "domain_height": kwargs['mesh_config'].domain_height,
                "num_cells_z": kwargs['mesh_config'].num_cells_z,
                "expansion_ratio_z": kwargs['mesh_config'].expansion_ratio_z,
                "patch_types": kwargs['mesh_config'].patch_types
            } if kwargs['mesh_config'] else None,
            
            "boundary_config": {
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
            
            "visualization_config": {
                "create_plots": kwargs['visualization_config'].create_plots,
                "show_grid_lines": kwargs['visualization_config'].show_grid_lines,
                "save_high_res": kwargs['visualization_config'].save_high_res,
                "plot_format": kwargs['visualization_config'].plot_format,
                "dpi": kwargs['visualization_config'].dpi
            },
            
            "processing_results": {
                "pixel_resolution": kwargs['pixel_res'],
                "coordinate_reference_system": str(crs),
                "spatial_transform": {
                    "geotransform": list(transform) if hasattr(transform, '__iter__') else str(transform)
                },
                
                "elevation_statistics": {
                    "original": {
                        "shape": list(elevation_data.shape),
                        "min_elevation": float(np.min(elevation_data)),
                        "max_elevation": float(np.max(elevation_data)),
                        "mean_elevation": float(np.mean(elevation_data)),
                        "std_elevation": float(np.std(elevation_data))
                    },
                    "treated": {
                        "shape": list(treated_elevation.shape),
                        "min_elevation": float(np.min(treated_elevation)),
                        "max_elevation": float(np.max(treated_elevation)),
                        "mean_elevation": float(np.mean(treated_elevation)),
                        "std_elevation": float(np.std(treated_elevation))
                    }
                },
                
                "grid_statistics": {
                    "number_of_points": grid.GetNumberOfPoints() if hasattr(grid, 'GetNumberOfPoints') else None,
                    "number_of_cells": grid.GetNumberOfCells() if hasattr(grid, 'GetNumberOfCells') else None,
                    # Add bounds if available
                    "bounds": list(grid.GetBounds()) if hasattr(grid, 'GetBounds') else None
                }
            }
        }
        
        # Save to file
        with open(kwargs['metadata_path'], 'w') as f:
            json.dump(metadata, f, indent=2, default=str)