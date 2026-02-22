"""Main pipeline orchestrating the complete terrain-to-mesh workflow.

This module provides the TerrainMeshPipeline class which coordinates all steps
of the mesh generation process from terrain extraction to OpenFOAM export.
"""
import logging
from pathlib import Path
from typing import Union, Optional, Dict
from .config import TerrainConfig, GridConfig, MeshConfig, BoundaryConfig, VisualizationConfig
from .utils import write_metadata
from .terrain_processor import TerrainProcessor
from .boundary_treatment import BoundaryTreatment
from .grid_generator import StructuredGridGenerator
from .visualizer import TerrainVisualizer
from .blockmesh_generator import BlockMeshGenerator

# Configure logging
logger = logging.getLogger(__name__)


class TerrainMeshPipeline:
    """Main pipeline orchestrating the entire terrain-to-mesh process.
    
    This pipeline coordinates the complete workflow:
    1. Extract and process terrain elevation data
    2. Apply boundary treatment and smoothing
    3. Generate structured grid with custom grading
    4. Export to VTK format
    5. Generate OpenFOAM blockMeshDict and z0 field
    6. Create visualization plots
    
    Attributes:
        processor: TerrainProcessor for DEM extraction
        boundary_treatment: BoundaryTreatment for boundary conditioning
        generator: StructuredGridGenerator for mesh creation
        visualizer: TerrainVisualizer for plotting
        blockmesh_generator: BlockMeshGenerator for OpenFOAM export
        metadata: Dictionary storing pipeline metadata
        
    Example:
        >>> pipeline = TerrainMeshPipeline()
        >>> results = pipeline.run(
        ...     dem_path="terrain.tif",
        ...     terrain_config=terrain_cfg,
        ...     grid_config=grid_cfg,
        ...     output_dir="output"
        ... )
    """
    
    def __init__(self):
        self.processor = TerrainProcessor()
        self.boundary_treatment = BoundaryTreatment()
        self.generator = StructuredGridGenerator()
        self.visualizer = TerrainVisualizer()
        self.blockmesh_generator = BlockMeshGenerator()
        self.metadata = {}
        logger.info("TerrainMeshPipeline initialized")
    
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
        """Run the complete terrain-to-mesh pipeline.
        
        Args:
            dem_path: Path to Digital Elevation Model file (GeoTIFF, DAT, or NetCDF)
            terrain_config: Configuration for terrain extraction and processing
            grid_config: Configuration for structured grid generation
            mesh_config: Optional configuration for mesh and OpenFOAM output
            boundary_config: Optional configuration for boundary treatment
            visualization_config: Optional configuration for plotting
            rmap_path: Optional path to roughness map for z0 field generation
            output_dir: Output directory for all generated files
            create_blockmesh: If True, generate OpenFOAM blockMeshDict
            save_metadata: If True, save pipeline metadata to JSON
            
        Returns:
            Dictionary containing paths to generated files:
            - output_dir: Path to output directory
            - vtk_path: Path to VTK mesh file
            - blockmesh_path: Path to blockMeshDict (if created)
            - metadata_path: Path to metadata JSON (if saved)
            - has_roughness: Boolean indicating if roughness data was processed
            
        Raises:
            FileNotFoundError: If dem_path doesn't exist
            ValueError: If configuration parameters are invalid
        """
        
        # Setup configs and output directory
        mesh_config = mesh_config or MeshConfig()
        boundary_config = boundary_config or BoundaryConfig()
        visualization_config = visualization_config or VisualizationConfig()
        
        if output_dir is None:
            output_dir = Path.cwd() / 'terrain_mesh_output'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Running Terrain Following Mesh Generation Pipeline")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        
        # For backward compatibility, also print to console
        print("=" * 60)
        print("Running Terrain Following Mesh Generation Pipeline")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        
        # Step 1: Extract terrain and roughness data
        logger.info("[1/6] Extracting terrain elevation...")
        print("\n[1/6] Extracting terrain elevation...")
        elevation_data, transform, crs, pixel_res, crop_mask, centre_utm = \
            self.processor.extract_rotated_terrain(dem_path, terrain_config)
        
        roughness_data, roughness_transform = None, None
        if rmap_path:
            logger.info("[1/6] Extracting roughness map...")
            print("[1/6] Extracting roughness map...")
            roughness_data, roughness_transform = \
                self.processor.extract_rotated_rmap(rmap_path, terrain_config)
        
        # Step 2: Apply boundary treatment
        logger.info("[2/6] Applying boundary treatment...")
        print("\n[2/6] Applying boundary treatment...")
        treated_elevation, boundary_elevations, treated_mask, zones = \
            self.boundary_treatment.process_boundaries(
                elevation_data, crop_mask, boundary_config, terrain_config.rotation_deg
            )
        
        # Step 3: Generate structured grid
        logger.info("[3/6] Generating structured grid...")
        print("\n[3/6] Generating structured grid...")
        grid = self.generator.create_grid(
            treated_elevation, 
            transform, 
            grid_config, 
            terrain_config, 
            treated_mask,
            centre_utm
        )
        
        # Step 4: Save VTK output
        logger.info("[4/6] Saving VTK mesh...")
        print("\n[4/6] Saving VTK mesh...")
        vtk_path = output_dir / 'terrain_structured.vtk'
        grid.save(str(vtk_path))
        logger.info(f"VTK mesh saved to: {vtk_path}")
        print(f"VTK mesh saved to: {vtk_path}")
        
        # Step 5: Generate OpenFOAM outputs
        logger.info("[5/6] Generating OpenFOAM files...")
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
            logger.info(f"z0 field saved with {z0_stats['n_faces']} faces")
            print(f"  ✓ z0 field saved with {z0_stats['n_faces']} faces")
        
        # Generate blockMeshDict if requested
        blockmesh_path = None
        if create_blockmesh:
            blockmesh_path = output_dir / 'system' / 'blockMeshDict'
            inletFaceInfo_path = output_dir / '0' / 'include' / 'inletFaceInfo.txt'
            self.blockmesh_generator.generate_blockMeshDict(
                mesh_config, str(vtk_path), str(blockmesh_path), str(inletFaceInfo_path),
                roughness_data, roughness_transform
            )
            logger.info(f"blockMeshDict saved to: {blockmesh_path}")
            logger.info(f"inletFaceInfo saved to: {inletFaceInfo_path}")
            print(f"  ✓ blockMeshDict saved to: {blockmesh_path}")
            print(f"  ✓ inletFaceInfo saved to: {inletFaceInfo_path}")
        
        # Step 6: Create visualizations
        if visualization_config.create_plots:
            logger.info("[6/6] Creating visualization plots...")
            print("\n[6/6] Creating visualization plots...")
            
            self.visualizer = TerrainVisualizer(visualization_config)
            
            # Terrain overview plots
            self.visualizer.create_overview_plots(
                                                    original_dem=elevation_data,
                                                    zones=zones,
                                                    treated_elevation=treated_elevation,
                                                    output_dir=output_dir,
                                                    grid=grid,
                                                    rotation_deg=terrain_config.rotation_deg,
                                                    crop_mask=crop_mask
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
            
            logger.info("Visualization plots created")
            print("  ✓ Visualization plots created")
        
        # Save metadata
        metadata_path = None
        if save_metadata:
            logger.info("Saving pipeline metadata...")
            print("\nSaving pipeline metadata...")
            metadata_path = output_dir / 'pipeline_metadata.json'
            write_metadata(
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
            logger.info(f"Metadata saved to: {metadata_path}")
            print(f"  ✓ Metadata saved to: {metadata_path}")
        
        # Final summary
        logger.info("Pipeline completed successfully!")
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