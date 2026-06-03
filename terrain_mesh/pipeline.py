"""Main pipeline orchestrating the complete terrain-to-mesh workflow.

This module provides the TerrainMeshPipeline class which coordinates all steps
of the mesh generation process from terrain extraction to OpenFOAM export.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict
from .config import TerrainConfig, GridConfig, MeshConfig, BoundaryConfig, VisualizationConfig
from .utils import write_metadata, build_roughness_interpolator
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
    4. Save terrain (and roughness) maps to maps/ as NPZ files
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
        logger.debug("Components: TerrainProcessor, BoundaryTreatment, StructuredGridGenerator, TerrainVisualizer, BlockMeshGenerator")
    
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
            - terrain_map_path: Path to terrain elevation NPZ map in maps/
            - roughness_map_path: Path to roughness NPZ map in maps/ (if roughness provided)
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
        
        # Step 1: Extract terrain and roughness data
        logger.info("[1/6] Extracting terrain elevation...")
        elevation_data, min_elevation, transform, crs, pixel_res, crop_mask, centre_utm = \
            self.processor.extract_rotated_terrain(dem_path, terrain_config)
        
        roughness_data, roughness_transform = None, None
        roughness_is_constant = False
        if rmap_path:
            logger.info("[1/6] Extracting roughness map...")
            roughness_data, roughness_transform = \
                self.processor.extract_rotated_rmap(rmap_path, terrain_config)
        
        if mesh_config.adjust_ceiling_for_terrain and min_elevation >= 0.0:
            logger.info("[1/6] Adjusting domain ceiling for terrain altitude...")
            mesh_config.domain_height = mesh_config.domain_height + min_elevation
        
        # Step 2: Apply boundary treatment
        logger.info("[2/6] Applying boundary treatment...")
        treated_elevation, boundary_elevations, treated_mask, zones = \
            self.boundary_treatment.process_boundaries(
                elevation_data, crop_mask, boundary_config, terrain_config.rotation_deg
            )
        
        # Step 3: Generate structured grid
        logger.info("[3/6] Generating structured grid...")
        grid = self.generator.create_grid(
            treated_elevation, 
            transform, 
            grid_config, 
            terrain_config, 
            treated_mask,
            centre_utm
        )
        
        # Step 4: Save ML-ready terrain and roughness maps to maps/ folder
        logger.info("[4/6] Saving terrain maps...")

        # When no roughness map was supplied, synthesise a constant raster so that
        # the ML training dataset always contains a paired roughness_map.npz file.
        if roughness_data is None:
            roughness_is_constant = True
            logger.info(
                f"[4/6] No roughness map provided — synthesising constant "
                f"roughness map (z0={mesh_config.default_z0:.4f} m)..."
            )
            roughness_data, roughness_transform = self._make_constant_roughness(
                grid=grid,
                center_coordinates=terrain_config.center_coordinates,
                centre_utm=centre_utm,
                z0_value=mesh_config.default_z0,
            )

        terrain_map_path, roughness_map_path = self._save_maps(
            grid=grid,
            roughness_data=roughness_data,
            roughness_transform=roughness_transform,
            centre_utm=centre_utm,
            center_coordinates=terrain_config.center_coordinates,
            output_dir=output_dir,
        )
        
        # Step 5: Generate OpenFOAM outputs
        logger.info("[5/6] Generating OpenFOAM files...")
        
        # Generate z0 field (roughness_data is always set at this point, either
        # from a real rmap_path or from the synthesised constant raster above).
        z0_file = output_dir / '0' / 'include' / 'z0Values'
        z0_stats = self.blockmesh_generator.generate_z0_field(
            terrain_map=str(terrain_map_path),
            roughness_data=roughness_data,
            roughness_transform=roughness_transform,
            output_file=str(z0_file),
            default_z0=mesh_config.default_z0
        )
        logger.info(f"z0 field saved with {z0_stats['n_faces']} faces")
        
        # Generate blockMeshDict if requested
        blockmesh_path = None
        if create_blockmesh:
            blockmesh_path = output_dir / 'system' / 'blockMeshDict'
            inletFaceInfo_path = output_dir / '0' / 'include' / 'inletFaceInfo.txt'
            self.blockmesh_generator.generate_blockMeshDict(
                mesh_config, str(terrain_map_path), str(blockmesh_path), str(inletFaceInfo_path),
                roughness_data, roughness_transform
            )
            logger.info(f"blockMeshDict saved to: {blockmesh_path}")
            logger.debug(f"inletFaceInfo saved to: {inletFaceInfo_path}")
        
        # Step 6: Create visualizations
        if visualization_config.create_plots:
            logger.info("[6/6] Creating visualization plots...")
            
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
            
            # Roughness plots (always available — constant raster is used when no rmap_path)
            self.visualizer.create_roughness_plots(
                roughness_data, 
                roughness_transform, 
                z0_stats, 
                output_dir, 
                str(terrain_map_path)
            )
            
            logger.debug("Visualization plots created")
        
        # Save metadata
        metadata_path = None
        if save_metadata:
            logger.debug("Saving pipeline metadata...")
            metadata_path = output_dir / 'pipeline_metadata.json'
            write_metadata(
                dem_path=dem_path,
                rmap_path=rmap_path,
                terrain_config=terrain_config,
                grid_config=grid_config,
                mesh_config=mesh_config,
                boundary_config=boundary_config,
                boundary_elevations=boundary_elevations,
                visualization_config=visualization_config,
                elevation_data=elevation_data,
                treated_elevation=treated_elevation,
                transform=transform,
                crs=crs,
                min_elevation=min_elevation,
                centre_utm=centre_utm,
                pixel_res=pixel_res,
                grid=grid,
                terrain_map_path=terrain_map_path,
                blockmesh_path=blockmesh_path,
                output_dir=output_dir,
                metadata_path=metadata_path
            )
            logger.debug(f"Metadata saved to: {metadata_path}")
        
        # Final summary
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
        # Return results dictionary
        results = {
            'output_dir': str(output_dir),
            'terrain_map_path': str(terrain_map_path),
            'roughness_map_path': str(roughness_map_path) if roughness_map_path else None,
            'blockmesh_path': str(blockmesh_path) if blockmesh_path else None,
            'metadata_path': str(metadata_path) if metadata_path else None,
            'has_roughness': True,  # roughness_map.npz is always written
            'roughness_is_constant': roughness_is_constant,
        }
        
        return results

    def _save_maps(self, grid, roughness_data, roughness_transform, centre_utm,
                   center_coordinates, output_dir):
        """Save terrain elevation and roughness maps as NPZ files for ML use.

        Creates a ``maps/`` folder inside *output_dir* and writes:

        * ``terrain_map.npz`` – structured surface mesh elevation with arrays:

          - ``elevation``, ``x``, ``y`` — vertex-grid arrays of shape ``(ny, nx)``,
            used directly by the OpenFOAM blockMeshDict generation step.
          - ``elevation_cc``, ``x_cc``, ``y_cc`` — cell-centre arrays of shape
            ``(ny-1, nx-1)``, computed by averaging the four surrounding vertices.
            These correspond to the values OpenFOAM reports at cell centres and
            are provided for post-processing pipelines (Fix 3).

        * ``roughness_map.npz`` – roughness length (z0) interpolated to the same
          grid, with arrays:

          - ``z0``, ``x``, ``y`` — vertex-grid arrays of shape ``(ny, nx)``.
          - ``z0_cc``, ``x_cc``, ``y_cc`` — cell-centre arrays of shape
            ``(ny-1, nx-1)``.

          Only written when *roughness_data* is provided.

        Args:
            grid: PyVista StructuredGrid produced by :class:`StructuredGridGenerator`.
            roughness_data: Optional 2-D numpy array of z0 values (may contain
                NaN outside the rotated crop region).
            roughness_transform: Affine transform mapping pixel indices to UTM
                coordinates for *roughness_data*.
            centre_utm: ``(x, y)`` UTM coordinates of the terrain centre used
                when *center_coordinates* is ``True``.
            center_coordinates: Whether the grid's X/Y values are centred at
                the terrain centre rather than being absolute UTM coordinates.
            output_dir: :class:`~pathlib.Path` of the pipeline output directory.

        Returns:
            Tuple ``(terrain_map_path, roughness_map_path)`` where
            *roughness_map_path* is ``None`` when no roughness data was provided.
        """
        maps_dir = output_dir / 'maps'
        maps_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Extract structured grid arrays – shape (ny, nx)                     #
        # Grid dimensions are (nx, ny, 1); points flattened in row-major order
        # ------------------------------------------------------------------ #
        nx, ny, _ = grid.dimensions
        points = grid.points.reshape((ny, nx, 3))
        X = points[:, :, 0]
        Y = points[:, :, 1]
        Z = points[:, :, 2]

        # Fix 3: Pre-compute cell-centre coordinates and elevation by averaging the
        # four surrounding vertex values.  Shape: (ny-1, nx-1).  These match the
        # cell-centre values used by OpenFOAM and eliminate the need for any
        # interpolation in post-processing pipelines.
        Z_cc = 0.25 * (Z[:-1, :-1] + Z[:-1, 1:] + Z[1:, :-1] + Z[1:, 1:])
        X_cc = 0.25 * (X[:-1, :-1] + X[:-1, 1:] + X[1:, :-1] + X[1:, 1:])
        Y_cc = 0.25 * (Y[:-1, :-1] + Y[:-1, 1:] + Y[1:, :-1] + Y[1:, 1:])

        # Terrain elevation map (no interpolation – direct read from grid).
        # Vertex arrays (elevation, x, y) have shape (ny, nx) and are used by the
        # blockMeshDict generator downstream.  Cell-centre arrays (*_cc) have shape
        # (ny-1, nx-1) and are provided for post-processing pipelines.
        terrain_map_path = maps_dir / 'terrain_map.npz'
        np.savez_compressed(
            terrain_map_path,
            elevation=Z, x=X, y=Y,
            elevation_cc=Z_cc, x_cc=X_cc, y_cc=Y_cc,
        )
        logger.debug(f"Terrain map saved to: {terrain_map_path}")

        roughness_map_path = None

        if roughness_data is not None and roughness_transform is not None:
            # Convert centred grid coordinates back to absolute UTM for lookup
            if center_coordinates:
                X_utm = X + centre_utm[0]
                Y_utm = Y + centre_utm[1]
            else:
                X_utm = X
                Y_utm = Y

            # Build shared interpolator (NaN-fill + bilinear) and query at
            # every grid point – same approach as generate_z0_field but at
            # grid vertices rather than cell-face centres.
            interpolator = build_roughness_interpolator(
                roughness_data, roughness_transform, default_z0=np.nan
            )
            query_points = np.column_stack((Y_utm.ravel(), X_utm.ravel()))
            z0_flat = interpolator(query_points)

            # Apply minimum roughness only where elevation is defined.
            # np.fmax is used instead of np.maximum so that any NaN returned by
            # the interpolator for points that fall outside the roughness raster
            # bounding box is treated as 0 (i.e. resolved to 0.0002) rather than
            # propagating NaN into the saved map.
            valid_elev = ~np.isnan(Z.ravel())
            z0_flat[valid_elev] = np.fmax(z0_flat[valid_elev], 0.0002)
            z0_flat[~valid_elev] = np.nan  # preserve NaN outside terrain

            Z0_grid = z0_flat.reshape((ny, nx))

            roughness_map_path = maps_dir / 'roughness_map.npz'
            # Cell-centre roughness: NaN propagates naturally when any vertex is NaN
            Z0_cc = 0.25 * (
                Z0_grid[:-1, :-1] + Z0_grid[:-1, 1:]
                + Z0_grid[1:, :-1] + Z0_grid[1:, 1:]
            )
            np.savez_compressed(
                roughness_map_path,
                z0=Z0_grid, x=X, y=Y,
                z0_cc=Z0_cc, x_cc=X_cc, y_cc=Y_cc,
            )
            logger.debug(f"Roughness map saved to: {roughness_map_path}")

        return terrain_map_path, roughness_map_path

    def _make_constant_roughness(
        self,
        grid,
        center_coordinates: bool,
        centre_utm,
        z0_value: float,
    ):
        """Synthesise a constant roughness raster that covers the structured grid extent.

        When no roughness map file is available (``rmap_path`` is ``None``), this
        method creates a 3×3 raster filled with *z0_value* whose pixel-origin
        coordinates (used by :func:`~terrain_mesh.utils.build_roughness_interpolator`)
        span the full terrain grid extent plus a small buffer.  This ensures every
        grid query point falls strictly inside the interpolator's coverage and
        receives the requested constant value rather than the out-of-bounds fallback.

        The raster is functionally identical to any real raster from the perspective
        of :func:`~terrain_mesh.utils.build_roughness_interpolator` and allows the
        full pipeline (NPZ save, z0Values, inletFaceInfo, visualisation) to produce
        consistent output for ML training datasets even when only a single constant
        roughness value is known.

        Args:
            grid: PyVista StructuredGrid produced by :class:`StructuredGridGenerator`.
            center_coordinates: Whether grid X/Y values are centred at
                *centre_utm* rather than being absolute UTM coordinates.
            centre_utm: ``(x, y)`` UTM coordinates of the terrain centre (used
                when *center_coordinates* is ``True``).
            z0_value: Constant roughness length (m) to fill the raster with.
                Values below 0.0002 are clipped to 0.0002 (minimum physically
                meaningful roughness, corresponding to open water).

        Returns:
            Tuple ``(roughness_data, roughness_transform)`` ready to be consumed
            by :meth:`_save_maps` and :meth:`~terrain_mesh.blockmesh_generator.BlockMeshGenerator.generate_z0_field`.
        """
        from rasterio.transform import Affine as _Affine

        z0_value = max(float(z0_value), 0.0002)

        nx_grid, ny_grid, _ = grid.dimensions
        points = grid.points.reshape((ny_grid, nx_grid, 3))
        X = points[:, :, 0]
        Y = points[:, :, 1]

        # Convert centred coordinates back to absolute UTM if required
        if center_coordinates:
            X_utm = X + centre_utm[0]
            Y_utm = Y + centre_utm[1]
        else:
            X_utm = X
            Y_utm = Y

        x_min, x_max = float(X_utm.min()), float(X_utm.max())
        y_min, y_max = float(Y_utm.min()), float(Y_utm.max())

        # Add a 1 % buffer so every grid vertex is strictly inside coverage.
        buf_x = max((x_max - x_min) * 0.01, 1.0)
        buf_y = max((y_max - y_min) * 0.01, 1.0)
        x_min -= buf_x
        x_max += buf_x
        y_min -= buf_y
        y_max += buf_y

        # build_roughness_interpolator uses pixel-origin coordinates:
        #   x_coords = np.arange(ncols) * x_res + transform.c   (origins)
        #   y_coords = np.arange(nrows) * (-y_res) + transform.f (descending origins)
        # For a 3×3 raster the last origin is at index 2, covering only 2/3 of
        # the domain when from_bounds is used.  Instead, set the pixel spacing
        # so that index-0 and index-2 origins land exactly at x_min and x_max:
        #   x_res = (x_max - x_min) / (ncols - 1)   →   x_coords = [x_min, mid, x_max]
        nrows, ncols = 3, 3
        x_res = (x_max - x_min) / (ncols - 1)
        y_res = (y_max - y_min) / (nrows - 1)

        roughness_data = np.full((nrows, ncols), z0_value, dtype=np.float32)
        # Affine(a, b, c, d, e, f):  x_pixel=a, y_pixel=e (<0 north-up), origin=(c, f)
        roughness_transform = _Affine(x_res, 0.0, x_min, 0.0, -y_res, y_max)

        logger.debug(
            f"Synthesised constant roughness raster: z0={z0_value:.4f} m, "
            f"extent x=[{x_min:.1f}, {x_max:.1f}] y=[{y_min:.1f}, {y_max:.1f}]"
        )
        return roughness_data, roughness_transform