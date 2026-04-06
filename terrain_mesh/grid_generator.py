import numpy as np
import logging
import pyvista as pv
from scipy.ndimage import map_coordinates
from typing import Tuple

from .config import GridConfig, TerrainConfig
from .utils import rotate_coordinates, create_blockMesh_spacing

logger = logging.getLogger(__name__)

class StructuredGridGenerator:
    """Generate structured grids from terrain data"""
    
    def create_structured_grid(self, elevation_data, transform, target_rows, target_cols, 
                                                 rotation_deg, crop_mask,centre_utm, center_coordinates=True, 
                                                 x_grading=None, y_grading=None):
        """
        Create a rotated structured grid that aligns with terrain orientation.
        
        Parameters:
        -----------
        elevation_data : np.ndarray
            2D array containing elevation values
        transform : Affine
            Geospatial transform for pixel to UTM conversion
        target_rows : int
            Number of grid vertex rows in the output grid (``ny`` from GridConfig).
            The mesh will contain ``target_rows - 1`` cells in the y-direction.
        target_cols : int
            Number of grid vertex columns in the output grid (``nx`` from GridConfig).
            The mesh will contain ``target_cols - 1`` cells in the x-direction.
        rotation_deg : float
            Meteorological wind direction (0°=N, 90°=E, 180°=S, 270°=W)
        crop_mask : np.ndarray
            Boolean mask defining terrain area
        center_coordinates : bool
            Whether to center coordinates around origin
        x_grading : list of tuples, optional
            BlockMesh-style grading for X direction: [(length_frac, cell_frac, expansion_ratio), ...]
            If None, creates uniform spacing
        y_grading : list of tuples, optional
            BlockMesh-style grading for Y direction: [(length_frac, cell_frac, expansion_ratio), ...]
            If None, creates uniform spacing
        
        Create a rotated structured grid that fits exactly to terrain bounds.
        After sampling, any NaN values at the edges (caused by grid vertices landing
        just outside the DEM extent due to the rotated crop) are trimmed symmetrically
        (Fix 2) and a ValueError is raised if the trimming reduced the cell count below
        the configured target (Fix 1).
        """
        
        # 1. Find valid terrain bounds in pixel coordinates
        terrain_rows, terrain_cols = np.where(crop_mask)
        min_row, max_row = terrain_rows.min(), terrain_rows.max()
        min_col, max_col = terrain_cols.min(), terrain_cols.max()
        
        # 2. Convert ALL valid terrain points to UTM to find rotated bounds
        logger.debug("Finding terrain bounds in rotated coordinate system...")
        
        # Get all terrain pixels in UTM
        terrain_utm_x = terrain_cols * transform.a + transform.c
        terrain_utm_y = terrain_rows * transform.e + transform.f
        
        # Find center of terrain
        terrain_center_x = terrain_utm_x.mean()
        terrain_center_y = terrain_utm_y.mean()
        
        logger.debug(f"Terrain center: ({terrain_center_x:.1f}, {terrain_center_y:.1f})")
        
        # 3. Rotate all terrain points to find bounds in rotated coordinate system.
        # Coordinates are UTM (y increases northward), so geographic=True is required
        # so that y_rotated increases in the downwind direction (inlet at min y_rot).
        x_rotated, y_rotated = rotate_coordinates(
                                                    terrain_utm_x, terrain_utm_y, 
                                                    terrain_center_x, terrain_center_y, 
                                                    rotation_deg, inverse=True, geographic=True
                                                )
        
        # Find bounds in rotated space
        min_x_rot, max_x_rot = x_rotated.min(), x_rotated.max()
        min_y_rot, max_y_rot = y_rotated.min(), y_rotated.max()
        
        terrain_width = max_x_rot - min_x_rot
        terrain_height = max_y_rot - min_y_rot
        
        logger.debug(f"Rotated terrain bounds: {terrain_width:.1f}m x {terrain_height:.1f}m")
        logger.debug(f"Rotation: {rotation_deg}° from north")
        
        # 4. Create grid coordinates to fit these exact bounds
        if x_grading is not None:
            logger.debug(f"Creating X grading: {x_grading}")
            x_norm = create_blockMesh_spacing(target_cols, x_grading)
            # Scale to fit exact terrain width
            x_coords = x_norm * terrain_width + min_x_rot
        else:
            logger.debug("Creating uniform X spacing")
            x_coords = np.linspace(min_x_rot, max_x_rot, target_cols)
        
        if y_grading is not None:
            logger.debug(f"Creating Y grading: {y_grading}")
            y_norm = create_blockMesh_spacing(target_rows, y_grading)
            # Scale to fit exact terrain height
            y_coords = y_norm * terrain_height + min_y_rot
        else:
            logger.debug("Creating uniform Y spacing")
            y_coords = np.linspace(min_y_rot, max_y_rot, target_rows)
        
        X_local, Y_local = np.meshgrid(x_coords, y_coords)
        
        # 5. Rotate grid back to UTM coordinates.
        # geographic=True matches the convention used for the inverse rotation above.
        X_rotated_back, Y_rotated_back = rotate_coordinates(
                                                            X_local, Y_local,
                                                            0, 0,  # Already centered in rotated space
                                                            rotation_deg, inverse=False, geographic=True
                                                        )
        # 6. Translate back to UTM coordinates
        X_utm = X_rotated_back + terrain_center_x
        Y_utm = Y_rotated_back + terrain_center_y
        
        # 7. Convert UTM coordinates to pixel coordinates for sampling
        col_coords = (X_utm - transform.c) / transform.a
        row_coords = (Y_utm - transform.f) / transform.e
        
        # 8. Sample elevation data at grid points

        Z = map_coordinates(
            elevation_data,
            [row_coords, col_coords],
            order=1,
            mode='constant',
            cval=np.nan,
            prefilter=False
        )

        # 9. Fix 2: Symmetrically trim NaN border vertices for deterministic output.
        #    Vertices at the edges of a rotated domain may land just outside the valid
        #    DEM area and receive NaN from the constant-padding sampler.  Trimming the
        #    same number of rows/cols from each side makes the output shape independent
        #    of which side has the larger rounding error.
        nan_count = int(np.sum(np.isnan(Z)))
        logger.debug(f"NaN values after sampling: {nan_count}/{Z.size} ({100*nan_count/Z.size:.1f}%)")

        if nan_count > 0:
            nan_row_flags = np.any(np.isnan(Z), axis=1)  # True if row contains any NaN
            nan_col_flags = np.any(np.isnan(Z), axis=0)  # True if col contains any NaN

            if not np.any(~nan_row_flags) or not np.any(~nan_col_flags):
                raise ValueError(
                    "No fully NaN-free rows or columns found in the sampled grid. "
                    "Increase crop_size_km to ensure the DEM covers the full domain."
                )

            # Count how many edge rows/cols from each side contain NaN
            rows_from_top = int(np.argmax(~nan_row_flags))
            rows_from_bottom = int(np.argmax(~nan_row_flags[::-1]))
            cols_from_left = int(np.argmax(~nan_col_flags))
            cols_from_right = int(np.argmax(~nan_col_flags[::-1]))

            # Symmetric trim: use the larger of the two sides so both are trimmed equally
            rows_trim = max(rows_from_top, rows_from_bottom)
            cols_trim = max(cols_from_left, cols_from_right)

            r0, r1 = rows_trim, target_rows - rows_trim
            c0, c1 = cols_trim, target_cols - cols_trim

            if r1 <= r0 or c1 <= c0:
                raise ValueError(
                    f"Symmetric NaN trim (rows ±{rows_trim}, cols ±{cols_trim}) would "
                    f"consume the entire grid ({target_rows}×{target_cols}). "
                    "Increase crop_size_km."
                )

            Z = Z[r0:r1, c0:c1]
            X_utm = X_utm[r0:r1, c0:c1]
            Y_utm = Y_utm[r0:r1, c0:c1]

            trimmed_rows, trimmed_cols = Z.shape

            if rows_trim > 0 or cols_trim > 0:
                logger.warning(
                    f"Symmetrically trimmed {rows_trim} row(s) and {cols_trim} col(s) "
                    f"from each edge to remove NaN border vertices. "
                    f"Grid: {target_rows}×{target_cols} → "
                    f"{trimmed_rows}×{trimmed_cols} "
                    f"({trimmed_rows - 1}×{trimmed_cols - 1} cells)."
                )

            # Check for remaining interior NaN — a genuine DEM coverage gap
            interior_nan = int(np.sum(np.isnan(Z)))
            if interior_nan > 0:
                raise ValueError(
                    f"{interior_nan} interior NaN value(s) remain after symmetric edge "
                    "trimming. This indicates gaps in DEM coverage inside the domain. "
                    "Check the DEM for NoData holes."
                )

            # Fix 1: Validate that the configured cell count is still met.
            #         Any trim means the DEM crop is too small for the rotated grid.
            if trimmed_cols - 1 < target_cols - 1 or trimmed_rows - 1 < target_rows - 1:
                raise ValueError(
                    f"Grid has fewer valid cells than configured after removing NaN "
                    f"border vertices: got {trimmed_rows - 1}×{trimmed_cols - 1} cells, "
                    f"expected {target_rows - 1}×{target_cols - 1}. "
                    "Increase crop_size_km to provide full DEM coverage for the "
                    "rotated grid."
                )

            target_rows, target_cols = trimmed_rows, trimmed_cols

        # 10. Center coordinates if requested
        if center_coordinates:
            X_final = X_utm - terrain_center_x
            Y_final = Y_utm - terrain_center_y
            logger.debug("Coordinates centered at origin")
        else:
            X_final = X_utm
            Y_final = Y_utm

        logger.debug(f"Final grid: {target_cols} x {target_rows} ({target_cols - 1} x {target_rows - 1} cells)")

        # 11. Create PyVista structured grid
        points = np.column_stack((X_final.ravel(), Y_final.ravel(), Z.ravel()))

        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (target_cols, target_rows, 1)
        grid.point_data['elevation'] = Z.ravel()

        return grid
    
    def create_grid(self, elevation_data: np.ndarray, transform, grid_config: GridConfig,
               terrain_config: TerrainConfig, crop_mask: np.ndarray, centre_utm) -> pv.StructuredGrid:
        """Create structured grid fitting terrain bounds exactly"""
        
        return self.create_structured_grid(
            elevation_data, 
            transform, 
            grid_config.ny,  # target_rows
            grid_config.nx,  # target_cols
            terrain_config.rotation_deg, 
            crop_mask,
            centre_utm,
            center_coordinates=terrain_config.center_coordinates,
            x_grading=grid_config.x_grading, 
            y_grading=grid_config.y_grading
        )