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

    def create_grid(self, elevation_data: np.ndarray, transform, grid_config: GridConfig,
                    terrain_config: TerrainConfig, crop_mask: np.ndarray,
                    centre_utm) -> pv.StructuredGrid:
        """Create a rotated structured grid fitting the terrain bounds exactly.

        Finds the actual terrain bounds in the rotated (flow-aligned) coordinate
        system, creates a structured grid with the requested cell counts and
        optional grading, then back-rotates to UTM coordinates and samples
        elevation values via bilinear interpolation.

        Args:
            elevation_data: 2D array of elevation values.
            transform: Affine geospatial transform for pixel-to-UTM conversion.
            grid_config: GridConfig specifying cell counts and optional grading.
            terrain_config: TerrainConfig specifying rotation and coordinate options.
            crop_mask: Boolean mask defining the valid terrain area.
            centre_utm: (x, y) UTM coordinates of the terrain centre.

        Returns:
            pv.StructuredGrid: Surface mesh with point coordinates and
            ``elevation`` point-data array.
        """
        target_rows = grid_config.ny
        target_cols = grid_config.nx
        rotation_deg = terrain_config.rotation_deg
        center_coordinates = terrain_config.center_coordinates
        x_grading = grid_config.x_grading
        y_grading = grid_config.y_grading

        # 1. Find valid terrain bounds in pixel coordinates
        terrain_rows, terrain_cols = np.where(crop_mask)

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

        # 9. Check for any NaN values (should be minimal since grid fits terrain)
        nan_count = np.sum(np.isnan(Z))
        total_points = Z.size
        logger.debug(f"NaN values: {nan_count}/{total_points} ({100*nan_count/total_points:.1f}%)")

        if nan_count > 0:
            logger.warning(f"{nan_count} grid points fall outside terrain bounds — consider adjusting domain")

        # 10. Center coordinates if requested
        if center_coordinates:
            X_final = X_utm - terrain_center_x
            Y_final = Y_utm - terrain_center_y
            logger.debug("Coordinates centered at origin")
        else:
            X_final = X_utm
            Y_final = Y_utm

        logger.debug(f"Final grid: {target_cols} x {target_rows}")

        # 11. Create PyVista structured grid
        points = np.column_stack((X_final.ravel(), Y_final.ravel(), Z.ravel()))

        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (target_cols, target_rows, 1)
        grid.point_data['elevation'] = Z.ravel()

        return grid