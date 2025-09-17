import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates
from typing import Tuple

from .config import GridConfig, TerrainConfig

class StructuredGridGenerator:
    """Generate structured grids from terrain data"""
    
    def create_structured_grid(self, elevation_data, transform, target_rows, target_cols, 
                                                 rotation_deg, crop_mask, center_coordinates=True, 
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
            Number of rows in output grid
        target_cols : int
            Number of columns in output grid
        rotation_deg : float
            Rotation angle in degrees (positive = clockwise from north)
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
        
        This approach:
        1. Finds actual terrain bounds from crop_mask
        2. Creates grid with specified divisions to fit those bounds exactly
        3. No cropping needed - all points are valid
        """
        
        # 1. Find valid terrain bounds in pixel coordinates
        terrain_rows, terrain_cols = np.where(crop_mask)
        min_row, max_row = terrain_rows.min(), terrain_rows.max()
        min_col, max_col = terrain_cols.min(), terrain_cols.max()
        
        # 2. Convert ALL valid terrain points to UTM to find rotated bounds
        print("Finding terrain bounds in rotated coordinate system...")
        
        # Get all terrain pixels in UTM
        terrain_utm_x = terrain_cols * transform.a + transform.c
        terrain_utm_y = terrain_rows * transform.e + transform.f
        
        # Find center of terrain
        terrain_center_x = terrain_utm_x.mean()
        terrain_center_y = terrain_utm_y.mean()
        
        print(f"Terrain center: ({terrain_center_x:.1f}, {terrain_center_y:.1f})")
        
        # 3. Rotate all terrain points to find bounds in rotated coordinate system
        theta = np.radians(rotation_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Translate to center, then rotate
        x_centered = terrain_utm_x - terrain_center_x
        y_centered = terrain_utm_y - terrain_center_y
        
        x_rotated = cos_theta * x_centered - sin_theta * y_centered
        y_rotated = sin_theta * x_centered + cos_theta * y_centered
        
        # Find bounds in rotated space
        min_x_rot, max_x_rot = x_rotated.min(), x_rotated.max()
        min_y_rot, max_y_rot = y_rotated.min(), y_rotated.max()
        
        terrain_width = max_x_rot - min_x_rot
        terrain_height = max_y_rot - min_y_rot
        
        print(f"Rotated terrain bounds: {terrain_width:.1f}m x {terrain_height:.1f}m")
        print(f"Rotation: {rotation_deg}° clockwise from north")
        
        # 4. Create grid coordinates to fit these exact bounds
        if x_grading is not None:
            print(f"Creating X grading: {x_grading}")
            x_norm = self.create_blockMesh_spacing(target_cols, x_grading)
            # Scale to fit exact terrain width
            x_coords = x_norm * terrain_width + min_x_rot
        else:
            print("Creating uniform X spacing")
            x_coords = np.linspace(min_x_rot, max_x_rot, target_cols)
        
        if y_grading is not None:
            print(f"Creating Y grading: {y_grading}")
            y_norm = self.create_blockMesh_spacing(target_rows, y_grading)
            # Scale to fit exact terrain height
            y_coords = y_norm * terrain_height + min_y_rot
        else:
            print("Creating uniform Y spacing")
            y_coords = np.linspace(min_y_rot, max_y_rot, target_rows)
        
        X_local, Y_local = np.meshgrid(x_coords, y_coords)
        
        # 5. Rotate grid back to UTM coordinates
        X_rotated_back = cos_theta * X_local + sin_theta * Y_local
        Y_rotated_back = -sin_theta * X_local + cos_theta * Y_local
        
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
        print(f"NaN values: {nan_count}/{total_points} ({100*nan_count/total_points:.1f}%)")
        
        if nan_count > 0:
            print(f"Warning: {nan_count} points outside terrain - consider adjusting bounds")
        
        # 10. Center coordinates if requested
        if center_coordinates:
            X_final = X_utm - terrain_center_x
            Y_final = Y_utm - terrain_center_y
            print(f"Coordinates centered at origin")
        else:
            X_final = X_utm
            Y_final = Y_utm
        
        print(f"Final grid: {target_cols} x {target_rows} (as requested)")
        
        # 11. Create PyVista structured grid
        points = np.column_stack((X_final.ravel(), Y_final.ravel(), Z.ravel()))
        
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (target_cols, target_rows, 1)
        grid.point_data['elevation'] = Z.ravel()
        
        # Save to file
        grid.save("terrain_structured.vtk")
        print("Grid saved to terrain_structured.vtk")
        return grid
    
    def create_blockMesh_spacing(self, n_points, grading_spec):
        """
        Create variable spacing coordinates from 0 to 1 using blockMesh-style grading.
        
        Parameters:
        -----------
        n_points : int
            Total number of points
        grading_spec : list of tuples
            [(length_fraction, cell_fraction, expansion_ratio), ...]
            - length_fraction: fraction of domain length for this region
            - cell_fraction: fraction of total cells for this region  
            - expansion_ratio: last_cell_size/first_cell_size in this region
        
        Returns:
        --------
        np.ndarray
            Coordinate array from 0 to 1 with blockMesh-style spacing
        """
        
        total_cells = n_points - 1
        n_regions = len(grading_spec)
        
        # Extract specifications
        length_fractions = np.array([spec[0] for spec in grading_spec])
        cell_fractions = np.array([spec[1] for spec in grading_spec])
        expansion_ratios = np.array([spec[2] for spec in grading_spec])
        
        # Validate inputs
        if abs(length_fractions.sum() - 1.0) > 1e-6:
            raise ValueError(f"Length fractions sum to {length_fractions.sum():.6f}, must sum to 1.0")
        
        if abs(cell_fractions.sum() - 1.0) > 1e-6:
            raise ValueError(f"Cell fractions sum to {cell_fractions.sum():.6f}, must sum to 1.0")
        
        # Calculate target cell counts (may not be integers)
        target_cells = cell_fractions * total_cells
        
        # Round to integers and adjust to maintain total
        actual_cells = np.round(target_cells).astype(int)
        
        # Adjust for rounding errors
        cell_diff = total_cells - actual_cells.sum()
        if cell_diff != 0:
            # Add/subtract cells from regions with largest rounding errors
            errors = target_cells - actual_cells
            if cell_diff > 0:
                # Need to add cells - add to regions with most positive error
                indices = np.argsort(errors)[::-1]
            else:
                # Need to remove cells - remove from regions with most negative error  
                indices = np.argsort(errors)
            
            for i in range(abs(cell_diff)):
                actual_cells[indices[i]] += np.sign(cell_diff)
        
        # Check for warnings
        relative_errors = np.abs(actual_cells - target_cells) / target_cells
        max_error = relative_errors.max()
        if max_error > 0.05:  # 5% threshold
            print(f"WARNING: Cell count adjustment needed. Max relative error: {max_error:.1%}")
            for i, (target, actual) in enumerate(zip(target_cells, actual_cells)):
                error = abs(actual - target) / target
                if error > 0.02:
                    print(f"  Region {i}: {target:.1f} -> {actual} cells ({error:.1%} error)")
        
        # Generate coordinates for each region
        coords = [0.0]  # Start at 0
        current_pos = 0.0
        
        print(f"Region breakdown:")
        for i, (length_frac, actual_cell_count, expansion_ratio) in enumerate(zip(length_fractions, actual_cells, expansion_ratios)):
            region_length = length_frac
            
            print(f"  Region {i}: {actual_cell_count} cells, length {region_length:.3f}, ratio {expansion_ratio}")
            
            if actual_cell_count == 0:
                continue
                
            # Generate spacing within this region
            region_coords = self.generate_region_coordinates(actual_cell_count, expansion_ratio)
            
            # Scale to region length and add to current position
            region_coords_scaled = region_coords * region_length + current_pos
            
            # Add coordinates (skip the first one as it's already included)
            coords.extend(region_coords_scaled[1:])
            
            current_pos += region_length
        
        return np.array(coords)
    
    def generate_region_coordinates(self, n_cells, expansion_ratio):
        """
        Generate coordinates within a single region [0,1] with given expansion ratio.
        
        Parameters:
        -----------
        n_cells : int
            Number of cells in this region
        expansion_ratio : float
            Ratio of last_cell_size/first_cell_size
            
        Returns:
        --------
        np.ndarray
            Coordinates from 0 to 1 for this region
        """
        
        if n_cells == 0:
            return np.array([0.0, 1.0])
        
        if n_cells == 1:
            return np.array([0.0, 1.0])
        
        # For uniform spacing (expansion_ratio ≈ 1)
        if abs(expansion_ratio - 1.0) < 1e-6:
            return np.linspace(0.0, 1.0, n_cells + 1)
        
        # For geometric progression
        # If first cell has size ds, then cell sizes are: ds, ds*r, ds*r², ..., ds*r^(n-1)
        # where r is the common ratio between adjacent cells
        # Total length = ds * (1 + r + r² + ... + r^(n-1)) = ds * (r^n - 1)/(r - 1) = 1
        # Also: last_cell/first_cell = ds*r^(n-1) / ds = r^(n-1) = expansion_ratio
        
        r = expansion_ratio**(1.0/(n_cells-1))  # Common ratio between adjacent cells
        
        # Calculate first cell size
        if abs(r - 1.0) < 1e-6:
            ds = 1.0 / n_cells
        else:
            ds = (r - 1.0) / (r**n_cells - 1.0)
        
        # Generate cell sizes
        cell_sizes = ds * r**np.arange(n_cells)
        
        # Generate coordinates
        coords = np.zeros(n_cells + 1)
        coords[1:] = np.cumsum(cell_sizes)
        
        return coords
    
    def create_grid(self, elevation_data: np.ndarray, transform, grid_config: GridConfig,
               terrain_config: TerrainConfig, crop_mask: np.ndarray) -> pv.StructuredGrid:
        """Create structured grid fitting terrain bounds exactly"""
        
        return self.create_structured_grid(
            elevation_data, 
            transform, 
            grid_config.ny,  # target_rows
            grid_config.nx,  # target_cols
            terrain_config.rotation_deg, 
            crop_mask,
            center_coordinates=terrain_config.center_coordinates,
            x_grading=grid_config.x_grading, 
            y_grading=grid_config.y_grading
        )