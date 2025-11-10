"""O-grid generator for circular domain structured mesh"""

import numpy as np
import pyvista as pv
from typing import Tuple, Dict
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from .ogrid_config import OGridConfig


class OGridGenerator:
    """Generate O-grid structured mesh for circular domains"""
    
    def __init__(self):
        self.vertices_ground = None
        self.vertex_map = None
        self.ogrid_topology = None
        self.vertices_elevations = None
    
    def create_ogrid(self,
                 elevation_data: np.ndarray,
                 transform: object,
                 ogrid_config: OGridConfig,
                 terrain_config: object,
                 boundary_config: object,
                 crop_mask: np.ndarray,
                 centre_utm: Tuple[float, float]) -> pv.StructuredGrid:
        """
        Create O-grid structured mesh from circular terrain data.
        
        Creates a proper structured grid where EVERY point has terrain elevation sampled.
        This is used by blockMeshDict generator.
        """
        
        print("\n" + "="*60)
        print("Generating O-Grid Structured Mesh")
        print("="*60)
        
        ogrid_config.print_summary()
        
        # Calculate domain parameters
        crop_radius_m = (terrain_config.crop_size_km * 1000) / 2
        aoi_half_size = crop_radius_m * boundary_config.aoi_fraction
        
        print(f"\nDomain parameters:")
        print(f"  Crop radius: {crop_radius_m:.1f} m")
        print(f"  AOI square half-size: {aoi_half_size:.1f} m")
        print(f"  Center UTM: {centre_utm}")
        
        cx, cy = centre_utm
        n_sectors = ogrid_config.n_sectors
        nr = ogrid_config.nr  # Total radial points
        
        # Create elevation interpolator
        from scipy.interpolate import RegularGridInterpolator
        
        nrows, ncols = elevation_data.shape
        x_min = transform.c
        y_max = transform.f
        x_res = transform.a
        y_res = -transform.e
        
        x_coords = np.arange(ncols) * x_res + x_min
        y_coords = np.arange(nrows) * (-y_res) + y_max
        
        # Fill NaN for interpolation
        elevation_filled = elevation_data.copy()
        invalid_mask = ~crop_mask | np.isnan(elevation_data)
        if np.any(invalid_mask):
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
            elevation_filled[invalid_mask] = elevation_data[tuple(indices[:, invalid_mask])]
        
        interpolator = RegularGridInterpolator(
            (y_coords, x_coords),
            elevation_filled,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Generate O-grid points
        print(f"\nGenerating O-grid points: {n_sectors} sectors × {nr} radial...")
        
        # Radial positions (0 = center, nr-1 = outer circle)
        radii = np.linspace(0, crop_radius_m, nr)
        
        # Angular positions for each sector
        thetas = np.linspace(0, 2*np.pi, n_sectors + 1)[:-1]  # n_sectors points (exclude duplicate at 2π)
        
        # Create meshgrid: (n_sectors, nr)
        theta_grid, r_grid = np.meshgrid(thetas, radii, indexing='ij')
        
        # Convert to Cartesian
        x_grid = cx + r_grid * np.cos(theta_grid)
        y_grid = cy + r_grid * np.sin(theta_grid)
        
        # Sample terrain elevation at all points
        print("Sampling terrain elevations at all O-grid points...")
        z_grid = np.zeros_like(x_grid)
        
        for i in range(n_sectors):
            for j in range(nr):
                x = x_grid[i, j]
                y = y_grid[i, j]
                z = float(interpolator([y, x])[0])
                
                # Check if within circular domain
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist > crop_radius_m or np.isnan(z):
                    z = np.nan  # Mark as invalid
                
                z_grid[i, j] = z
        
        # Store for blockMeshDict generation
        self.vertices_ground = np.stack([x_grid, y_grid], axis=-1)  # (n_sectors, nr, 2)
        self.vertices_elevations = z_grid  # (n_sectors, nr)
        
        # Create 3D points array for VTK: (n_sectors, nr, 3)
        points_3d = np.stack([x_grid, y_grid, z_grid], axis=-1)
        
        # Flatten to (n_sectors * nr, 3) for PyVista
        points_flat = points_3d.reshape(-1, 3)
        
        print(f"\nO-grid mesh created:")
        print(f"  Dimensions: {n_sectors} × {nr}")
        print(f"  Total points: {n_sectors * nr:,}")
        print(f"  Valid elevations: {np.sum(~np.isnan(z_grid)):,}")
        print(f"  Elevation range: [{np.nanmin(z_grid):.1f}, {np.nanmax(z_grid):.1f}] m")
        
        # Create PyVista StructuredGrid
        # Dimensions are (n_circumferential, n_radial, 1) for 2D grid in 3D space
        grid = pv.StructuredGrid()
        grid.points = points_flat
        grid.dimensions = (nr, n_sectors, 1)  # (x, y, z) = (radial, circumferential, vertical)
        
        # Add elevation as scalar data
        grid.point_data['elevation'] = z_grid.flatten()
        
        print("\nO-grid generation complete!")
        print("="*60)
        
        return grid
    
    def _generate_ogrid_vertices(self,
                                centre_utm: Tuple[float, float],
                                aoi_half_size: float,
                                crop_radius_m: float,
                                ogrid_config: OGridConfig) -> np.ndarray:
        """
        Generate O-grid vertex positions (x, y) only.
        
        Topology:
        - Outer circle: n_sectors vertices
        - Inner ring: n_sectors vertices (transitioning to square)
        - Center square: 4 corner vertices + optional center point
        
        Returns:
            Array of shape (N, 2) with (x, y) positions
        """
        
        cx, cy = centre_utm
        n_sectors = ogrid_config.n_sectors
        
        vertices = []
        
        # Layer 1: Outer circle vertices (n_sectors vertices)
        print(f"\nGenerating outer circle vertices ({n_sectors} vertices)...")
        for i in range(n_sectors):
            angle = 2 * np.pi * i / n_sectors  # Start at 0 (East), counter-clockwise
            x = cx + crop_radius_m * np.cos(angle)
            y = cy + crop_radius_m * np.sin(angle)
            vertices.append([x, y])
        
        # Layer 2: Inner transition ring (n_sectors vertices)
        # These connect smoothly to the square corners/edges
        print(f"Generating inner ring vertices ({n_sectors} vertices)...")
        inner_radius = aoi_half_size * np.sqrt(2)  # Distance to square corners
        
        for i in range(n_sectors):
            angle = 2 * np.pi * i / n_sectors
            
            # Calculate position on inner ring (blend between circle and square)
            # Use projection toward square
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            # Find intersection with square boundary
            # Square: |x| <= aoi_half_size, |y| <= aoi_half_size
            if abs(cos_a) > abs(sin_a):
                # Hit vertical edges first
                x_square = aoi_half_size * np.sign(cos_a)
                y_square = x_square * np.tan(angle)
            else:
                # Hit horizontal edges first
                y_square = aoi_half_size * np.sign(sin_a)
                x_square = y_square / np.tan(angle) if sin_a != 0 else 0
            
            # Blend circle and square (70% toward square for smooth transition)
            blend_factor = 0.7
            x_circle = inner_radius * cos_a
            y_circle = inner_radius * sin_a
            
            x = cx + (blend_factor * x_square + (1 - blend_factor) * x_circle)
            y = cy + (blend_factor * y_square + (1 - blend_factor) * y_circle)
            vertices.append([x, y])
        
        # Layer 3: Center square vertices (4 corners)
        print("Generating center square vertices (4 corners)...")
        square_corners = [
            [cx + aoi_half_size, cy + aoi_half_size],   # NE (45°)
            [cx - aoi_half_size, cy + aoi_half_size],   # NW (135°)
            [cx - aoi_half_size, cy - aoi_half_size],   # SW (225°)
            [cx + aoi_half_size, cy - aoi_half_size],   # SE (315°)
        ]
        vertices.extend(square_corners)
        
        # Optional: Center point if subdividing
        if ogrid_config.subdivide_center:
            print("Adding center point...")
            vertices.append([cx, cy])
        
        vertices_array = np.array(vertices)
        print(f"Total vertices generated: {len(vertices_array)}")
        
        return vertices_array
    
    def _sample_terrain_at_vertices(self,
                                   vertices_xy: np.ndarray,
                                   elevation_data: np.ndarray,
                                   transform: object,
                                   crop_mask: np.ndarray) -> np.ndarray:
        """
        Sample terrain elevation at O-grid vertex positions using interpolation.
        
        Args:
            vertices_xy: (N, 2) array of (x, y) positions
            elevation_data: 2D elevation array
            transform: Affine transform
            crop_mask: Boolean mask
        
        Returns:
            Array of shape (N,) with elevation values (NaN for invalid)
        """
        
        print("\nSampling terrain elevations at vertices...")
        
        nrows, ncols = elevation_data.shape
        
        # Create coordinate arrays from transform
        x_min = transform.c
        y_max = transform.f
        x_res = transform.a
        y_res = -transform.e
        
        x_coords = np.arange(ncols) * x_res + x_min
        y_coords = np.arange(nrows) * (-y_res) + y_max  # Descending
        
        # Fill NaN values for interpolation
        elevation_filled = elevation_data.copy()
        invalid_mask = ~crop_mask | np.isnan(elevation_data)
        
        if np.any(invalid_mask):
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
            elevation_filled[invalid_mask] = elevation_data[tuple(indices[:, invalid_mask])]
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (y_coords, x_coords),
            elevation_filled,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Sample at vertex positions
        elevations = interpolator(vertices_xy[:, [1, 0]])  # (y, x) order
        
        valid_count = np.sum(~np.isnan(elevations))
        print(f"Sampled {valid_count}/{len(elevations)} valid elevations")
        
        return elevations
    
    def _create_ogrid_topology(self, ogrid_config: OGridConfig) -> Dict:
        """
        Create O-grid block connectivity topology.
        
        Returns:
            Dict with block definitions
        """
        
        n_sectors = ogrid_config.n_sectors
        
        # Vertex indices:
        # 0 to n_sectors-1: outer circle
        # n_sectors to 2*n_sectors-1: inner ring
        # 2*n_sectors to 2*n_sectors+3: square corners (4 vertices)
        # Optional: 2*n_sectors+4: center point
        
        outer_start = 0
        inner_start = n_sectors
        square_start = 2 * n_sectors
        center_idx = 2 * n_sectors + 4 if ogrid_config.subdivide_center else None
        
        blocks = []
        
        # Outer radial blocks (16 blocks)
        print(f"\nDefining outer radial blocks ({n_sectors} blocks)...")
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            
            # Vertices (counter-clockwise from outer):
            v0 = outer_start + i
            v1 = outer_start + i_next
            v2 = inner_start + i_next
            v3 = inner_start + i
            
            blocks.append({
                'type': 'outer_radial',
                'sector': i,
                'vertices': [v0, v1, v2, v3]
            })
        
        # Inner blocks (connecting inner ring to square)
        print(f"Defining inner transition blocks ({n_sectors} blocks)...")
        
        # Calculate which sectors align with square corners (Option A)
        # Corners at indices: 0, n_sectors/4, n_sectors/2, 3*n_sectors/4
        corner_sectors = [0, n_sectors//4, n_sectors//2, 3*n_sectors//4]
        
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            
            # Determine which square corner this sector connects to
            corner_idx = None
            for j, corner_sector in enumerate(corner_sectors):
                if i == corner_sector:
                    corner_idx = j
                    break
            
            if corner_idx is not None:
                # This sector connects directly to a corner
                # Quad: inner_ring[i] -> inner_ring[i+1] -> square_corner -> square_corner
                v0 = inner_start + i
                v1 = inner_start + i_next
                v2 = square_start + (corner_idx + 1) % 4
                v3 = square_start + corner_idx
            else:
                # This sector connects between corners (to square edges)
                # Determine which two square corners to use
                prev_corner_idx = max([idx for idx, cs in enumerate(corner_sectors) if cs < i], default=3)
                next_corner_idx = (prev_corner_idx + 1) % 4
                
                v0 = inner_start + i
                v1 = inner_start + i_next
                v2 = square_start + next_corner_idx
                v3 = square_start + prev_corner_idx
            
            blocks.append({
                'type': 'inner_transition',
                'sector': i,
                'vertices': [v0, v1, v2, v3]
            })
        
        # Center square blocks (optional subdivision)
        if ogrid_config.subdivide_center:
            print("Defining center square blocks (4 blocks)...")
            for i in range(4):
                i_next = (i + 1) % 4
                
                v0 = square_start + i
                v1 = square_start + i_next
                v2 = center_idx
                v3 = center_idx
                
                blocks.append({
                    'type': 'center_square',
                    'corner': i,
                    'vertices': [v0, v1, v2, v3]
                })
        
        print(f"Total blocks defined: {len(blocks)}")
        
        return {'blocks': blocks, 'n_vertices': len(self.vertices_ground) if self.vertices_ground is not None else 0}
    
    def _create_structured_grid_from_ogrid(self,
                                      vertices_3d: np.ndarray,
                                      ogrid_config: OGridConfig) -> pv.UnstructuredGrid:
        """
        Create O-grid structured mesh from circular terrain data.
        ...
        Returns:
            PyVista UnstructuredGrid
        """
        
        n_ground = len(vertices_3d)
        n_sectors = ogrid_config.n_sectors
        
        # Vertex indices
        outer_start = 0
        inner_start = n_sectors
        square_start = 2 * n_sectors
        center_idx = 2 * n_sectors + 4 if ogrid_config.subdivide_center else None
        
        cells = []
        cell_types = []
        
        print(f"\nCreating UnstructuredGrid with cell connectivity...")
        
        # Define outer radial blocks (16 blocks)
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            
            # Ground face vertices (counter-clockwise)
            v0 = outer_start + i
            v1 = outer_start + i_next
            v2 = inner_start + i_next
            v3 = inner_start + i
            
            # Check if all vertices are valid (not NaN)
            z_vals = vertices_3d[[v0, v1, v2, v3], 2]
            if np.any(np.isnan(z_vals)):
                continue  # Skip blocks with NaN vertices
            
            # For VTK, we'll create a single quad cell for ground surface
            # Format: [n_points, p0, p1, p2, p3]
            cells.append([4, v0, v1, v2, v3])
            cell_types.append(pv.CellType.QUAD)
        
        # Define inner transition blocks (16 blocks)
        corner_sectors = [0, n_sectors//4, n_sectors//2, 3*n_sectors//4]
        
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            
            # Determine which square corners to connect to
            corner_idx = min(range(4), key=lambda j: abs((i - corner_sectors[j]) % n_sectors))
            next_corner_idx = (corner_idx + 1) % 4
            
            v0 = inner_start + i
            v1 = inner_start + i_next
            v2 = square_start + next_corner_idx
            v3 = square_start + corner_idx
            
            # Check validity
            z_vals = vertices_3d[[v0, v1, v2, v3], 2]
            if np.any(np.isnan(z_vals)):
                continue
            
            cells.append([4, v0, v1, v2, v3])
            cell_types.append(pv.CellType.QUAD)
        
        # Center square blocks (if subdivided)
        if ogrid_config.subdivide_center and center_idx is not None:
            for i in range(4):
                i_next = (i + 1) % 4
                
                v0 = square_start + i
                v1 = square_start + i_next
                v2 = center_idx
                v3 = center_idx  # Triangle collapsed to quad
                
                z_vals = vertices_3d[[v0, v1, v2], 2]
                if np.any(np.isnan(z_vals)):
                    continue
                
                # For triangle, use 3 vertices
                cells.append([3, v0, v1, v2])
                cell_types.append(pv.CellType.TRIANGLE)
        else:
            # Single center quad (4 corners)
            v0 = square_start + 0
            v1 = square_start + 1
            v2 = square_start + 2
            v3 = square_start + 3
            
            z_vals = vertices_3d[[v0, v1, v2, v3], 2]
            if not np.any(np.isnan(z_vals)):
                cells.append([4, v0, v1, v2, v3])
                cell_types.append(pv.CellType.QUAD)
        
        # Convert to VTK format
        cells_array = np.hstack(cells)
        
        # Create UnstructuredGrid
        grid = pv.UnstructuredGrid(cells_array, cell_types, vertices_3d)
        
        # Add elevation as scalar data for visualization
        elevations = vertices_3d[:, 2].copy()
        grid.point_data['elevation'] = elevations
        
        print(f"Created UnstructuredGrid:")
        print(f"  Points: {grid.n_points}")
        print(f"  Cells: {grid.n_cells}")
        print(f"  Valid elevations: {np.sum(~np.isnan(elevations))}")
        
        return grid