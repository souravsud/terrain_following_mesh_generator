import numpy as np
import os
from datetime import datetime
import json
from pathlib import Path
from typing import List, Optional, Tuple
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator


_GRADING_TOLERANCE = 1e-6


def validate_grading_fractions(grading: List[Tuple[float, float, float]], name: str) -> None:
    """Validate that grading length and cell fractions each sum to 1.0.

    Args:
        grading: List of (length_fraction, cell_fraction, expansion_ratio) tuples.
        name: Name of the grading parameter, used in error messages.

    Raises:
        ValueError: If fractions don't sum to 1.0 within tolerance.
    """
    length_sum = sum(spec[0] for spec in grading)
    cell_sum = sum(spec[1] for spec in grading)
    if abs(length_sum - 1.0) > _GRADING_TOLERANCE:
        raise ValueError(f"{name} length fractions must sum to 1.0, got {length_sum}")
    if abs(cell_sum - 1.0) > _GRADING_TOLERANCE:
        raise ValueError(f"{name} cell fractions must sum to 1.0, got {cell_sum}")


def get_valid_cell_indices(valid_mask: np.ndarray):
    """Yield (j, i) index pairs for all 2×2 grid cells whose four corners are valid.

    Args:
        valid_mask: Boolean array of shape (ny, nx) where True marks valid points.

    Returns:
        An iterable of (j, i) integer tuples such that valid_mask[j, i],
        valid_mask[j, i+1], valid_mask[j+1, i+1], and valid_mask[j+1, i] are
        all True.
    """
    cell_valid = (
        valid_mask[:-1, :-1] & valid_mask[:-1, 1:] &
        valid_mask[1:, 1:] & valid_mask[1:, :-1]
    )
    return zip(*np.where(cell_valid))


def build_roughness_interpolator(
    roughness_data: np.ndarray,
    roughness_transform: object,
    default_z0: float = 0.1,
) -> RegularGridInterpolator:
    """Build a bilinear interpolator for a roughness raster.

    Fills NaN values in *roughness_data* using nearest-neighbour propagation
    before constructing the interpolator, ensuring continuous coverage over
    the rotated terrain crop region.

    Args:
        roughness_data: 2-D numpy array of z0 values (may contain NaN outside
            the rotated crop region).
        roughness_transform: Affine transform for the roughness raster.  The
            relevant attributes are ``c`` (x_min), ``f`` (y_max), ``a``
            (x_pixel_size > 0), and ``e`` (y_pixel_size < 0 for north-up).
        default_z0: Fill value used for query points that fall outside the
            raster extent (should not occur in practice).

    Returns:
        :class:`scipy.interpolate.RegularGridInterpolator` that accepts query
        points as ``(y, x)`` coordinate pairs.
    """
    from scipy.ndimage import distance_transform_edt

    nrows, ncols = roughness_data.shape
    x_res = roughness_transform.a
    y_res = -roughness_transform.e  # positive pixel height (transform.e < 0)
    x_coords = np.arange(ncols) * x_res + roughness_transform.c
    y_coords = np.arange(nrows) * (-y_res) + roughness_transform.f  # descending

    filled = roughness_data.copy()
    invalid = np.isnan(filled)
    if np.any(invalid):
        indices = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        filled[invalid] = roughness_data[tuple(indices[:, invalid])]

    return RegularGridInterpolator(
        (y_coords, x_coords), filled,
        method='linear', bounds_error=False, fill_value=default_z0,
    )


def load_terrain_points(terrain_map: str) -> Tuple[int, int, np.ndarray]:
    """Load terrain surface points from an NPZ terrain map file.

    This is the inverse of the ``np.savez_compressed`` call in
    :meth:`TerrainMeshPipeline._save_maps`.  It returns the same ``(ny, nx,
    3)`` points array that the VTK reader previously produced, so existing
    downstream code that indexes ``points[j, i, :]`` works unchanged.

    Args:
        terrain_map: Path to ``maps/terrain_map.npz``.

    Returns:
        Tuple ``(ny, nx, points)`` where *points* has shape ``(ny, nx, 3)``
        with columns ``[x, y, elevation]``.
    """
    data = np.load(terrain_map)
    ny, nx = data['elevation'].shape
    points = np.stack([data['x'], data['y'], data['elevation']], axis=-1)
    return ny, nx, points


def rotate_coordinates(x, y, center_x, center_y, rotation_deg, inverse=False, geographic=False):
    """
    Rotate coordinates around center point.

    Parameters:
    -----------
    rotation_deg : float
        Meteorological wind direction (0°=N, 90°=E, 180°=S, 270°=W)
    inverse : bool
        If True, apply inverse rotation (terrain → flow-aligned).
        If False, apply forward rotation (flow-aligned → terrain).
    geographic : bool
        If True, use geographic/UTM convention where y increases northward.
        The inverse rotation maps UTM coordinates to flow-aligned coordinates
        such that positive y_rot points downwind.
        If False (default), use pixel/image convention where y increases
        southward (row 0 = top = North).
    """
    if geographic:
        # UTM convention (y = North): the inverse rotation (terrain → flow-aligned)
        # should map the downwind direction to positive y_rot.
        # Setting the base theta to -(rotation_deg + 180) achieves this: when
        # inverse=True the applied angle becomes rotation_deg + 180, placing the
        # downwind direction along the positive y_rot axis.
        theta = np.radians(-(rotation_deg + 180))
    else:
        # Pixel convention (y = South, row 0 = top = North)
        theta = np.radians(rotation_deg - 270)
    if inverse:
        theta = -theta

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_centered = x - center_x
    y_centered = y - center_y

    x_rot = cos_theta * x_centered - sin_theta * y_centered
    y_rot = sin_theta * x_centered + cos_theta * y_centered

    return x_rot, y_rot

def smooth_terrain_for_cfd(elevation_data, sigma=2.0, preserve_nan=True):
    """Smooth terrain data for better CFD mesh quality.

    Args:
        elevation_data: 2D numpy array of elevation values.
        sigma: Smoothing strength (higher = more smoothing).
        preserve_nan: If True, keep NaN areas (outside rotated crop) as NaN.

    Returns:
        Smoothed elevation array of the same shape.
    """
    if preserve_nan:
        valid_mask = ~np.isnan(elevation_data)
        smoothed = elevation_data.copy()
        
        # Only smooth valid areas
        valid_data = elevation_data[valid_mask]
        if len(valid_data) > 0:
            # Create temporary array for smoothing
            temp_array = np.zeros_like(elevation_data)
            temp_array[valid_mask] = valid_data
            
            # Fill NaN regions with nearest-neighbor values to avoid artificial boundaries
            indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
            temp_array[~valid_mask] = temp_array[tuple(indices[:, ~valid_mask])]
            
            # Apply smoothing
            smoothed_temp = gaussian_filter(temp_array, sigma=sigma)
            
            # Restore only valid areas
            smoothed[valid_mask] = smoothed_temp[valid_mask]
    else:
        smoothed = gaussian_filter(elevation_data, sigma=sigma)
    
    return smoothed

def write_metadata(
    dem_path,
    rmap_path,
    terrain_config,
    grid_config,
    mesh_config,
    boundary_config,
    visualization_config,
    elevation_data: np.ndarray,
    treated_elevation: np.ndarray,
    transform,
    crs,
    min_elevation: float,
    centre_utm,
    pixel_res,
    grid,
    terrain_map_path,
    blockmesh_path,
    output_dir,
    metadata_path,
) -> None:
    """Save pipeline metadata to a JSON file.

    Args:
        dem_path: Path to the DEM input file.
        rmap_path: Optional path to the roughness map input file.
        terrain_config: TerrainConfig instance.
        grid_config: GridConfig instance.
        mesh_config: MeshConfig instance.
        boundary_config: BoundaryConfig instance.
        visualization_config: VisualizationConfig instance.
        elevation_data: Raw elevation array before boundary treatment.
        treated_elevation: Elevation array after boundary treatment.
        transform: Affine transform for the processed raster.
        crs: Coordinate reference system of the processed raster.
        min_elevation: Minimum terrain elevation (metres).
        centre_utm: (x, y) UTM centre coordinates.
        pixel_res: Pixel resolution tuple (x_res, y_res).
        grid: PyVista StructuredGrid surface mesh.
        terrain_map_path: Path to the saved terrain NPZ map.
        blockmesh_path: Path to the saved blockMeshDict, or None.
        output_dir: Pipeline output directory path.
        metadata_path: Destination path for the JSON metadata file.
    """
    # Delay import to avoid circular imports at module level
    from terrain_mesh import __version__

    output_dir = Path(output_dir)

    # Store only the filename for machine-specific input paths
    dem_filename = Path(dem_path).name if dem_path else None
    roughness_filename = Path(rmap_path).name if rmap_path else None

    def _relative(path):
        """Return path relative to output_dir, or just the filename if that fails."""
        if path is None:
            return None
        try:
            return str(Path(path).relative_to(output_dir))
        except ValueError:
            return Path(path).name

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "software": {
            "name": "terrain_following_mesh_generator",
            "version": __version__,
            "description": (
                "Structured terrain-following mesh generator for OpenFOAM "
                "atmospheric boundary layer (ABL) simulations."
            ),
            "repository": "https://github.com/souravsud/terrain_following_mesh_generator",
        },

        "input_files": {
            "dem_filename": dem_filename,
            "roughness_filename": roughness_filename,
        },

        "output_files": {
            "terrain_map": _relative(terrain_map_path),
            "blockmesh_dict": _relative(blockmesh_path),
            "metadata_file": _relative(metadata_path),
        },

        "configurations": {
            "terrain": {
                "center_lat": terrain_config.center_lat,
                "center_lon": terrain_config.center_lon,
                "easting": centre_utm[1],
                "northing": centre_utm[0],
                "center_utm": terrain_config.center_coordinates,
                "crop_size_km": terrain_config.crop_size_km,
                "rotation_deg": terrain_config.rotation_deg,
                "smoothing_sigma": terrain_config.smoothing_sigma,
            },

            "grid": {
                "nx": grid_config.nx,
                "ny": grid_config.ny,
                "x_grading": grid_config.x_grading,
                "y_grading": grid_config.y_grading,
            },

            "mesh": {
                "domain_height_m": mesh_config.domain_height,
                "min_terrain_elevation_m": min_elevation,
                "terrain_normal_first_layer": mesh_config.terrain_normal_first_layer,
                "total_z_cells": mesh_config.total_z_cells,
                "z_grading": mesh_config.z_grading,
                "patch_types": mesh_config.patch_types,
            } if mesh_config else None,

            "boundary": {
                "aoi_fraction": boundary_config.aoi_fraction,
                "boundary_mode": boundary_config.boundary_mode,
                "flat_boundary_thickness_fraction": boundary_config.flat_boundary_thickness_fraction,
                "enabled_boundaries": boundary_config.enabled_boundaries,
                "smoothing_method": boundary_config.smoothing_method,
                "kernel_progression": boundary_config.kernel_progression,
                "base_kernel_size": boundary_config.base_kernel_size,
                "max_kernel_size": boundary_config.max_kernel_size,
                "progression_rate": boundary_config.progression_rate,
                "boundary_flatness_mode": boundary_config.boundary_flatness_mode,
                "uniform_elevation": boundary_config.uniform_elevation,
            },

            "visualization": {
                "create_plots": visualization_config.create_plots,
                "show_grid_lines": visualization_config.show_grid_lines,
                "save_high_res": visualization_config.save_high_res,
                "plot_format": visualization_config.plot_format,
                "dpi": visualization_config.dpi,
            },
        },

        "processing_results": {
            "geographic_coverage": {
                "center_lat_deg": terrain_config.center_lat,
                "center_lon_deg": terrain_config.center_lon,
                "domain_size_km": terrain_config.crop_size_km,
                "wind_direction_deg": terrain_config.rotation_deg,
            },

            "coordinate_system": {
                "crs": str(crs),
                "pixel_resolution_m": pixel_res,
                "transform": (
                    list(transform)
                    if hasattr(transform, '__iter__')
                    else str(transform)
                ),
            },

            "elevation_statistics": {
                "units": "meters above mean sea level (MSL)",
                "original": get_array_stats(elevation_data),
                "treated": get_array_stats(treated_elevation),
            },

            "grid_statistics": {
                "units": "meters",
                "number_of_points": (
                    grid.GetNumberOfPoints()
                    if hasattr(grid, 'GetNumberOfPoints') else None
                ),
                "number_of_cells": (
                    grid.GetNumberOfCells()
                    if hasattr(grid, 'GetNumberOfCells') else None
                ),
                "bounds": (
                    list(grid.GetBounds())
                    if hasattr(grid, 'GetBounds') else None
                ),
            },
        },
    }

    # Save to file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

def get_array_stats(data: np.ndarray) -> dict:
    """Extract summary statistics from a numpy array, ignoring NaN values.

    Args:
        data: Input numpy array (may contain NaN).

    Returns:
        Dict with keys ``shape``, ``min``, ``max``, ``mean``, ``std``.
    """
    return {
        "shape": list(data.shape),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data))
    }


def generate_region_coordinates(n_cells, expansion_ratio):
    """Generate coordinates within a single region [0, 1] with a given expansion ratio.

    Args:
        n_cells: Number of cells in this region.
        expansion_ratio: Ratio of last_cell_size / first_cell_size.

    Returns:
        np.ndarray: Coordinate array from 0 to 1 for this region.
    """
    if n_cells <= 1:
        return np.array([0.0, 1.0])

    # For uniform spacing (expansion_ratio ≈ 1)
    if abs(expansion_ratio - 1.0) < 1e-6:
        return np.linspace(0.0, 1.0, n_cells + 1)

    # For geometric progression:
    # cell sizes are ds, ds*r, ds*r², ..., ds*r^(n-1)
    # where r^(n-1) = expansion_ratio  =>  r = expansion_ratio^(1/(n-1))
    r = expansion_ratio ** (1.0 / (n_cells - 1))

    ds = (r - 1.0) / (r ** n_cells - 1.0) if abs(r - 1.0) >= 1e-6 else 1.0 / n_cells

    cell_sizes = ds * r ** np.arange(n_cells)
    coords = np.zeros(n_cells + 1)
    coords[1:] = np.cumsum(cell_sizes)
    return coords


def create_blockMesh_spacing(n_points, grading_spec):
    """Create variable spacing coordinates from 0 to 1 using blockMesh-style grading.

    Args:
        n_points: Total number of points (cells + 1).
        grading_spec: List of ``(length_fraction, cell_fraction, expansion_ratio)``
            tuples. ``length_fraction`` is the fraction of domain length for the
            region, ``cell_fraction`` is the fraction of total cells, and
            ``expansion_ratio`` is last_cell_size / first_cell_size in the region.

    Returns:
        np.ndarray: Coordinate array from 0 to 1 with blockMesh-style spacing.
    """
    total_cells = n_points - 1

    length_fractions = np.array([spec[0] for spec in grading_spec])
    cell_fractions = np.array([spec[1] for spec in grading_spec])
    expansion_ratios = np.array([spec[2] for spec in grading_spec])

    validate_grading_fractions(grading_spec, "grading_spec")

    target_cells = cell_fractions * total_cells
    actual_cells = np.round(target_cells).astype(int)

    # Adjust for rounding errors
    cell_diff = total_cells - actual_cells.sum()
    if cell_diff != 0:
        errors = target_cells - actual_cells
        indices = np.argsort(errors)[::-1] if cell_diff > 0 else np.argsort(errors)
        for i in range(abs(cell_diff)):
            actual_cells[indices[i]] += np.sign(cell_diff)

    coords = [0.0]
    current_pos = 0.0
    for length_frac, actual_cell_count, expansion_ratio in zip(length_fractions, actual_cells, expansion_ratios):
        if actual_cell_count == 0:
            continue
        region_coords = generate_region_coordinates(actual_cell_count, expansion_ratio)
        coords.extend((region_coords[1:] * length_frac + current_pos).tolist())
        current_pos += length_frac

    return np.array(coords)
