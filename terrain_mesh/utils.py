import numpy as np
import os
from datetime import datetime
import json
from scipy.ndimage import gaussian_filter

def rotate_coordinates(x, y, center_x, center_y, rotation_deg, inverse=False):
    """
    Rotate coordinates around center point.

    Parameters:
    -----------
    rotation_deg : float
        Meteorological wind direction (0°=N, 90°=E, 180°=S, 270°=W)
    inverse : bool
        If True, apply inverse rotation
    """
    # Convert to rotation for domain alignment
    theta = np.radians(270 - rotation_deg)
    if inverse:
        theta = -theta

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_centered = x - center_x
    y_centered = y - center_y

    x_rot = cos_theta * x_centered - sin_theta * y_centered
    y_rot = sin_theta * x_centered + cos_theta * y_centered

    return x_rot, y_rot

def smooth_terrain_for_cfd(self, elevation_data, sigma=2.0, preserve_nan=True):
    """
    Smooth terrain data for better CFD mesh quality
    
    Parameters:
    - sigma: smoothing strength (higher = more smoothing)
    - preserve_nan: keep NaN areas (outside rotated crop) as NaN
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
            temp_array[~valid_mask] = np.mean(valid_data)  # Fill NaN with mean for smoothing
            
            # Apply smoothing
            smoothed_temp = gaussian_filter(temp_array, sigma=sigma)
            
            # Restore only valid areas
            smoothed[valid_mask] = smoothed_temp[valid_mask]
    else:
        smoothed = gaussian_filter(elevation_data, sigma=sigma)
    
    return smoothed

def write_metadata(**kwargs):
    """Save pipeline metadata to JSON file"""
    
    metadata = {
        "pipeline_info": {
            "timestamp": datetime.now().isoformat(),
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
                "original": get_array_stats(kwargs['elevation_data']),
                "treated": get_array_stats(kwargs['treated_elevation'])
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

def get_array_stats(data: np.ndarray) -> dict:
    """Helper to extract statistics from numpy array"""
    return {
        "shape": list(data.shape),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data))
    }