"""Boundary treatment for terrain data to improve mesh quality"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Optional, List
from .config import BoundaryConfig

class BoundaryTreatment:
    """Handle boundary smoothing and feature preservation for terrain data"""
    
    def process_boundaries(self, elevation_data: np.ndarray, crop_mask: np.ndarray, 
                          config: BoundaryConfig, rotation_deg: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply configurable boundary treatment to terrain data"""
        
        print("Applying boundary treatment...")
        
        # Calculate target elevations for each boundary
        boundary_elevations = self._calculate_boundary_target_elevations(
            elevation_data, crop_mask, config, rotation_deg)
        
        # Apply treatment based on mode
        result = self._apply_configurable_boundary_treatment(
            elevation_data, crop_mask, boundary_elevations, config, rotation_deg)
        
        # Create zone visualization data
        zones = self._create_zones_for_visualization(crop_mask, config, rotation_deg)
        
        # Final cleanup
        treated_mask = crop_mask.copy()
        result[~treated_mask] = np.nan
        
        print("Boundary treatment complete")
        return result, treated_mask, zones
    
    def _calculate_boundary_target_elevations(self, elevation_data: np.ndarray, 
                                            crop_mask: np.ndarray, config: BoundaryConfig,
                                            rotation_deg: float) -> Dict[str, float]:
        """Calculate target elevation for each boundary using filtered mean"""
        
        print("Calculating boundary target elevations...")
        
        # Get flow-oriented coordinate system
        flow_coords = self._get_flow_coordinates(crop_mask, rotation_deg)
        
        boundary_elevations = {}
        
        for direction in config.flat_boundaries:
            # Get boundary region for sampling
            boundary_mask = self._get_boundary_sampling_region(
                flow_coords, crop_mask, direction, config)
            
            if np.any(boundary_mask):
                region_elevations = elevation_data[boundary_mask]
                valid_elevations = region_elevations[~np.isnan(region_elevations)]
                
                if len(valid_elevations) > 0:
                    # Filter out extremes
                    low_thresh = np.percentile(valid_elevations, 10)
                    high_thresh = np.percentile(valid_elevations, 80)
                    filtered_elevations = valid_elevations[
                        (valid_elevations >= low_thresh) & (valid_elevations <= high_thresh)]
                    
                    if len(filtered_elevations) > 0:
                        target_elevation = np.mean(filtered_elevations)
                        boundary_elevations[direction] = target_elevation
                        print(f"{direction} boundary target: {target_elevation:.1f}m")
        
        return boundary_elevations
    
    def _apply_configurable_boundary_treatment(self, elevation_data: np.ndarray, crop_mask: np.ndarray,
                                             boundary_elevations: Dict[str, float], config: BoundaryConfig,
                                             rotation_deg: float) -> np.ndarray:
        """Apply treatment based on uniform or directional mode"""
        
        # Calculate basic zone parameters
        rows, cols = np.where(crop_mask)
        center_row, center_col = np.mean(rows), np.mean(cols)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        terrain_radius = min(max_row - min_row, max_col - min_col) / 2
        
        result = elevation_data.copy()
        
        if hasattr(config, 'boundary_mode') and config.boundary_mode == 'directional':
            return self._apply_directional_treatment(result, elevation_data, crop_mask, 
                                                   boundary_elevations, config, rotation_deg,
                                                   center_row, center_col, terrain_radius)
        else:
            return self._apply_uniform_treatment(result, elevation_data, crop_mask,
                                               boundary_elevations, config,
                                               center_row, center_col, terrain_radius)
    
    def _apply_uniform_treatment(self, result: np.ndarray, elevation_data: np.ndarray,
                               crop_mask: np.ndarray, boundary_elevations: Dict[str, float],
                               config: BoundaryConfig, center_row: float, center_col: float,
                               terrain_radius: float) -> np.ndarray:
        """Uniform treatment with circular AOI"""
        
        # Calculate radial distances from center
        y_grid, x_grid = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
        distances = np.sqrt((x_grid - center_col)**2 + (y_grid - center_row)**2)
        
        # Define zone boundaries
        aoi_radius = terrain_radius * config.aoi_fraction
        flat_thickness = terrain_radius * config.flat_fraction
        transition_start_radius = terrain_radius - flat_thickness
        
        # Create zones
        aoi_mask = (distances <= aoi_radius) & crop_mask
        flat_mask = (distances >= transition_start_radius) & crop_mask
        transition_mask = (distances > aoi_radius) & (distances < transition_start_radius) & crop_mask
        
        # Determine target elevation
        target_elevation = getattr(config, 'uniform_elevation', None)
        if target_elevation is None:
            target_elevation = min(boundary_elevations.values()) if boundary_elevations else np.nanmean(elevation_data[crop_mask])
        
        # Apply flat zone
        result[flat_mask] = target_elevation
        print(f"Applied uniform flat zone: {np.sum(flat_mask)} pixels at {target_elevation:.1f}m")
        
        # Apply transition zone
        if np.any(transition_mask):
            transition_distances = distances[transition_mask]
            normalized_dist = (transition_distances - aoi_radius) / (transition_start_radius - aoi_radius)
            blend_factors = 0.5 * (1 - np.cos(np.pi * normalized_dist))
            
            result[transition_mask] = (
                elevation_data[transition_mask] * (1 - blend_factors) +
                target_elevation * blend_factors
            )
        
        print(f"Uniform zones - AOI: {np.sum(aoi_mask)}, Transition: {np.sum(transition_mask)}, Flat: {np.sum(flat_mask)}")
        return result
    
    def _apply_directional_treatment(self, result: np.ndarray, elevation_data: np.ndarray,
                                   crop_mask: np.ndarray, boundary_elevations: Dict[str, float],
                                   config: BoundaryConfig, rotation_deg: float,
                                   center_row: float, center_col: float, terrain_radius: float) -> np.ndarray:
        """Directional treatment with square AOI"""
        
        # Get flow coordinates
        flow_coords = self._get_flow_coordinates(crop_mask, rotation_deg)
        flow_x = flow_coords['flow_x']
        flow_y = flow_coords['flow_y']
        bounds = flow_coords['bounds']
        
        # Calculate square AOI in flow coordinates
        terrain_width = bounds['max_x'] - bounds['min_x']
        terrain_height = bounds['max_y'] - bounds['min_y']
        terrain_size = min(terrain_width, terrain_height)
        
        aoi_half_size = terrain_size * config.aoi_fraction / 2
        flat_thickness = terrain_size * config.flat_fraction / 2
        
        center_flow_x = (bounds['min_x'] + bounds['max_x']) / 2
        center_flow_y = (bounds['min_y'] + bounds['max_y']) / 2
        
        # Define square AOI
        aoi_mask = ((np.abs(flow_x - center_flow_x) <= aoi_half_size) & 
                   (np.abs(flow_y - center_flow_y) <= aoi_half_size)) & crop_mask
        
        # Define flat zones for each direction
        all_flat_mask = np.zeros_like(crop_mask, dtype=bool)
        
        for direction, target_elevation in boundary_elevations.items():
            if direction == 'north':
                flat_mask = (flow_y >= (bounds['max_y'] - flat_thickness)) & crop_mask
            elif direction == 'south':
                flat_mask = (flow_y <= (bounds['min_y'] + flat_thickness)) & crop_mask
            elif direction == 'east':
                flat_mask = (flow_x >= (bounds['max_x'] - flat_thickness)) & crop_mask
            elif direction == 'west':
                flat_mask = (flow_x <= (bounds['min_x'] + flat_thickness)) & crop_mask
            else:
                continue
            
            # Apply flat elevation (but not in AOI)
            directional_flat = flat_mask & ~aoi_mask
            result[directional_flat] = target_elevation
            all_flat_mask |= directional_flat
            
            print(f"Applied {direction} flat zone: {np.sum(directional_flat)} pixels at {target_elevation:.1f}m")
        
        # Define transition zones (everything not AOI or flat)
        transition_mask = crop_mask & ~aoi_mask & ~all_flat_mask
        
        # Apply transition zones with distance-based blending
        if np.any(transition_mask):
            self._apply_directional_transitions(result, elevation_data, transition_mask, 
                                              flow_coords, boundary_elevations, config)
        
        print(f"Directional zones - AOI: {np.sum(aoi_mask)}, Transition: {np.sum(transition_mask)}, Flat: {np.sum(all_flat_mask)}")
        return result
    
    def _apply_directional_transitions(self, result: np.ndarray, elevation_data: np.ndarray,
                                     transition_mask: np.ndarray, flow_coords: Dict,
                                     boundary_elevations: Dict[str, float], config: BoundaryConfig):
        """Apply smooth transitions for directional mode"""
        
        flow_x = flow_coords['flow_x']
        flow_y = flow_coords['flow_y']
        bounds = flow_coords['bounds']
        
        # For each transition pixel, find nearest boundary and blend accordingly
        for i, j in zip(*np.where(transition_mask)):
            pixel_flow_x = flow_x[i, j]
            pixel_flow_y = flow_y[i, j]
            
            # Calculate distances to each boundary
            distances_to_boundaries = {}
            for direction in boundary_elevations.keys():
                if direction == 'north':
                    dist = bounds['max_y'] - pixel_flow_y
                elif direction == 'south':
                    dist = pixel_flow_y - bounds['min_y']
                elif direction == 'east':
                    dist = bounds['max_x'] - pixel_flow_x
                elif direction == 'west':
                    dist = pixel_flow_x - bounds['min_x']
                else:
                    continue
                distances_to_boundaries[direction] = max(0, dist)
            
            if not distances_to_boundaries:
                continue
            
            # Find closest boundary
            closest_direction = min(distances_to_boundaries.keys(), 
                                  key=lambda d: distances_to_boundaries[d])
            closest_distance = distances_to_boundaries[closest_direction]
            
            # Calculate blend factor (closer to boundary = more flat elevation)
            terrain_size = min(bounds['max_x'] - bounds['min_x'], bounds['max_y'] - bounds['min_y'])
            max_transition_distance = terrain_size * (0.5 - config.aoi_fraction/2 - config.flat_fraction/2)
            
            if max_transition_distance > 0:
                blend_factor = 1.0 - (closest_distance / max_transition_distance)
                blend_factor = np.clip(blend_factor, 0, 1)
                
                # Cosine smoothing
                blend_factor = 0.5 * (1 - np.cos(np.pi * blend_factor))
                
                target_elevation = boundary_elevations[closest_direction]
                result[i, j] = (
                    elevation_data[i, j] * (1 - blend_factor) +
                    target_elevation * blend_factor
                )
    
    def _get_flow_coordinates(self, crop_mask: np.ndarray, rotation_deg: float) -> Dict:
        """Get coordinate system relative to flow direction"""
        
        # Find center of terrain
        rows, cols = np.where(crop_mask)
        center_row, center_col = np.mean(rows), np.mean(cols)
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
        
        # Transform to FLOW coordinate system
        rotation_rad = np.deg2rad(rotation_deg)
        cos_theta = np.cos(rotation_rad)
        sin_theta = np.sin(rotation_rad)
        
        # Rotate coordinates to align with flow direction
        rel_x = x_grid - center_col
        rel_y = y_grid - center_row
        
        flow_x = cos_theta * rel_x - sin_theta * rel_y
        flow_y = sin_theta * rel_x + cos_theta * rel_y
        
        # Find bounds in flow coordinate system
        valid_flow_x = flow_x[crop_mask]
        valid_flow_y = flow_y[crop_mask]
        
        return {
            'flow_x': flow_x,
            'flow_y': flow_y,
            'center_row': center_row,
            'center_col': center_col,
            'bounds': {
                'min_x': valid_flow_x.min(),
                'max_x': valid_flow_x.max(),
                'min_y': valid_flow_y.min(),
                'max_y': valid_flow_y.max()
            }
        }
    
    def _get_boundary_sampling_region(self, flow_coords: Dict, crop_mask: np.ndarray,
                                    direction: str, config: BoundaryConfig) -> np.ndarray:
        """Get region for sampling boundary elevation using flow coordinates"""
        
        flow_x = flow_coords['flow_x']
        flow_y = flow_coords['flow_y']
        bounds = flow_coords['bounds']
        
        # Define sampling width
        sampling_fraction = 0.2 * (1 - config.aoi_fraction)
        width_x = (bounds['max_x'] - bounds['min_x']) * sampling_fraction
        width_y = (bounds['max_y'] - bounds['min_y']) * sampling_fraction
        
        if direction == 'west':
            boundary_mask = flow_x <= (bounds['min_x'] + width_x)
        elif direction == 'east':
            boundary_mask = flow_x >= (bounds['max_x'] - width_x)
        elif direction == 'south':
            boundary_mask = flow_y <= (bounds['min_y'] + width_y)
        elif direction == 'north':
            boundary_mask = flow_y >= (bounds['max_y'] - width_y)
        else:
            boundary_mask = np.zeros_like(crop_mask, dtype=bool)
        
        return boundary_mask & crop_mask
    
    def _create_zones_for_visualization(self, crop_mask: np.ndarray, config: BoundaryConfig,
                                      rotation_deg: float) -> Dict:
        """Create zone masks for visualization"""
        
        rows, cols = np.where(crop_mask)
        center_row, center_col = np.mean(rows), np.mean(cols)
        
        # Use radial distance for visualization regardless of mode
        y_grid, x_grid = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
        distances_from_center = np.sqrt((x_grid - center_col)**2 + (y_grid - center_row)**2)
        max_distance = np.max(distances_from_center[crop_mask])
        normalized_distances = distances_from_center / max_distance
        
        # Create zones based on radial distance for visualization
        aoi_threshold = config.aoi_fraction
        flat_threshold = 1.0 - config.flat_fraction
        
        zones = {
            'aoi': (normalized_distances <= aoi_threshold) & crop_mask,
            'transition': ((normalized_distances > aoi_threshold) & 
                          (normalized_distances < flat_threshold)) & crop_mask,
            'flat': (normalized_distances >= flat_threshold) & crop_mask,
            'center': (center_row, center_col)
        }
        
        return zones