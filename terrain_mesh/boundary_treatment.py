"""Clean 4-zone boundary treatment with proper kernel progression and zone definitions"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter, generic_filter
from typing import Tuple, Dict, Optional, List
from .config import BoundaryConfig
from .utils import rotate_coordinates

class BoundaryTreatment:
    """4-zone boundary smoothing: AOI → Transition → Blend → Flat"""
    
    def __init__(self):
        self._pyramid_cache = {}  # Cache for smoothed pyramids
    
    def process_boundaries(self, elevation_data: np.ndarray, crop_mask: np.ndarray, 
                          config: BoundaryConfig, rotation_deg: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply 4-zone progressive smoothing boundary treatment"""
        
        print("Applying 4-zone progressive smoothing boundary treatment...")
        
        # Validate configuration
        self._validate_config(config)
        
        # Clear pyramid cache
        self._pyramid_cache.clear()
        
        # Handle NaN pixels first
        elevation_cleaned = self._fill_nan_pixels(elevation_data, crop_mask)
        
        # Calculate boundary target elevations
        boundary_elevations = self._calculate_boundary_heights_from_strips(
            elevation_cleaned, crop_mask, config, rotation_deg)
        
        # Apply 4-zone treatment
        result = self._apply_four_zone_treatment(
            elevation_cleaned, crop_mask, boundary_elevations, config, rotation_deg)
        
        # Create zone visualization
        zones = self._create_zones_for_visualization(crop_mask, config, rotation_deg)
        
        # Final cleanup
        treated_mask = crop_mask.copy()
        result[~treated_mask] = np.nan
        
        print("4-zone progressive smoothing boundary treatment complete")
        return result, boundary_elevations, treated_mask, zones
    
    def _validate_config(self, config: BoundaryConfig) -> None:
        """Validate configuration parameters"""
        
        if config.aoi_fraction <= 0 or config.aoi_fraction >= 1:
            raise ValueError("aoi_fraction must be between 0 and 1")
        
        if config.flat_boundary_thickness_fraction <= 0 or config.flat_boundary_thickness_fraction >= 1:
            raise ValueError("flat_boundary_thickness_fraction must be between 0 and 1")
        
        if config.aoi_fraction + config.flat_boundary_thickness_fraction >= 0.9:
            print(f"Warning: AOI fraction ({config.aoi_fraction}) + flat thickness ({config.flat_boundary_thickness_fraction}) "
                  f"may leave very little transition zone")
        
        if config.progression_rate <= 1:
            raise ValueError("progression_rate must be > 1 for meaningful smoothing progression")
    
    def _fill_nan_pixels(self, elevation_data: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        """Fill NaN pixels using 3x3 neighborhood mean"""
        
        elevation_cleaned = elevation_data.copy()
        nan_mask = np.isnan(elevation_cleaned) & crop_mask
        
        if not np.any(nan_mask):
            return elevation_cleaned
            
        print(f"Filling {np.sum(nan_mask)} NaN pixels...")
        
        # Simple 3x3 neighborhood filling
        
        def nan_mean_filter(values):
            valid = values[~np.isnan(values)]
            return np.mean(valid) if len(valid) > 0 else 0.0
        
        filled_values = generic_filter(elevation_cleaned, nan_mean_filter, size=3, mode='nearest')
        elevation_cleaned[nan_mask] = filled_values[nan_mask]
        
        return elevation_cleaned
    
    def _calculate_boundary_heights_from_strips(self, elevation_data: np.ndarray, 
                                              crop_mask: np.ndarray, config: BoundaryConfig,
                                              rotation_deg: float) -> Dict[str, float]:
        """Calculate boundary target elevations from outermost strips"""
        
        print("Calculating boundary heights from perimeter strips...")
        
        if config.boundary_mode == 'directional':
            return self._sample_directional_boundary_strips(elevation_data, crop_mask, config, rotation_deg)
        else:
            return self._sample_radial_boundary_strip(elevation_data, crop_mask, config)
    
    def _sample_radial_boundary_strip(self, elevation_data: np.ndarray, 
                                    crop_mask: np.ndarray, config: BoundaryConfig) -> Dict[str, float]:
        """Sample outermost annular ring"""
        
        rows, cols = np.where(crop_mask)
        center_row, center_col = np.mean(rows), np.mean(cols)
        
        y_grid, x_grid = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
        distances = np.sqrt((x_grid - center_col)**2 + (y_grid - center_row)**2)
        max_radius = np.max(distances[crop_mask])
        
        # Sample from outermost flat zone only
        total_flat_thickness = max_radius * config.flat_boundary_thickness_fraction
        true_flat_thickness = total_flat_thickness / 2
        strip_start_radius = max_radius - true_flat_thickness
        
        strip_mask = ((distances >= strip_start_radius) & (distances <= max_radius)) & crop_mask
        
        if np.any(strip_mask):
            strip_elevations = elevation_data[strip_mask]
            target_height = self._calculate_filtered_average(strip_elevations)
            print(f"Radial boundary height: {target_height:.2f}m")
            return {'uniform': target_height}
        else:
            print("Warning: No valid boundary strip pixels, using 0.0m")
            return {'uniform': 0.0}
    
    def _sample_directional_boundary_strips(self, elevation_data: np.ndarray,
                                          crop_mask: np.ndarray, config: BoundaryConfig,
                                          rotation_deg: float) -> Dict[str, float]:
        """Sample outermost rectangular strips for enabled boundaries"""
        
        flow_coords = self._get_flow_coordinates(crop_mask, rotation_deg)
        flow_x = flow_coords['flow_x']
        bounds = flow_coords['bounds']
        
        terrain_width = bounds['max_x'] - bounds['min_x']
        total_flat_thickness = terrain_width * config.flat_boundary_thickness_fraction
        true_flat_thickness = total_flat_thickness / 2
        
        boundary_elevations = {}
        
        for direction in config.enabled_boundaries:
            if direction == 'east':
                strip_mask = (flow_x >= (bounds['max_x'] - true_flat_thickness)) & crop_mask
            elif direction == 'west':
                strip_mask = (flow_x <= (bounds['min_x'] + true_flat_thickness)) & crop_mask
            else:
                print(f"Warning: Unsupported boundary direction '{direction}', skipping")
                continue
            
            if np.any(strip_mask):
                strip_elevations = elevation_data[strip_mask]
                target_height = self._calculate_filtered_average(strip_elevations)
                boundary_elevations[direction] = target_height
                print(f"{direction.capitalize()} boundary height: {target_height:.2f}m")
            else:
                print(f"Warning: No valid {direction} boundary pixels, using 0.0m")
                boundary_elevations[direction] = 0.0
        
        return boundary_elevations
    
    def _calculate_filtered_average(self, elevations: np.ndarray) -> float:
        """Calculate 10-80 percentile filtered average"""
        
        if len(elevations) == 0:
            return 0.0
        
        low_thresh = np.percentile(elevations, 10)
        high_thresh = np.percentile(elevations, 80)
        filtered = elevations[(elevations >= low_thresh) & (elevations <= high_thresh)]
        
        return np.mean(filtered) if len(filtered) > 0 else np.mean(elevations)
    
    def _apply_four_zone_treatment(self, elevation_data: np.ndarray, crop_mask: np.ndarray,
                                   boundary_elevations: Dict[str, float], config: BoundaryConfig,
                                   rotation_deg: float) -> np.ndarray:
        """Apply 4-zone treatment based on boundary mode"""
        
        if config.boundary_mode == 'directional':
            return self._apply_directional_four_zone_treatment(
                elevation_data, crop_mask, boundary_elevations, config, rotation_deg)
        else:
            return self._apply_radial_four_zone_treatment(
                elevation_data, crop_mask, boundary_elevations, config)
    
    def _apply_radial_four_zone_treatment(self, elevation_data: np.ndarray, crop_mask: np.ndarray,
                                         boundary_elevations: Dict[str, float], 
                                         config: BoundaryConfig) -> np.ndarray:
        """Apply radial 4-zone treatment"""
        
        result = elevation_data.copy()
        
        # Calculate distances from center
        rows, cols = np.where(crop_mask)
        center_row, center_col = np.mean(rows), np.mean(cols)
        
        y_grid, x_grid = np.mgrid[0:result.shape[0], 0:result.shape[1]]
        distances = np.sqrt((x_grid - center_col)**2 + (y_grid - center_row)**2)
        max_distance = np.max(distances[crop_mask])
        
        # Define zone boundaries
        aoi_radius = max_distance * config.aoi_fraction
        total_flat_thickness = max_distance * config.flat_boundary_thickness_fraction
        blend_thickness = total_flat_thickness / 2
        true_flat_thickness = total_flat_thickness / 2
        
        # Zone boundaries from outside in
        true_flat_start_radius = max_distance - true_flat_thickness
        blend_start_radius = max_distance - total_flat_thickness
        transition_end_radius = blend_start_radius
        
        # Create zone masks
        aoi_mask = (distances <= aoi_radius) & crop_mask
        transition_mask = (distances > aoi_radius) & (distances <= transition_end_radius) & crop_mask
        blend_mask = (distances > blend_start_radius) & (distances < true_flat_start_radius) & crop_mask
        true_flat_mask = (distances >= true_flat_start_radius) & crop_mask
        
        target_elevation = boundary_elevations['uniform']
        
        # Apply zones
        print(f"Preserved AOI: {np.sum(aoi_mask)} pixels")
        
        result[true_flat_mask] = target_elevation
        print(f"Applied true flat zone: {np.sum(true_flat_mask)} pixels at {target_elevation:.2f}m")
        
        if np.any(transition_mask):
            print(f"Applying progressive smoothing: {np.sum(transition_mask)} pixels...")
            transition_distances = distances[transition_mask]
            distance_factors = (transition_distances - aoi_radius) / (transition_end_radius - aoi_radius)
            result = self._apply_progressive_smoothing_to_zone(
                result, crop_mask, transition_mask, distance_factors, config)
        
        if np.any(blend_mask):
            print(f"Applying blend zone: {np.sum(blend_mask)} pixels...")
            blend_distances = distances[blend_mask]
            blend_factors = (blend_distances - blend_start_radius) / (true_flat_start_radius - blend_start_radius)
            result = self._apply_blend_to_zone(
                result, crop_mask, blend_mask, blend_factors, target_elevation, config)
        
        return result
    
    def _apply_directional_four_zone_treatment(self, elevation_data: np.ndarray, crop_mask: np.ndarray,
                                              boundary_elevations: Dict[str, float], 
                                              config: BoundaryConfig, rotation_deg: float) -> np.ndarray:
        """Apply directional 4-zone treatment"""
        
        result = elevation_data.copy()
        
        # Get flow coordinates
        flow_coords = self._get_flow_coordinates(crop_mask, rotation_deg)
        flow_x = flow_coords['flow_x']
        flow_y = flow_coords['flow_y']
        bounds = flow_coords['bounds']
        
        # Calculate AOI (square)
        terrain_width = bounds['max_x'] - bounds['min_x']
        terrain_height = bounds['max_y'] - bounds['min_y']
        terrain_size = min(terrain_width, terrain_height)
        
        aoi_half_size = terrain_size * config.aoi_fraction / 2
        center_flow_x = (bounds['min_x'] + bounds['max_x']) / 2
        center_flow_y = (bounds['min_y'] + bounds['max_y']) / 2
        
        aoi_mask = ((np.abs(flow_x - center_flow_x) <= aoi_half_size) & 
                   (np.abs(flow_y - center_flow_y) <= aoi_half_size)) & crop_mask
        
        # Calculate zone thicknesses
        total_flat_thickness = terrain_width * config.flat_boundary_thickness_fraction
        blend_thickness = total_flat_thickness / 2
        true_flat_thickness = total_flat_thickness / 2
        
        # Apply zones for each enabled boundary
        all_blend_mask = np.zeros_like(crop_mask, dtype=bool)
        all_true_flat_mask = np.zeros_like(crop_mask, dtype=bool)
        
        for direction, target_elevation in boundary_elevations.items():
            if direction == 'east':
                blend_mask = ((flow_x >= (bounds['max_x'] - total_flat_thickness)) & 
                            (flow_x < (bounds['max_x'] - true_flat_thickness))) & crop_mask & ~aoi_mask
                true_flat_mask = (flow_x >= (bounds['max_x'] - true_flat_thickness)) & crop_mask & ~aoi_mask
            elif direction == 'west':
                blend_mask = ((flow_x <= (bounds['min_x'] + total_flat_thickness)) & 
                            (flow_x > (bounds['min_x'] + true_flat_thickness))) & crop_mask & ~aoi_mask
                true_flat_mask = (flow_x <= (bounds['min_x'] + true_flat_thickness)) & crop_mask & ~aoi_mask
            else:
                continue
            
            result[true_flat_mask] = target_elevation
            all_true_flat_mask |= true_flat_mask
            all_blend_mask |= blend_mask
            
            print(f"Applied {direction} true flat zone: {np.sum(true_flat_mask)} pixels at {target_elevation:.2f}m")
        
        # Define transition zone
        transition_mask = crop_mask & ~aoi_mask & ~all_blend_mask & ~all_true_flat_mask
        
        print(f"Preserved AOI: {np.sum(aoi_mask)} pixels")
        
        # Apply transition zone progressive smoothing
        if np.any(transition_mask):
            print(f"Applying progressive smoothing: {np.sum(transition_mask)} pixels...")
            result = self._apply_directional_progressive_smoothing(
                result, crop_mask, transition_mask, flow_coords, aoi_half_size, 
                total_flat_thickness, config)
        
        # Apply blend zones
        if np.any(all_blend_mask):
            print(f"Applying blend zones: {np.sum(all_blend_mask)} pixels...")
            result = self._apply_directional_blend_zones(
                result, crop_mask, all_blend_mask, flow_coords, 
                total_flat_thickness, true_flat_thickness, boundary_elevations, config)
        
        return result
    
    def _apply_progressive_smoothing_to_zone(self, result: np.ndarray, crop_mask: np.ndarray,
                                           zone_mask: np.ndarray, distance_factors: np.ndarray,
                                           config: BoundaryConfig) -> np.ndarray:
        """Apply multi-scale progressive smoothing to a zone"""
        
        # Get or create smoothed pyramid
        pyramid_key = self._get_pyramid_cache_key(config)
        if pyramid_key not in self._pyramid_cache:
            print("Creating multi-scale pyramid...")
            self._pyramid_cache[pyramid_key] = self._create_multiscale_pyramid(result, crop_mask, config)
        
        smoothed_images = self._pyramid_cache[pyramid_key]
        
        # Process pixels in chunks
        zone_indices = np.where(zone_mask)
        zone_pixels = list(zip(zone_indices[0], zone_indices[1]))
        
        chunk_size = 10000
        num_chunks = (len(zone_pixels) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(zone_pixels))
            chunk_pixels = zone_pixels[start_idx:end_idx]
            
            for pixel_idx, (i, j) in enumerate(chunk_pixels):
                distance_factor = distance_factors[start_idx + pixel_idx]
                scale_blend = self._calculate_scale_blending(distance_factor, len(smoothed_images))
                
                blended_value = 0.0
                total_weight = 0.0
                
                for scale_idx, weight in scale_blend.items():
                    if weight > 0:
                        blended_value += smoothed_images[scale_idx][i, j] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    result[i, j] = blended_value / total_weight
        
        return result
    
    def _apply_directional_progressive_smoothing(self, result: np.ndarray, crop_mask: np.ndarray,
                                               transition_mask: np.ndarray, flow_coords: Dict,
                                               aoi_half_size: float, total_flat_thickness: float,
                                               config: BoundaryConfig) -> np.ndarray:
        """Apply directional progressive smoothing"""
        
        flow_x = flow_coords['flow_x']
        bounds = flow_coords['bounds']
        
        # Calculate distance to nearest enabled boundary
        distance_to_boundary = np.full_like(flow_x, np.inf)
        
        for direction in config.enabled_boundaries:
            if direction == 'east':
                dist = bounds['max_x'] - flow_x
            elif direction == 'west':
                dist = flow_x - bounds['min_x']
            else:
                continue
            
            closer_mask = dist < distance_to_boundary
            distance_to_boundary[closer_mask] = dist[closer_mask]
        
        # Calculate max transition distance
        terrain_width = bounds['max_x'] - bounds['min_x']
        max_transition_distance = terrain_width/2 - aoi_half_size - total_flat_thickness
        
        if max_transition_distance <= 0:
            print("Warning: Transition zone too small")
            return result
        
        # Calculate distance factors
        transition_indices = np.where(transition_mask)
        distance_factors = 1.0 - (distance_to_boundary[transition_indices] / max_transition_distance)
        distance_factors = np.clip(distance_factors, 0, 1)
        
        # Apply progressive smoothing
        return self._apply_progressive_smoothing_to_zone(
            result, crop_mask, transition_mask, distance_factors, config)
    
    def _apply_blend_to_zone(self, result: np.ndarray, crop_mask: np.ndarray,
                           blend_mask: np.ndarray, blend_factors: np.ndarray,
                           target_elevation: float, config: BoundaryConfig) -> np.ndarray:
        """Apply blending between heavy smoothing and target elevation"""
        
        # Apply maximum smoothing to blend zone
        max_kernel_size = self._calculate_max_kernel_size(config)
        heavily_smoothed = self._apply_heavy_smoothing(result, crop_mask, max_kernel_size, config)
        
        # Vectorized blend: result = smoothed * (1-t) + target * t
        result[blend_mask] = (
            heavily_smoothed[blend_mask] * (1.0 - blend_factors) +
            target_elevation * blend_factors
        )
        
        return result
    
    def _apply_directional_blend_zones(self, result: np.ndarray, crop_mask: np.ndarray,
                                     blend_mask: np.ndarray, flow_coords: Dict,
                                     total_flat_thickness: float, true_flat_thickness: float,
                                     boundary_elevations: Dict[str, float], 
                                     config: BoundaryConfig) -> np.ndarray:
        """Apply directional blend zones"""
        
        flow_x = flow_coords['flow_x']
        bounds = flow_coords['bounds']
        blend_thickness = total_flat_thickness / 2
        
        max_kernel_size = self._calculate_max_kernel_size(config)
        heavily_smoothed = self._apply_heavy_smoothing(result, crop_mask, max_kernel_size, config)
        
        for direction, target_elevation in boundary_elevations.items():
            if direction == 'east':
                directional_blend_mask = ((flow_x >= (bounds['max_x'] - total_flat_thickness)) & 
                                        (flow_x < (bounds['max_x'] - true_flat_thickness))) & blend_mask
                if np.any(directional_blend_mask):
                    blend_distances = flow_x[directional_blend_mask]
                    blend_factors = (blend_distances - (bounds['max_x'] - total_flat_thickness)) / blend_thickness
                    blend_factors = np.clip(blend_factors, 0, 1)
                    
                    result[directional_blend_mask] = (
                        heavily_smoothed[directional_blend_mask] * (1 - blend_factors) +
                        target_elevation * blend_factors)
                    
            elif direction == 'west':
                directional_blend_mask = ((flow_x <= (bounds['min_x'] + total_flat_thickness)) & 
                                        (flow_x > (bounds['min_x'] + true_flat_thickness))) & blend_mask
                if np.any(directional_blend_mask):
                    blend_distances = flow_x[directional_blend_mask]
                    blend_factors = 1.0 - ((blend_distances - (bounds['min_x'] + true_flat_thickness)) / blend_thickness)
                    blend_factors = np.clip(blend_factors, 0, 1)
                    
                    result[directional_blend_mask] = (
                        heavily_smoothed[directional_blend_mask] * (1 - blend_factors) +
                        target_elevation * blend_factors)
        
        return result
    
    def _calculate_max_kernel_size(self, config: BoundaryConfig) -> int:
        """Calculate maximum kernel size for heavy smoothing"""
        
        if config.base_kernel_size is not None:
            return max(int(config.base_kernel_size * (config.progression_rate ** 2)), 
                      config.base_kernel_size * 3)
        elif config.max_kernel_size is not None:
            return config.max_kernel_size
        else:
            return max(int(3 * (config.progression_rate ** 2)), 15)
    
    def _apply_heavy_smoothing(self, data: np.ndarray, mask: np.ndarray, 
                             kernel_size: int, config: BoundaryConfig) -> np.ndarray:
        """Apply heavy smoothing to entire image"""
        
        # Prepare data
        working_data = data.copy()
        invalid_mask = ~mask | np.isnan(working_data)
        if np.any(invalid_mask):
            valid_data = working_data[mask & ~np.isnan(working_data)]
            if len(valid_data) > 0:
                fill_value = np.median(valid_data)
                working_data[invalid_mask] = fill_value
        
        # Apply smoothing
        if config.smoothing_method == 'gaussian':
            sigma = kernel_size / 6.0
            return gaussian_filter(working_data, sigma=sigma, mode='nearest')
        elif config.smoothing_method == 'mean':
            return uniform_filter(working_data.astype(float), size=kernel_size, mode='nearest')
        elif config.smoothing_method == 'median':
            return median_filter(working_data, size=kernel_size, mode='nearest')
        else:
            return working_data.copy()
    
    def _create_multiscale_pyramid(self, data: np.ndarray, mask: np.ndarray, 
                                  config: BoundaryConfig) -> List[np.ndarray]:
        """Create pyramid of smoothed images at different scales"""
        
        # Calculate kernel size range with proper progression
        base_size, max_size = self._calculate_kernel_size_range(config)
        
        print(f"Creating pyramid: kernel range {base_size:.1f} to {max_size:.1f}")
        
        num_scales = 7
        smoothed_images = []
        
        # Prepare data
        working_data = data.copy()
        invalid_mask = ~mask | np.isnan(working_data)
        if np.any(invalid_mask):
            valid_data = working_data[mask & ~np.isnan(working_data)]
            if len(valid_data) > 0:
                fill_value = np.median(valid_data)
                working_data[invalid_mask] = fill_value
        
        for scale_idx in range(num_scales):
            scale_factor = scale_idx / (num_scales - 1)  # 0 to 1
            
            if config.kernel_progression == 'exponential':
                # Proper exponential progression from base to max
                kernel_size = base_size * ((max_size / base_size) ** scale_factor)
            else:  # linear
                kernel_size = base_size + scale_factor * (max_size - base_size)
            
            kernel_size = max(int(kernel_size), 1)
            print(f"Scale {scale_idx+1}/{num_scales}: kernel size {kernel_size}")
            
            # Apply smoothing
            if config.smoothing_method == 'gaussian':
                sigma = kernel_size / 6.0
                smoothed = gaussian_filter(working_data, sigma=sigma, mode='nearest')
            elif config.smoothing_method == 'mean':
                smoothed = uniform_filter(working_data.astype(float), size=kernel_size, mode='nearest')
            elif config.smoothing_method == 'median':
                smoothed = median_filter(working_data, size=kernel_size, mode='nearest')
            else:
                smoothed = working_data.copy()
            
            smoothed_images.append(smoothed)
        
        return smoothed_images
    
    def _calculate_kernel_size_range(self, config: BoundaryConfig) -> Tuple[float, float]:
        """Calculate proper kernel size range"""
        
        if config.base_kernel_size is not None:
            base_size = float(config.base_kernel_size)
            # Calculate max size for meaningful progression
            max_size = base_size * (config.progression_rate ** 3)  # Cube for strong progression
        elif config.max_kernel_size is not None:
            max_size = float(config.max_kernel_size)
            base_size = max_size / (config.progression_rate ** 3)
        else:
            base_size = 3.0
            max_size = base_size * (config.progression_rate ** 3)
        
        # Ensure reasonable bounds
        base_size = max(base_size, 1.0)
        max_size = max(max_size, base_size * 2)
        
        return base_size, max_size
    
    def _calculate_scale_blending(self, distance_factor: float, num_scales: int) -> Dict[int, float]:
        """Calculate blending weights between adjacent scales"""
        
        scale_position = distance_factor * (num_scales - 1)
        lower_scale = int(np.floor(scale_position))
        upper_scale = int(np.ceil(scale_position))
        
        lower_scale = np.clip(lower_scale, 0, num_scales - 1)
        upper_scale = np.clip(upper_scale, 0, num_scales - 1)
        
        if lower_scale == upper_scale:
            return {lower_scale: 1.0}
        else:
            blend_factor = scale_position - lower_scale
            return {
                lower_scale: 1.0 - blend_factor,
                upper_scale: blend_factor
            }
    
    def _get_pyramid_cache_key(self, config: BoundaryConfig) -> str:
        """Generate cache key for pyramid"""
        return f"{config.smoothing_method}_{config.kernel_progression}_{config.base_kernel_size}_{config.max_kernel_size}_{config.progression_rate}"
    
    def _get_flow_coordinates(self, crop_mask: np.ndarray, rotation_deg: float) -> Dict:
        """Get flow-aligned coordinate system"""
        
        rows, cols = np.where(crop_mask)
        center_row, center_col = np.mean(rows), np.mean(cols)
        
        y_grid, x_grid = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
        
        rel_x = x_grid - center_col
        rel_y = y_grid - center_row
        
        flow_x, flow_y = rotate_coordinates(rel_x, rel_y, 0, 0, rotation_deg, inverse=True)
        
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
    
    def _create_zones_for_visualization(self, crop_mask: np.ndarray, config: BoundaryConfig,
                                      rotation_deg: float) -> Dict:
        """Create zone masks for visualization"""
        
        rows, cols = np.where(crop_mask)
        center_row, center_col = np.mean(rows), np.mean(cols)
        
        if config.boundary_mode == 'directional':
            # Directional 4-zone visualization
            flow_coords = self._get_flow_coordinates(crop_mask, rotation_deg)
            flow_x = flow_coords['flow_x']
            flow_y = flow_coords['flow_y']
            bounds = flow_coords['bounds']
            
            terrain_width = bounds['max_x'] - bounds['min_x']
            terrain_size = min(terrain_width, bounds['max_y'] - bounds['min_y'])
            
            aoi_half_size = terrain_size * config.aoi_fraction / 2
            total_flat_thickness = terrain_width * config.flat_boundary_thickness_fraction
            true_flat_thickness = total_flat_thickness / 2
            
            center_flow_x = (bounds['min_x'] + bounds['max_x']) / 2
            center_flow_y = (bounds['min_y'] + bounds['max_y']) / 2
            
            # AOI mask
            aoi_mask = ((np.abs(flow_x - center_flow_x) <= aoi_half_size) & 
                       (np.abs(flow_y - center_flow_y) <= aoi_half_size)) & crop_mask
            
            # Boundary zone masks
            blend_mask = np.zeros_like(crop_mask, dtype=bool)
            flat_mask = np.zeros_like(crop_mask, dtype=bool)
            
            for direction in config.enabled_boundaries:
                if direction == 'east':
                    boundary_blend = ((flow_x >= (bounds['max_x'] - total_flat_thickness)) & 
                                    (flow_x < (bounds['max_x'] - true_flat_thickness))) & crop_mask
                    boundary_flat = (flow_x >= (bounds['max_x'] - true_flat_thickness)) & crop_mask
                elif direction == 'west':
                    boundary_blend = ((flow_x <= (bounds['min_x'] + total_flat_thickness)) & 
                                    (flow_x > (bounds['min_x'] + true_flat_thickness))) & crop_mask
                    boundary_flat = (flow_x <= (bounds['min_x'] + true_flat_thickness)) & crop_mask
                else:
                    continue
                    
                blend_mask |= boundary_blend
                flat_mask |= boundary_flat
            
            # Remove AOI overlap
            blend_mask = blend_mask & ~aoi_mask
            flat_mask = flat_mask & ~aoi_mask
            
            # Transition zone is everything else
            transition_mask = crop_mask & ~aoi_mask & ~blend_mask & ~flat_mask
            
        else:
            # Radial 4-zone visualization
            y_grid, x_grid = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
            distances = np.sqrt((x_grid - center_col)**2 + (y_grid - center_row)**2)
            max_distance = np.max(distances[crop_mask])
            
            aoi_radius = max_distance * config.aoi_fraction
            total_flat_thickness = max_distance * config.flat_boundary_thickness_fraction
            true_flat_thickness = total_flat_thickness / 2
            
            true_flat_start_radius = max_distance - true_flat_thickness
            blend_start_radius = max_distance - total_flat_thickness
            
            aoi_mask = (distances <= aoi_radius) & crop_mask
            transition_mask = (distances > aoi_radius) & (distances <= blend_start_radius) & crop_mask
            blend_mask = (distances > blend_start_radius) & (distances < true_flat_start_radius) & crop_mask
            flat_mask = (distances >= true_flat_start_radius) & crop_mask
        
        return {
            'aoi': aoi_mask,
            'transition': transition_mask,
            'blend': blend_mask,
            'flat': flat_mask,
            'center': (center_row, center_col)
        }