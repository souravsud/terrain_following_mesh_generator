import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Union, Tuple
import warnings

from .config import TerrainConfig

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

class TerrainProcessor:
    """Handle DEM loading, reprojection, and terrain extraction"""
    
    def __init__(self):
        self.utm_crs = None
        self.original_crs = None
    
    def get_utm_crs(self, longitude, latitude):
        """
        Determine the appropriate UTM CRS for given coordinates.
        """
        # Calculate UTM zone
        utm_zone = int((longitude + 180) / 6) + 1
        
        # Determine hemisphere
        if latitude >= 0:
            epsg_code = 32600 + utm_zone  # Northern hemisphere
        else:
            epsg_code = 32700 + utm_zone  # Southern hemisphere
        
        return CRS.from_epsg(epsg_code)
    
    def reproject_to_utm(self, input_path, output_path=None):
        """
        Reproject a DEM from geographic coordinates to UTM projection.
        """
        with rasterio.open(input_path) as src:
            # Get the center coordinates to determine UTM zone
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            
            # Get appropriate UTM CRS
            dst_crs = self.get_utm_crs(center_lon, center_lat)
            
            print(f"Reprojecting to {dst_crs}")
            
            # Calculate transform and new dimensions
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            
            print(f"Transform: a={transform.a}, e={transform.e}, c={transform.c}, f={transform.f}")

            # Define output profile
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            if output_path is None:
                # Create temporary file name
                base_name = os.path.splitext(input_path)[0]
                output_path = f"{base_name}_utm.tif"
            
            # Reproject and save
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)
            
            print(f"UTM reprojection saved to: {output_path}")
            return output_path, dst_crs
        
    def latlon_to_utm(self, lat, lon, utm_crs):
        """
        Convert lat/lon coordinates to UTM coordinates.
        """
        # Create transformer from WGS84 to UTM
        transformer = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
        utm_x, utm_y = transformer.transform(lon, lat)
        return utm_x, utm_y

    def crop_dem_around_point_rotated(self, dem_path, center_lat, center_lon, crop_size_km, rotation_deg=0, utm_crs=None):
        """
        Create a rotated crop of a DEM around a specified center point.
        """
        with rasterio.open(dem_path) as src:
            # If no UTM CRS provided, determine it
            if utm_crs is None:
                utm_crs = self.get_utm_crs(center_lon, center_lat)
            
            # Convert center point to UTM if the DEM is in UTM
            if src.crs != CRS.from_epsg(4326):
                # Assume DEM is already in UTM
                center_utm_x, center_utm_y = self.latlon_to_utm(center_lat, center_lon, src.crs)
            else:
                # DEM is in geographic coordinates, need to reproject first
                print("DEM appears to be in geographic coordinates. Reprojecting...")
                utm_dem_path, utm_crs = self.reproject_to_utm(dem_path)
                return self.crop_dem_around_point_rotated(utm_dem_path, center_lat, center_lon, crop_size_km, rotation_deg, utm_crs)
            
            crop_size_m = crop_size_km * 1000
            
            # Calculate expanded bounds to ensure we capture all rotated pixels
            # For a rotated square, the diagonal is sqrt(2) times the side length
            buffer_size = crop_size_m * np.sqrt(2) / 2
            
            expanded_bounds = [
                center_utm_x - buffer_size,  # left
                center_utm_y - buffer_size,  # bottom
                center_utm_x + buffer_size,  # right
                center_utm_y + buffer_size   # top
            ]
            
            print(f"Expanded bounds for rotation (UTM): {expanded_bounds}")
            
            # Convert bounds to pixel coordinates
            left_px = int((expanded_bounds[0] - src.bounds.left) / src.res[0])
            right_px = int((expanded_bounds[2] - src.bounds.left) / src.res[0])
            bottom_px = int((src.bounds.top - expanded_bounds[3]) / src.res[1])
            top_px = int((src.bounds.top - expanded_bounds[1]) / src.res[1])
            
            # Ensure we don't go outside the image bounds
            left_px = max(0, left_px)
            right_px = min(src.width, right_px)
            bottom_px = max(0, bottom_px)
            top_px = min(src.height, top_px)
            
            print(f"Expanded pixel window: ({left_px}, {bottom_px}, {right_px}, {top_px})")
            
            # Read the expanded data
            window = rasterio.windows.Window.from_slices((bottom_px, top_px), (left_px, right_px))
            expanded_data = src.read(1, window=window)
            
            if expanded_data.size == 0:
                raise ValueError("Expanded crop area is empty. Check your coordinates and crop size.")
            
            # Calculate transform for expanded data
            expanded_transform = rasterio.windows.transform(window, src.transform)
            
            # Create coordinate arrays for the expanded data
            nrows, ncols = expanded_data.shape
            
            # Create x, y coordinate arrays in UTM
            x_coords = np.arange(ncols) * src.res[0] + expanded_transform.c
            y_coords = np.arange(nrows) * (-src.res[1]) + expanded_transform.f  # Note: y resolution is typically negative
            
            # Create coordinate grids
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            
            # Create the rotated crop mask
            print(f"Creating rotated crop mask (rotation: {rotation_deg}°)...")
            crop_mask = self.create_rotated_crop_mask(center_utm_x, center_utm_y, crop_size_m, rotation_deg, x_grid, y_grid)
            
            # Apply mask to elevation data
            cropped_data = expanded_data.copy()
            cropped_data[~crop_mask] = np.nan
            
            print(f"Rotated crop completed. Valid pixels: {np.sum(crop_mask)} / {crop_mask.size}")
            
            return cropped_data, expanded_transform, src.crs, src.res, crop_mask
    
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
    
    def create_rotated_crop_mask(self, center_x, center_y, crop_size_m, rotation_deg, x_coords, y_coords):
        """
        Create a mask for a rotated rectangular crop.
        """
        # Convert rotation to radians
        rotation_rad = np.deg2rad(rotation_deg-90)
        
        # Half dimensions of the crop
        half_size = crop_size_m / 2
        
        # Create coordinate grids relative to center
        rel_x = x_coords - center_x
        rel_y = y_coords - center_y
        
        # Apply inverse rotation to coordinates (rotate coordinate system, not the crop)
        cos_theta = np.cos(-rotation_rad)
        sin_theta = np.sin(-rotation_rad)
        
        rotated_x = rel_x * cos_theta - rel_y * sin_theta
        rotated_y = rel_x * sin_theta + rel_y * cos_theta
        
        # Check if points fall within the rectangular bounds
        mask = ((np.abs(rotated_x) <= half_size) & (np.abs(rotated_y) <= half_size))
        
        return mask
    
    def load_and_prepare_dem(self, dem_path: Union[str, Path]) -> str:
        """Load DEM and reproject to UTM if needed"""
        dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        return str(dem_path)

    def extract_rotated_terrain(self, dem_path: str, config: TerrainConfig):
        """Extract rotated crop using TerrainConfig - handles reprojection internally"""
        elevation_data, transform, crs, pixel_res, crop_mask = self.crop_dem_around_point_rotated(
            dem_path, 
            config.center_lat, 
            config.center_lon, 
            config.crop_size_km, 
            config.rotation_deg
        )
        
        # Apply smoothing if specified
        if config.smoothing_sigma > 0:
            elevation_data = self.smooth_terrain_for_cfd(
                elevation_data, 
                sigma=config.smoothing_sigma
            )
        
        return elevation_data, transform, crs, pixel_res, crop_mask