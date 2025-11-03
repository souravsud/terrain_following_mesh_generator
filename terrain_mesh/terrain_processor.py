import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings
import tempfile

from .config import TerrainConfig

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

class TerrainProcessor:
    """Handle DEM loading, reprojection, and terrain extraction - now supports .dat files"""
    
    def __init__(self):
        self.utm_crs = None
        self.original_crs = None
        self.is_dat_file = False
        self.dat_metadata = {}
        self.centre_utm = None
        self.expanded_bounds = None
        
    def load_and_prepare_raster_data(self, dem_path: Union[str, Path], rmap_path: Optional[Union[str, Path]] = None) -> str:
        """Load DEM and reproject to UTM if needed"""
        dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
            
        # Check if it's a DAT file
        if dem_path.suffix.lower() == '.dat':
            print("Detected DAT file - will skip reprojection and use UTM coordinates directly")
            self.is_dat_file = True
            
            # For DAT files, we can return the path directly
            # The actual loading will happen in extract_rotated_terrain
            return str(dem_path)
        else:
            print("Detected GeoTIFF file - will handle reprojection if needed")
            self.is_dat_file = False
            return str(dem_path)
    
    def extract_rotated_terrain(self, dem_path: str, config: TerrainConfig):
        """Extract rotated crop using TerrainConfig - now handles DAT files"""
        
        # For DAT files, use UTM coordinates directly
        if self.is_dat_file:
            # Use the center coordinates from the DAT metadata if available
            if hasattr(config, 'center_coordinates') and config.center_coordinates:
                center_x, center_y = config.center_coordinates
                print(f"Using provided UTM coordinates: ({center_x}, {center_y})")
            else:
                # Use the center of the DAT file domain
                if not self.dat_metadata:
                    # Load metadata if not already loaded
                    _, _, _, _ = self.load_dat_file(dem_path)
                
                center_x, center_y = self.dat_metadata['center_utm']
                print(f"Using DAT file center: ({center_x}, {center_y})")
                
            self.centre_utm = (center_x, center_y)
            
            elevation_data, transform, crs, pixel_res, crop_mask = self.crop_rotated_dem_dat(
                dem_path, center_x, center_y, config.crop_size_km, config.rotation_deg
            )
            self.original_crs = crs
        else:
            #GeoTIFF logic
            elevation_data, transform, crs, pixel_res, crop_mask = self.crop_rotated_dem_geotiff(
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
    
    def extract_rotated_rmap(self, rmap_path: str, config: TerrainConfig) -> Tuple[np.ndarray, object]:
        """
        Crop roughness map using same geographic parameters as DEM.
        Uses centre_utm and expanded_bounds calculated during DEM extraction.
        """
        if self.centre_utm is None or self.expanded_bounds is None:
            raise ValueError("Must call extract_rotated_terrain() before extract_rotated_rmap()")
        
        center_utm_x, center_utm_y = self.centre_utm
        crop_size_m = config.crop_size_km * 1000
        rotation_deg = config.rotation_deg
        rmap_path = Path(rmap_path)
        
        # Detect file format
        if rmap_path.suffix.lower() == '.nc':
            print("Cropping NetCDF roughness map...")
            return self._crop_roughness_netcdf(rmap_path, center_utm_x, center_utm_y, 
                                            crop_size_m, rotation_deg, self.expanded_bounds)
        elif rmap_path.suffix.lower() in ['.tif', '.tiff']:
            print("Cropping GeoTIFF roughness map...")
            return self._crop_roughness_geotiff(rmap_path, center_utm_x, center_utm_y, 
                                                crop_size_m, rotation_deg, self.expanded_bounds)
        else:
            raise ValueError(f"Unsupported roughness map format: {rmap_path.suffix}")

    #helper methods for DAT file handling
    
    def load_dat_file(self, dat_path: Union[str, Path]) -> Tuple[np.ndarray, object, tuple, tuple]:
        """
        Load .dat file and convert to format compatible with existing pipeline
        Returns: (elevation_grid, transform, bounds, pixel_resolution)
        """
        dat_path = Path(dat_path)
        print(f"Loading DAT file: {dat_path}")
        
        # Read grid dimensions from header
        with open(dat_path, 'r') as f:
            first_line = f.readline().strip()
            cols, rows = map(int, first_line.split())
            print(f"DAT grid: {cols} x {rows}")
        
        # Load XYZ data
        data = np.loadtxt(dat_path, skiprows=1)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        
        # Handle no-data values if present
        if np.any(z == -9999):
            print(f"Found {np.sum(z == -9999)} no-data values (-9999)")
            z[z == -9999] = np.nan
        
        # Calculate bounds and resolution
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_res = (x_max - x_min) / (cols - 1)
        y_res = (y_max - y_min) / (rows - 1)
        
        print(f"Bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
        print(f"Resolution: {x_res:.3f} x {y_res:.3f} meters")
        
        # Reshape to grid (assuming row-major order)
        try:
            elevation_grid = z.reshape(rows, cols)
        except ValueError:
            print("Warning: Reshape failed, trying column-major order")
            elevation_grid = z.reshape(cols, rows).T
        
        # Create rasterio-compatible transform
        transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)
        bounds = (x_min, y_min, x_max, y_max)
        pixel_res = (x_res, y_res)
        
        # Store metadata for later use
        self.dat_metadata = {
            'bounds': bounds,
            'shape': elevation_grid.shape,
            'transform': transform,
            'resolution': pixel_res,
            'center_utm': ((x_min + x_max) / 2, (y_min + y_max) / 2)
        }
        
        return elevation_grid, transform, bounds, pixel_res
    
    def crop_rotated_dem_dat(self, dem_path, center_utm_x, center_utm_y, crop_size_km, rotation_deg=0):
        """
        Crop DAT file around UTM coordinates (no lat/lon conversion needed)
        """
        # Load the DAT data (we've already processed it in load_and_prepare_dem)
        elevation_grid, transform, bounds, pixel_res = self.load_dat_file(dem_path)
        
        crop_size_m = crop_size_km * 1000
        
        # Calculate expanded bounds for rotation
        buffer_size = crop_size_m * np.sqrt(2) / 2
        
        self.expanded_bounds = [
            center_utm_x - buffer_size,  # left
            center_utm_y - buffer_size,  # bottom
            center_utm_x + buffer_size,  # right
            center_utm_y + buffer_size   # top
        ]
        
        print(f"DAT crop - Center: ({center_utm_x:.2f}, {center_utm_y:.2f})")
        print(f"DAT crop - Expanded bounds: {self.expanded_bounds}")
        
        # Convert bounds to pixel coordinates
        x_min, y_min, x_max, y_max = bounds
        
        left_px = max(0, int((self.expanded_bounds[0] - x_min) / pixel_res[0]))
        right_px = min(elevation_grid.shape[1], int((self.expanded_bounds[2] - x_min) / pixel_res[0]))
        bottom_px = max(0, int((y_max - self.expanded_bounds[3]) / pixel_res[1]))
        top_px = min(elevation_grid.shape[0], int((y_max - self.expanded_bounds[1]) / pixel_res[1]))
        
        print(f"DAT crop - Pixel window: ({left_px}, {bottom_px}, {right_px}, {top_px})")
        
        # Extract the region
        expanded_data = elevation_grid[bottom_px:top_px, left_px:right_px]
        
        if expanded_data.size == 0:
            raise ValueError("Crop area is outside the DAT file bounds")
        
        # Create coordinate grids for the cropped region
        nrows, ncols = expanded_data.shape
        x_coords = np.linspace(
            x_min + left_px * pixel_res[0], 
            x_min + right_px * pixel_res[0], 
            ncols
        )
        y_coords = np.linspace(
            y_max - bottom_px * pixel_res[1], 
            y_max - top_px * pixel_res[1], 
            nrows
        )
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Create rotated crop mask
        print(f"Creating rotated crop mask (rotation: {rotation_deg}°)...")
        crop_mask = self.create_rotated_crop_mask(center_utm_x, center_utm_y, crop_size_m, rotation_deg, x_grid, y_grid)
        
        # Apply mask
        cropped_data = expanded_data.copy()
        cropped_data[~crop_mask] = np.nan
        
        # Create transform for the cropped region
        expanded_transform = from_bounds(
            x_coords[0], y_coords[-1], x_coords[-1], y_coords[0], 
            ncols, nrows
        )
        
        # Fake CRS for consistency (not used for DAT files)
        fake_crs = CRS.from_epsg(32610)
        
        print(f"DAT crop completed. Valid pixels: {np.sum(crop_mask)} / {crop_mask.size}")
        
        return cropped_data, expanded_transform, fake_crs, pixel_res, crop_mask
    
    def crop_rotated_dem_geotiff(self, dem_path, center_lat, center_lon, crop_size_km, rotation_deg=0, utm_crs=None):
        """
        Create a rotated crop of a DEM around a specified center point.
        Modified to handle DAT files with UTM coordinates directly.
        """
        
        # Original GeoTIFF logic
        with rasterio.open(dem_path) as src:
            # If no UTM CRS provided, determine it
            if utm_crs is None:
                utm_crs = self.get_utm_crs(center_lon, center_lat)
            
            # Convert center point to UTM if the DEM is in UTM
            if src.crs != CRS.from_epsg(4326):
                # Assume DEM is already in UTM
                center_utm_x, center_utm_y = self.latlon_to_utm(center_lat, center_lon, src.crs)
                self.centre_utm = (center_utm_x, center_utm_y)
            else:
                # DEM is in geographic coordinates, need to reproject first
                print("DEM appears to be in geographic coordinates. Reprojecting...")
                utm_dem_path, utm_crs = self.reproject_to_utm(dem_path)
                return self.crop_rotated_dem_geotiff(utm_dem_path, center_lat, center_lon, crop_size_km, rotation_deg, utm_crs)
            
            #saving crs data for roughness map processing
            self.original_crs = src.crs
            
            crop_size_m = crop_size_km * 1000
            
            # Calculate expanded bounds to ensure we capture all rotated pixels
            # For a rotated square, the diagonal is sqrt(2) times the side length
            buffer_size = crop_size_m * np.sqrt(2) / 2
            
            self.expanded_bounds = [
                center_utm_x - buffer_size,  # left
                center_utm_y - buffer_size,  # bottom
                center_utm_x + buffer_size,  # right
                center_utm_y + buffer_size   # top
            ]
            
            print(f"Expanded bounds for rotation (UTM): {self.expanded_bounds}")
            
            # Convert bounds to pixel coordinates
            left_px = int((self.expanded_bounds[0] - src.bounds.left) / src.res[0])
            right_px = int((self.expanded_bounds[2] - src.bounds.left) / src.res[0])
            bottom_px = int((src.bounds.top - self.expanded_bounds[3]) / src.res[1])
            top_px = int((src.bounds.top - self.expanded_bounds[1]) / src.res[1])
            
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
    
    def latlon_to_utm(self, lat, lon, utm_crs):
        """
        Convert lat/lon coordinates to UTM coordinates.
        """
        # Create transformer from WGS84 to UTM
        transformer = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
        utm_x, utm_y = transformer.transform(lon, lat)
        return utm_x, utm_y
    
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
    
    def _crop_roughness_netcdf(self, rmap_path, center_utm_x, center_utm_y, 
                            crop_size_m, rotation_deg, expanded_bounds):
        """Crop NetCDF roughness map (for DAT DEM)"""
        
        # Load NetCDF
        ds = xr.open_dataset(rmap_path)
        print(f"Loading roughness map from: {rmap_path}")
        
        # Get z0 data and 2D coordinate arrays
        z0_data = ds['z0'].values  # (j, i) array
        x_2d = ds['x'].values      # (j, i) array - UTM x coordinates
        y_2d = ds['y'].values      # (j, i) array - UTM y coordinates
        
        print(f"Roughness map shape: {z0_data.shape}")
        print(f"Roughness map bounds: X[{x_2d.min():.2f}, {x_2d.max():.2f}], Y[{y_2d.min():.2f}, {y_2d.max():.2f}]")
        print(f"Crop expanded bounds: {expanded_bounds}")
        
        # Find pixels within expanded bounds (coarse filter)
        mask_bounds = (
            (x_2d >= expanded_bounds[0]) & 
            (x_2d <= expanded_bounds[2]) &
            (y_2d >= expanded_bounds[1]) & 
            (y_2d <= expanded_bounds[3])
        )
        
        # Get bounding box of valid pixels
        rows, cols = np.where(mask_bounds)
        if len(rows) == 0:
            raise ValueError("Roughness map does not cover the crop area")
        
        j_min, j_max = rows.min(), rows.max() + 1
        i_min, i_max = cols.min(), cols.max() + 1
        
        # Extract region
        z0_crop = z0_data[j_min:j_max, i_min:i_max]
        x_crop = x_2d[j_min:j_max, i_min:i_max]
        y_crop = y_2d[j_min:j_max, i_min:i_max]
        
        print(f"Cropped to indices: j[{j_min}:{j_max}], i[{i_min}:{i_max}]")
        print(f"Cropped shape: {z0_crop.shape}")
        
        # Apply rotation mask (REUSE existing function)
        crop_mask = self.create_rotated_crop_mask(
            center_utm_x, center_utm_y, crop_size_m, rotation_deg, x_crop, y_crop
        )
        
        # Apply mask
        roughness_data = z0_crop.copy()
        roughness_data[~crop_mask] = np.nan
        
        # Calculate approximate resolution for transform
        # Take resolution from center of cropped region
        mid_j, mid_i = z0_crop.shape[0] // 2, z0_crop.shape[1] // 2
        if mid_i < z0_crop.shape[1] - 1 and mid_j < z0_crop.shape[0] - 1:
            x_res = abs(x_crop[mid_j, mid_i + 1] - x_crop[mid_j, mid_i])
            y_res = abs(y_crop[mid_j + 1, mid_i] - y_crop[mid_j, mid_i])
        else:
            # Fallback: use overall bounds
            x_res = (x_crop.max() - x_crop.min()) / (z0_crop.shape[1] - 1)
            y_res = (y_crop.max() - y_crop.min()) / (z0_crop.shape[0] - 1)
        
        # Create transform using bounds of cropped data
        nrows, ncols = z0_crop.shape
        transform = from_bounds(
            x_crop.min(), y_crop.min(), x_crop.max(), y_crop.max(), 
            ncols, nrows
        )
        
        print(f"Roughness resolution: {x_res:.2f} x {y_res:.2f} m")
        print(f"Valid roughness pixels: {np.sum(crop_mask)} / {crop_mask.size}")
        
        ds.close()
        return roughness_data, transform

    def _crop_roughness_geotiff(self, rmap_path, center_utm_x, center_utm_y, 
                            crop_size_m, rotation_deg, expanded_bounds):
        """Crop GeoTIFF roughness map (for GeoTIFF DEM)"""
        
        with rasterio.open(rmap_path) as src:
            print(f"Loading roughness GeoTIFF from: {rmap_path}")
            print(f"Roughness map CRS: {src.crs}")
            print(f"Roughness map bounds: {src.bounds}")
            
            # Check if roughness map needs reprojection
            if src.crs.is_geographic:
                # Geographic coordinates - need to reproject to UTM
                print("Roughness map is in geographic coordinates. Reprojecting to UTM...")
                
                # Use the same UTM CRS as the DEM
                if self.original_crs is None:
                    raise ValueError("DEM CRS not available. Process DEM before roughness map.")
                
                # Reproject roughness map
                utm_rmap_path, _ = self.reproject_to_utm(rmap_path)
                
                # Recursively call with reprojected file
                return self._crop_roughness_geotiff(utm_rmap_path, center_utm_x, center_utm_y, 
                                                crop_size_m, rotation_deg, expanded_bounds)
            
            # Roughness map is already in projected coordinates (assumed UTM-compatible)
            print("Roughness map is in projected coordinates")
            
            # Convert expanded bounds (UTM) to pixel coordinates
            left_px = int((expanded_bounds[0] - src.bounds.left) / src.res[0])
            right_px = int((expanded_bounds[2] - src.bounds.left) / src.res[0])
            bottom_px = int((src.bounds.top - expanded_bounds[3]) / src.res[1])
            top_px = int((src.bounds.top - expanded_bounds[1]) / src.res[1])
            
            # Clamp to valid range
            left_px = max(0, left_px)
            right_px = min(src.width, right_px)
            bottom_px = max(0, bottom_px)
            top_px = min(src.height, top_px)
            
            print(f"Roughness pixel window: ({left_px}, {bottom_px}, {right_px}, {top_px})")
            
            # Read data
            window = rasterio.windows.Window.from_slices((bottom_px, top_px), (left_px, right_px))
            expanded_data = src.read(1, window=window)
            
            if expanded_data.size == 0:
                raise ValueError("Roughness crop area is empty. Check coordinate alignment with DEM.")
            
            # Get transform
            expanded_transform = rasterio.windows.transform(window, src.transform)
            
            # Create coordinate grids
            nrows, ncols = expanded_data.shape
            x_coords = np.arange(ncols) * src.res[0] + expanded_transform.c
            y_coords = np.arange(nrows) * (-src.res[1]) + expanded_transform.f
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            
            print(f"Roughness crop shape: {expanded_data.shape}")
            print(f"Roughness crop bounds: X[{x_coords[0]:.2f}, {x_coords[-1]:.2f}], Y[{y_coords[-1]:.2f}, {y_coords[0]:.2f}]")
            
            # Apply rotation mask
            crop_mask = self.create_rotated_crop_mask(
                center_utm_x, center_utm_y, crop_size_m, rotation_deg, x_grid, y_grid
            )
            
            # Apply mask
            cropped_data = expanded_data.copy()
            cropped_data[~crop_mask] = np.nan
            
            print(f"Roughness resolution: {src.res[0]:.2f} x {abs(src.res[1]):.2f} m")
            print(f"Valid roughness pixels: {np.sum(crop_mask)} / {crop_mask.size}")
            
            return cropped_data, expanded_transform