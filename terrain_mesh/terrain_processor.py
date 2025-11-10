from .config import TerrainConfig
from .utils import rotate_coordinates,smooth_terrain_for_cfd
from typing import Union, Tuple
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from pyproj import Transformer
import warnings

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

class TerrainProcessor:
    
    def __init__(self):
        self.centre_utm = None
        self.original_crs = None
        self.expanded_bounds = None
    
    def extract_rotated_terrain(self, dem_path: str, config: TerrainConfig):
        """
        Extract rotated terrain crop using TerrainConfig.
        Simplified - delegates to master function.
        """
        # Get center coordinates (from config or metadata)
        if config.center_coordinates:
            # User provided UTM coordinates directly
            center_utm = config.center_coordinates
        else:
            # Load from metadata JSON if available
            metadata_path = Path(dem_path).with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    center_utm = tuple(metadata['center_utm'])
                    print(f"Loaded center UTM from metadata: {center_utm}")
            else:
                # Fallback: convert lat/lon to UTM
                print("Warning: No metadata found. Converting lat/lon to UTM...")
                utm_crs = self.get_utm_crs(config.center_lon, config.center_lat)
                center_utm = self.latlon_to_utm(config.center_lat, config.center_lon, utm_crs)
        
        self.centre_utm = center_utm
        
        # Crop using master function
        elevation_data, transform, crs, pixel_res, crop_mask = self.crop_and_rotate_raster(
            raster_path=dem_path,
            center_utm=center_utm,
            crop_size_km=config.crop_size_km,
            rotation_deg=config.rotation_deg
        )
        
        self.original_crs = crs
        
        # Apply smoothing if specified
        if config.smoothing_sigma > 0:
            elevation_data = self.smooth_terrain_for_cfd(
                elevation_data, 
                sigma=config.smoothing_sigma
            )
        
        return elevation_data, transform, crs, pixel_res, crop_mask, center_utm

    def extract_rotated_rmap(self, rmap_path: str, config: TerrainConfig) -> Tuple[np.ndarray, object]:
        """
        Crop roughness map using same parameters as DEM.
        Simplified - uses master function.
        """
        if self.centre_utm is None:
            raise ValueError("Must call extract_rotated_terrain() before extract_rotated_rmap()")
        
        print("Cropping roughness map...")
        
        roughness_data, transform, _, _, _ = self.crop_and_rotate_raster(
            raster_path=rmap_path,
            center_utm=self.centre_utm,
            crop_size_km=config.crop_size_km,
            rotation_deg=config.rotation_deg
        )
        
        return roughness_data, transform
    
    def extract_circular_terrain(self, dem_path: str, config: TerrainConfig):
        """
        Extract circular terrain crop using TerrainConfig.
        Similar to extract_rotated_terrain but creates circular crop.
        
        Args:
            dem_path: Path to DEM file
            config: TerrainConfig with domain_shape='circular'
        
        Returns:
            Tuple of (elevation_data, transform, crs, pixel_res, crop_mask, center_utm)
        """
        # Get center coordinates
        if config.center_coordinates:
            center_utm = config.center_coordinates
        else:
            metadata_path = Path(dem_path).with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    center_utm = tuple(metadata['center_utm'])
                    print(f"Loaded center UTM from metadata: {center_utm}")
            else:
                print("Warning: No metadata found. Converting lat/lon to UTM...")
                utm_crs = self.get_utm_crs(config.center_lon, config.center_lat)
                center_utm = self.latlon_to_utm(config.center_lat, config.center_lon, utm_crs)
        
        self.centre_utm = center_utm
        
        # Crop using circular mask
        elevation_data, transform, crs, pixel_res, crop_mask = self._crop_circular_raster(
            raster_path=dem_path,
            center_utm=center_utm,
            diameter_km=config.crop_size_km
        )
        
        self.original_crs = crs
        
        # Apply smoothing if specified
        if config.smoothing_sigma > 0:
            from utils import smooth_terrain_for_cfd
            elevation_data = smooth_terrain_for_cfd(
                elevation_data, 
                sigma=config.smoothing_sigma
            )
        
        return elevation_data, transform, crs, pixel_res, crop_mask, center_utm


    def _crop_circular_raster(self, 
                            raster_path: Union[str, Path],
                            center_utm: Tuple[float, float],
                            diameter_km: float) -> tuple:
        """
        Crop raster to circular domain.
        
        Args:
            raster_path: Path to raster file
            center_utm: (x, y) UTM coordinates of center
            diameter_km: Circle diameter in kilometers
        
        Returns:
            (cropped_data, transform, crs, resolution, crop_mask)
        """
        raster_path = Path(raster_path)
        suffix = raster_path.suffix.lower()
        
        print(f"Processing {suffix} file for circular crop: {raster_path.name}")
        
        radius_m = (diameter_km * 1000) / 2
        
        # Format-specific adaptation
        if suffix == '.dat':
            memfile = self._adapt_dat_to_rasterio(raster_path)
            with memfile.open() as src:
                return self._crop_raster_circular_core(src, center_utm, radius_m)
        
        elif suffix == '.nc':
            memfile = self._adapt_netcdf_to_rasterio(raster_path)
            with memfile.open() as src:
                return self._crop_raster_circular_core(src, center_utm, radius_m)
        
        elif suffix in ['.tif', '.tiff']:
            with rasterio.open(str(raster_path)) as src:
                return self._crop_raster_circular_core(src, center_utm, radius_m)
        
        else:
            raise ValueError(f"Unsupported raster format: {suffix}")


    def _crop_raster_circular_core(self, src, center_utm, radius_m):
        """
        Core circular cropping logic.
        
        Args:
            src: Open rasterio DatasetReader
            center_utm: (x, y) tuple in UTM coordinates
            radius_m: Circle radius in meters
        
        Returns:
            (cropped_data, transform, crs, resolution, crop_mask)
        """
        center_utm_x, center_utm_y = center_utm
        
        # Calculate bounding box (square containing circle)
        buffer_size = radius_m
        expanded_bounds = [
            center_utm_x - buffer_size,  # left
            center_utm_y - buffer_size,  # bottom
            center_utm_x + buffer_size,  # right
            center_utm_y + buffer_size   # top
        ]
        
        self.expanded_bounds = expanded_bounds
        
        print(f"Circular crop bounds (UTM): {expanded_bounds}")
        print(f"Circle radius: {radius_m:.1f} m")
        
        # Convert to pixel coordinates
        left_px = int((expanded_bounds[0] - src.bounds.left) / src.res[0])
        right_px = int((expanded_bounds[2] - src.bounds.left) / src.res[0])
        bottom_px = int((src.bounds.top - expanded_bounds[3]) / src.res[1])
        top_px = int((src.bounds.top - expanded_bounds[1]) / src.res[1])
        
        # Clamp to bounds
        left_px = max(0, left_px)
        right_px = min(src.width, right_px)
        bottom_px = max(0, bottom_px)
        top_px = min(src.height, top_px)
        
        print(f"Pixel window: ({left_px}, {bottom_px}, {right_px}, {top_px})")
        
        # Extract window
        window = rasterio.windows.Window.from_slices(
            (bottom_px, top_px), (left_px, right_px)
        )
        expanded_data = src.read(1, window=window)
        
        if expanded_data.size == 0:
            raise ValueError("Crop area is outside raster bounds")
        
        # Calculate transform for extracted region
        expanded_transform = rasterio.windows.transform(window, src.transform)
        
        # Create coordinate grids
        nrows, ncols = expanded_data.shape
        x_coords = np.arange(ncols) * src.res[0] + expanded_transform.c
        y_coords = np.arange(nrows) * (-src.res[1]) + expanded_transform.f
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Create circular mask
        print(f"Creating circular crop mask (radius: {radius_m:.1f} m)...")
        distances = np.sqrt((x_grid - center_utm_x)**2 + (y_grid - center_utm_y)**2)
        crop_mask = distances <= radius_m
        
        # Apply mask
        cropped_data = expanded_data.copy()
        cropped_data[~crop_mask] = np.nan
        
        print(f"Circular crop completed. Valid pixels: {np.sum(crop_mask)} / {crop_mask.size}")
        print(f"Circle fill: {100 * np.sum(crop_mask) / crop_mask.size:.1f}%")
        
        return cropped_data, expanded_transform, src.crs, src.res, crop_mask

    def crop_and_rotate_raster(self, 
                           raster_path: Union[str, Path],
                           center_utm: Tuple[float, float],
                           crop_size_km: float,
                           rotation_deg: float = 0.0) -> tuple:
        """
        Master function for cropping any raster format.
        All inputs assumed to be in UTM coordinates.
        
        Args:
            raster_path: Path to raster file (.tif, .dat, .nc)
            center_utm: (x, y) UTM coordinates
            crop_size_km: Crop size in kilometers
            rotation_deg: Rotation angle in degrees
        
        Returns:
            (cropped_data, transform, crs, resolution, crop_mask)
        """
        raster_path = Path(raster_path)
        suffix = raster_path.suffix.lower()
        
        print(f"Processing {suffix} file: {raster_path.name}")
        
        crop_size_m = crop_size_km * 1000
        
        # Format-specific adaptation
        if suffix == '.dat':
            memfile = self._adapt_dat_to_rasterio(raster_path)
            with memfile.open() as src:
                return self._crop_raster_core(src, center_utm, crop_size_m, rotation_deg)
        
        elif suffix == '.nc':
            memfile = self._adapt_netcdf_to_rasterio(raster_path)
            with memfile.open() as src:
                return self._crop_raster_core(src, center_utm, crop_size_m, rotation_deg)
        
        elif suffix in ['.tif', '.tiff']:
            # GeoTIFF - direct processing (already in UTM from downloader)
            with rasterio.open(str(raster_path)) as src:
                return self._crop_raster_core(src, center_utm, crop_size_m, rotation_deg)
        
        else:
            raise ValueError(f"Unsupported raster format: {suffix}")
    
    def _crop_raster_core(self, src, center_utm, crop_size_m, rotation_deg):
        """
        Core cropping logic - works for all formats (all assumed to be in UTM).
        
        Args:
            src: Open rasterio DatasetReader
            center_utm: (x, y) tuple in UTM coordinates
            crop_size_m: Crop size in meters
            rotation_deg: Rotation angle in degrees
        
        Returns:
            (cropped_data, transform, crs, resolution, crop_mask)
        """
        center_utm_x, center_utm_y = center_utm
        
        # Calculate expanded bounds for rotation
        buffer_size = crop_size_m * np.sqrt(2) / 2
        expanded_bounds = [
            center_utm_x - buffer_size,  # left
            center_utm_y - buffer_size,  # bottom
            center_utm_x + buffer_size,  # right
            center_utm_y + buffer_size   # top
        ]
        
        self.expanded_bounds = expanded_bounds
        
        print(f"Expanded bounds (UTM): {expanded_bounds}")
        
        # Convert to pixel coordinates
        left_px = int((expanded_bounds[0] - src.bounds.left) / src.res[0])
        right_px = int((expanded_bounds[2] - src.bounds.left) / src.res[0])
        bottom_px = int((src.bounds.top - expanded_bounds[3]) / src.res[1])
        top_px = int((src.bounds.top - expanded_bounds[1]) / src.res[1])
        
        # Clamp to bounds
        left_px = max(0, left_px)
        right_px = min(src.width, right_px)
        bottom_px = max(0, bottom_px)
        top_px = min(src.height, top_px)
        
        print(f"Pixel window: ({left_px}, {bottom_px}, {right_px}, {top_px})")
        
        # Extract window
        window = rasterio.windows.Window.from_slices(
            (bottom_px, top_px), (left_px, right_px)
        )
        expanded_data = src.read(1, window=window)
        
        if expanded_data.size == 0:
            raise ValueError("Crop area is outside raster bounds")
        
        # Calculate transform for extracted region
        expanded_transform = rasterio.windows.transform(window, src.transform)
        
        # Create coordinate grids
        nrows, ncols = expanded_data.shape
        x_coords = np.arange(ncols) * src.res[0] + expanded_transform.c
        y_coords = np.arange(nrows) * (-src.res[1]) + expanded_transform.f
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Create rotation mask
        print(f"Creating rotated crop mask (rotation: {rotation_deg}°)...")
        crop_mask = self.create_rotated_crop_mask(
            center_utm_x, center_utm_y, crop_size_m, rotation_deg, x_grid, y_grid
        )
        
        # Apply mask
        cropped_data = expanded_data.copy()
        cropped_data[~crop_mask] = np.nan
        
        print(f"Rotated crop completed. Valid pixels: {np.sum(crop_mask)} / {crop_mask.size}")
        
        return cropped_data, expanded_transform, src.crs, src.res, crop_mask
    
    #Adapting function for DAT file
    def _adapt_dat_to_rasterio(self, dat_path: Path):
        """
        Convert DAT file to rasterio MemoryFile.
        DAT files are already in UTM.
        """
        print(f"Adapting DAT file: {dat_path}")
        
        # Read DAT file
        with open(dat_path, 'r') as f:
            first_line = f.readline().strip()
            cols, rows = map(int, first_line.split())
        
        data = np.loadtxt(dat_path, skiprows=1)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        
        # Handle no-data
        if np.any(z == -9999):
            z[z == -9999] = np.nan
        
        # Calculate bounds and resolution
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_res = (x_max - x_min) / (cols - 1)
        y_res = (y_max - y_min) / (rows - 1)
        
        # Reshape to grid
        try:
            elevation_grid = z.reshape(rows, cols)
        except ValueError:
            elevation_grid = z.reshape(cols, rows).T
        
        transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)
        
        # Assume UTM Zone (you can make this configurable)
        # For now, use a placeholder - will be overridden by actual UTM
        utm_crs = CRS.from_epsg(32610)  # Adjust as needed
        
        memfile = MemoryFile()
        with memfile.open(
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype=elevation_grid.dtype,
            crs=utm_crs,
            transform=transform
        ) as dataset:
            dataset.write(elevation_grid, 1)
        
        return memfile
    
    #Adapting function for DAT file
    def _adapt_netcdf_to_rasterio(self, nc_path: Path):
        """
        Convert NetCDF file to rasterio MemoryFile.
        NC files are already in UTM with 2D coordinate arrays.
        """

        
        print(f"Adapting NetCDF file: {nc_path}")
        
        ds = xr.open_dataset(nc_path)
        
        # Extract data and coordinates
        z0_data = ds['z0'].values  # Assuming 'z0' variable
        x_2d = ds['x'].values      # 2D UTM x coordinates
        y_2d = ds['y'].values      # 2D UTM y coordinates
        
        # Calculate approximate bounds and resolution
        x_min, x_max = x_2d.min(), x_2d.max()
        y_min, y_max = y_2d.min(), y_2d.max()
        
        rows, cols = z0_data.shape
        
        transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)
        
        # Use placeholder UTM CRS
        utm_crs = CRS.from_epsg(32610)  # Adjust as needed
        
        memfile = MemoryFile()
        with memfile.open(
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype=rasterio.float32,
            crs=utm_crs,
            transform=transform
        ) as dataset:
            dataset.write(z0_data.astype(np.float32), 1)
        
        ds.close()
        return memfile
    
    #Utility functions
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
    
    def create_rotated_crop_mask(self, center_x, center_y, crop_size_m, rotation_deg, x_coords, y_coords):
        """
        Create a mask for a rotated rectangular crop.
        """
        half_size = crop_size_m / 2
        rel_x = x_coords - center_x
        rel_y = y_coords - center_y
        
        # Use helper with inverse rotation
        rotated_x, rotated_y = rotate_coordinates(rel_x, rel_y, 0, 0, rotation_deg,inverse=True)
        
        mask = ((np.abs(rotated_x) <= half_size) & (np.abs(rotated_y) <= half_size))
        return mask