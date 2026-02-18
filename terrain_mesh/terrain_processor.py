"""Terrain data extraction and processing module.

This module handles extraction and preprocessing of Digital Elevation Model (DEM)
data from various formats (GeoTIFF, DAT, NetCDF) with support for rotation,
cropping, and coordinate transformations.
"""
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

# Constants
SQRT_2 = np.sqrt(2)  # Square root of 2 for rotation buffer calculation
NODATA_VALUE = -9999  # Standard no-data value for DAT files
UTM_ZONE_WIDTH = 6  # Width of UTM zones in degrees
WGS84_EPSG = 4326  # EPSG code for WGS84 coordinate system
UTM_NORTH_BASE = 32600  # Base EPSG for northern hemisphere UTM zones
UTM_SOUTH_BASE = 32700  # Base EPSG for southern hemisphere UTM zones


class TerrainProcessor:
    """Process and extract terrain elevation data from various raster formats.
    
    This class handles:
    - Loading DEM data from GeoTIFF, DAT, and NetCDF formats
    - Cropping terrain to specified region
    - Rotating terrain by specified angle
    - Coordinate transformations (lat/lon to UTM)
    - Terrain smoothing for CFD applications
    
    Attributes:
        centre_utm: UTM coordinates of terrain center (x, y)
        original_crs: Original coordinate reference system of input data
        expanded_bounds: Bounds of expanded region for rotation [left, bottom, right, top]
    """
    
    def __init__(self):
        self.centre_utm = None
        self.original_crs = None
        self.expanded_bounds = None
    
    def extract_rotated_terrain(
        self, dem_path: str, config: TerrainConfig
    ) -> Tuple[np.ndarray, object, CRS, Tuple[float, float], np.ndarray, Tuple[float, float]]:
        """Extract and process terrain elevation data.
        
        This method:
        1. Determines center coordinates (from config or metadata)
        2. Crops terrain to specified size
        3. Rotates terrain by specified angle
        4. Applies optional smoothing
        
        Args:
            dem_path: Path to DEM file (GeoTIFF, DAT, or NetCDF)
            config: TerrainConfig with extraction parameters
            
        Returns:
            Tuple containing:
            - elevation_data: 2D array of elevation values (meters)
            - transform: Affine transform for geospatial coordinates
            - crs: Coordinate reference system
            - pixel_res: Pixel resolution (x_res, y_res) in meters
            - crop_mask: Boolean mask indicating valid terrain region
            - center_utm: Center coordinates in UTM (x, y)
            
        Raises:
            FileNotFoundError: If DEM file doesn't exist
            ValueError: If crop region is outside raster bounds
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
        """Extract roughness map using same parameters as DEM.
        
        This method crops the roughness map to match the terrain extraction,
        ensuring spatial alignment for z0 field generation.
        
        Args:
            rmap_path: Path to roughness map file
            config: TerrainConfig with same parameters used for DEM
            
        Returns:
            Tuple of (roughness_data, transform)
            
        Raises:
            ValueError: If extract_rotated_terrain() hasn't been called first
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
    
    def crop_and_rotate_raster(
        self, 
        raster_path: Union[str, Path],
        center_utm: Tuple[float, float],
        crop_size_km: float,
        rotation_deg: float = 0.0
    ) -> Tuple[np.ndarray, object, CRS, Tuple[float, float], np.ndarray]:
        """Master function for cropping and rotating any raster format.
        
        This function handles multiple raster formats (GeoTIFF, DAT, NetCDF) and
        applies rotation by first cropping an expanded region, then masking to
        the rotated rectangle.
        
        All inputs are assumed to be in UTM coordinates.
        
        Args:
            raster_path: Path to raster file (.tif, .dat, .nc)
            center_utm: (x, y) UTM coordinates of region center
            crop_size_km: Size of region to extract (kilometers)
            rotation_deg: Rotation angle clockwise from North (degrees)
        
        Returns:
            Tuple containing:
            - cropped_data: 2D array with rotated crop (NaN outside region)
            - transform: Affine transform for the cropped region
            - crs: Coordinate reference system
            - resolution: Pixel resolution (x_res, y_res) in meters
            - crop_mask: Boolean mask indicating valid region
            
        Raises:
            ValueError: If crop area is outside raster bounds
            ValueError: If raster format is not supported
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
        buffer_size = crop_size_m * SQRT_2 / 2
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
        if np.any(z == NODATA_VALUE):
            z[z == NODATA_VALUE] = np.nan
        
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
    
    def get_utm_crs(self, longitude: float, latitude: float) -> CRS:
        """Determine the appropriate UTM CRS for given coordinates.
        
        UTM zones are 6 degrees wide, starting at -180 degrees longitude.
        Northern hemisphere uses EPSG codes 32601-32660.
        Southern hemisphere uses EPSG codes 32701-32760.
        
        Args:
            longitude: Longitude in decimal degrees (-180 to 180)
            latitude: Latitude in decimal degrees (-90 to 90)
            
        Returns:
            rasterio CRS object for the appropriate UTM zone
        """
        # Calculate UTM zone
        utm_zone = int((longitude + 180) / UTM_ZONE_WIDTH) + 1
        
        # Determine hemisphere
        if latitude >= 0:
            epsg_code = UTM_NORTH_BASE + utm_zone  # Northern hemisphere
        else:
            epsg_code = UTM_SOUTH_BASE + utm_zone  # Southern hemisphere
        
        return CRS.from_epsg(epsg_code)
    
    def latlon_to_utm(self, lat: float, lon: float, utm_crs: CRS) -> Tuple[float, float]:
        """Convert lat/lon coordinates to UTM coordinates.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            utm_crs: Target UTM coordinate reference system
            
        Returns:
            Tuple of (utm_x, utm_y) coordinates in meters
        """
        # Create transformer from WGS84 to UTM
        transformer = Transformer.from_crs(
            CRS.from_epsg(WGS84_EPSG), utm_crs, always_xy=True
        )
        utm_x, utm_y = transformer.transform(lon, lat)
        return utm_x, utm_y
    
    def create_rotated_crop_mask(
        self, 
        center_x: float, 
        center_y: float, 
        crop_size_m: float, 
        rotation_deg: float, 
        x_coords: np.ndarray, 
        y_coords: np.ndarray
    ) -> np.ndarray:
        """Create a boolean mask for a rotated rectangular crop region.
        
        Args:
            center_x: X coordinate of region center (UTM meters)
            center_y: Y coordinate of region center (UTM meters)
            crop_size_m: Size of square region (meters)
            rotation_deg: Rotation angle clockwise from North (degrees)
            x_coords: 2D array of X coordinates
            y_coords: 2D array of Y coordinates
            
        Returns:
            Boolean mask where True indicates points inside rotated rectangle
        """
        half_size = crop_size_m / 2
        rel_x = x_coords - center_x
        rel_y = y_coords - center_y
        
        # Use helper with inverse rotation
        rotated_x, rotated_y = rotate_coordinates(rel_x, rel_y, 0, 0, rotation_deg, inverse=True)
        
        mask = ((np.abs(rotated_x) <= half_size) & (np.abs(rotated_y) <= half_size))
        return mask