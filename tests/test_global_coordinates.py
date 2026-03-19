"""Tests for global-coordinate portability.

Verifies that:
- get_utm_crs() returns the correct UTM zone for a representative set of
  locations spread across the globe (all major continents / hemispheres).
- The Norway zone 32V and Svalbard zone X special cases are handled.
- lon = 180 is clamped to zone 60 (not the non-existent zone 61).
- Polar regions (lat > 84 N or lat < -80 S) raise a clear ValueError.
- TerrainConfig rejects invalid latitude / longitude values early.
- GeoTIFF rasters supplied in a geographic CRS (WGS84) are automatically
  reprojected to the site's UTM zone before cropping.
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from terrain_mesh.terrain_processor import TerrainProcessor
from terrain_mesh.config import TerrainConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wgs84_tif(lon_min, lat_min, lon_max, lat_max, ncols=50, nrows=50):
    """Create an in-memory GeoTIFF in WGS84 (EPSG:4326) with synthetic elevation."""
    elevation = np.random.uniform(100, 500, (nrows, ncols)).astype(np.float32)
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
    mf = MemoryFile()
    with mf.open(
        driver='GTiff', height=nrows, width=ncols, count=1,
        dtype='float32', crs=CRS.from_epsg(4326), transform=transform,
    ) as ds:
        ds.write(elevation, 1)
    return mf


def _make_utm_tif(x_min, y_min, x_max, y_max, epsg, ncols=50, nrows=50):
    """Create an in-memory GeoTIFF in a UTM CRS with synthetic elevation."""
    elevation = np.random.uniform(100, 500, (nrows, ncols)).astype(np.float32)
    transform = from_bounds(x_min, y_min, x_max, y_max, ncols, nrows)
    mf = MemoryFile()
    with mf.open(
        driver='GTiff', height=nrows, width=ncols, count=1,
        dtype='float32', crs=CRS.from_epsg(epsg), transform=transform,
    ) as ds:
        ds.write(elevation, 1)
    return mf


# ---------------------------------------------------------------------------
# get_utm_crs() correctness
# ---------------------------------------------------------------------------

class TestGetUtmCrs:
    """Verify correct UTM zone selection across the globe."""

    def setup_method(self):
        self.proc = TerrainProcessor()

    # --- standard cases ---

    def test_portugal_northern_hemisphere(self):
        # Portalegre, Portugal: 39.71°N, -7.73°E → zone 29N
        crs = self.proc.get_utm_crs(-7.73, 39.71)
        assert crs.to_epsg() == 32629

    def test_new_york_northern_hemisphere(self):
        # New York: 40.7°N, -74.0°W → zone 18N
        crs = self.proc.get_utm_crs(-74.0, 40.7)
        assert crs.to_epsg() == 32618

    def test_tokyo_northern_hemisphere(self):
        # Tokyo: 35.7°N, 139.7°E → zone 54N
        crs = self.proc.get_utm_crs(139.7, 35.7)
        assert crs.to_epsg() == 32654

    def test_sydney_southern_hemisphere(self):
        # Sydney: 33.9°S, 151.2°E → zone 56S
        crs = self.proc.get_utm_crs(151.2, -33.9)
        assert crs.to_epsg() == 32756

    def test_buenos_aires_southern_hemisphere(self):
        # Buenos Aires: 34.6°S, 58.4°W → zone 21S
        crs = self.proc.get_utm_crs(-58.4, -34.6)
        assert crs.to_epsg() == 32721

    def test_nairobi_equator(self):
        # Nairobi: 1.3°S, 36.8°E → zone 37S
        crs = self.proc.get_utm_crs(36.8, -1.3)
        assert crs.to_epsg() == 32737

    def test_cape_town_southern_hemisphere(self):
        # Cape Town: 33.9°S, 18.4°E → zone 34S
        crs = self.proc.get_utm_crs(18.4, -33.9)
        assert crs.to_epsg() == 32734

    def test_reykjavik_northern_hemisphere(self):
        # Reykjavik: 64.1°N, -21.9°W → zone 27N
        crs = self.proc.get_utm_crs(-21.9, 64.1)
        assert crs.to_epsg() == 32627

    # --- lon = 180 boundary ---

    def test_lon_180_northern_maps_to_zone_60(self):
        """Longitude 180° should use zone 60, not the non-existent zone 61."""
        crs = self.proc.get_utm_crs(180.0, 10.0)
        assert crs.to_epsg() == 32660  # zone 60N

    def test_lon_180_southern_maps_to_zone_60(self):
        crs = self.proc.get_utm_crs(180.0, -10.0)
        assert crs.to_epsg() == 32760  # zone 60S

    def test_lon_minus_180_maps_to_zone_1(self):
        crs = self.proc.get_utm_crs(-180.0, 10.0)
        assert crs.to_epsg() == 32601  # zone 1N

    # --- Norway zone 32V exception ---

    def test_norway_bergen_uses_zone_32(self):
        """Bergen (5.3°E, 60.4°N) lies in zone 32V, not 31V."""
        crs = self.proc.get_utm_crs(5.3, 60.4)
        assert crs.to_epsg() == 32632

    def test_norway_edge_3e_56n_uses_zone_32(self):
        """Western edge of the exception band (3°E, 56°N) → zone 32."""
        crs = self.proc.get_utm_crs(3.0, 56.0)
        assert crs.to_epsg() == 32632

    def test_norway_just_outside_exception_uses_standard(self):
        """Just above 64°N the standard formula applies again."""
        # 64°N is still band W — exception is only 56°N to <64°N
        crs = self.proc.get_utm_crs(5.3, 64.0)
        # Standard formula: zone = int((5.3+180)/6)+1 = 31
        assert crs.to_epsg() == 32631

    def test_standard_zone_outside_norway_band(self):
        """East Germany (13°E, 52°N) is not in the Norway exception band."""
        crs = self.proc.get_utm_crs(13.0, 52.0)
        # Standard zone 33N
        assert crs.to_epsg() == 32633

    # --- Svalbard zone X exceptions ---

    def test_svalbard_longyearbyen_uses_zone_33(self):
        """Longyearbyen (15.5°E, 78.2°N): zone 32 unused → zone 33."""
        crs = self.proc.get_utm_crs(15.5, 78.2)
        assert crs.to_epsg() == 32633

    def test_svalbard_west_uses_zone_31(self):
        """Western Svalbard (5°E, 78°N): longitude < 9°E → zone 31."""
        crs = self.proc.get_utm_crs(5.0, 78.0)
        assert crs.to_epsg() == 32631

    def test_svalbard_east_uses_zone_35(self):
        """Eastern Svalbard (25°E, 80°N): 21°E–33°E → zone 35."""
        crs = self.proc.get_utm_crs(25.0, 80.0)
        assert crs.to_epsg() == 32635

    # --- Polar region errors ---

    def test_polar_north_raises(self):
        with pytest.raises(ValueError, match="84"):
            self.proc.get_utm_crs(0.0, 85.0)

    def test_polar_north_exact_boundary_raises(self):
        with pytest.raises(ValueError, match="84"):
            self.proc.get_utm_crs(0.0, 84.1)

    def test_polar_south_raises(self):
        with pytest.raises(ValueError, match="80"):
            self.proc.get_utm_crs(0.0, -81.0)

    def test_polar_south_exact_boundary_raises(self):
        with pytest.raises(ValueError, match="80"):
            self.proc.get_utm_crs(0.0, -80.1)

    def test_just_below_north_polar_limit_ok(self):
        """lat = 84.0 is the last valid latitude for UTM."""
        crs = self.proc.get_utm_crs(0.0, 84.0)
        assert crs is not None

    def test_just_above_south_polar_limit_ok(self):
        """lat = -80.0 is the last valid latitude for UTM in the south."""
        crs = self.proc.get_utm_crs(0.0, -80.0)
        assert crs is not None


# ---------------------------------------------------------------------------
# TerrainConfig lat/lon validation
# ---------------------------------------------------------------------------

class TestTerrainConfigValidation:
    """Ensure bad lat/lon values are caught early with a clear message."""

    def test_latitude_too_high_raises(self):
        with pytest.raises(ValueError, match="[Ll]atitude"):
            TerrainConfig(center_lat=91.0, center_lon=0.0,
                          crop_size_km=10, rotation_deg=0)

    def test_latitude_too_low_raises(self):
        with pytest.raises(ValueError, match="[Ll]atitude"):
            TerrainConfig(center_lat=-91.0, center_lon=0.0,
                          crop_size_km=10, rotation_deg=0)

    def test_longitude_too_high_raises(self):
        with pytest.raises(ValueError, match="[Ll]ongitude"):
            TerrainConfig(center_lat=0.0, center_lon=181.0,
                          crop_size_km=10, rotation_deg=0)

    def test_longitude_too_low_raises(self):
        with pytest.raises(ValueError, match="[Ll]ongitude"):
            TerrainConfig(center_lat=0.0, center_lon=-181.0,
                          crop_size_km=10, rotation_deg=0)

    @pytest.mark.parametrize("lat,lon", [
        (39.71, -7.73),     # Portugal
        (-33.9, 151.2),     # Sydney
        (35.7, 139.7),      # Tokyo
        (-34.6, -58.4),     # Buenos Aires
        (60.4, 5.3),        # Bergen, Norway
        (78.2, 15.5),       # Svalbard
        (0.0, 0.0),         # Null Island
        (0.0, 180.0),       # International date line
        (-80.0, 0.0),       # UTM southern limit
        (84.0, 0.0),        # UTM northern limit
    ])
    def test_valid_global_coordinates_accepted(self, lat, lon):
        cfg = TerrainConfig(center_lat=lat, center_lon=lon,
                            crop_size_km=10, rotation_deg=0)
        assert cfg.center_lat == lat
        assert cfg.center_lon == lon


# ---------------------------------------------------------------------------
# GeoTIFF auto-reprojection (WGS84 → UTM)
# ---------------------------------------------------------------------------

class TestGeoTiffReprojection:
    """Verify that GeoTIFFs in a geographic CRS are reprojected transparently."""

    def setup_method(self):
        self.proc = TerrainProcessor()
        # Portugal site: UTM zone 29N
        self.lon_c, self.lat_c = -7.73, 39.71
        self.proc.utm_crs = self.proc.get_utm_crs(self.lon_c, self.lat_c)
        utm_crs = self.proc.utm_crs
        # Compute center in UTM
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        self.cx, self.cy = tr.transform(self.lon_c, self.lat_c)

    def test_wgs84_geotiff_is_reprojected_and_cropped(self):
        """A WGS84 GeoTIFF should produce a valid (non-empty) crop."""
        # Build a synthetic WGS84 GeoTIFF centred roughly on the Portugal site
        mf = _make_wgs84_tif(-9.0, 38.5, -6.5, 41.0)
        with mf.open() as src:
            assert not src.crs.is_projected, "Precondition: input must be geographic"

        # Write to a temp file so rasterio can open it by path
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tf:
            tmp_path = tf.name
        try:
            with mf.open() as src:
                data = src.read(1)
                profile = src.profile
            with rasterio.open(tmp_path, 'w', **profile) as dst:
                dst.write(data, 1)

            result = self.proc.crop_and_rotate_raster(
                raster_path=tmp_path,
                center_utm=(self.cx, self.cy),
                crop_size_km=15,
                rotation_deg=0,
            )
            cropped, transform, crs, res, mask = result
            assert np.any(mask), "Expected at least some valid pixels after cropping"
            assert np.any(np.isfinite(cropped[mask])), "Cropped values should be finite"
            assert crs.is_projected, "Result CRS should be projected (UTM)"
        finally:
            os.unlink(tmp_path)

    def test_utm_geotiff_passes_through_unchanged(self):
        """A GeoTIFF already in UTM should not be reprojected."""
        # Build a UTM tile: 50x50 km centred on the Portugal site
        half = 25_000
        mf = _make_utm_tif(
            self.cx - half, self.cy - half,
            self.cx + half, self.cy + half,
            epsg=self.proc.utm_crs.to_epsg(),
        )
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tf:
            tmp_path = tf.name
        try:
            with mf.open() as src:
                data = src.read(1)
                profile = src.profile
            with rasterio.open(tmp_path, 'w', **profile) as dst:
                dst.write(data, 1)

            result = self.proc.crop_and_rotate_raster(
                raster_path=tmp_path,
                center_utm=(self.cx, self.cy),
                crop_size_km=15,
                rotation_deg=0,
            )
            cropped, transform, crs, res, mask = result
            assert np.any(mask), "Expected valid pixels"
            assert crs.is_projected
        finally:
            os.unlink(tmp_path)

    def test_no_utm_crs_set_raises_clear_error(self):
        """If utm_crs is not set, a geographic GeoTIFF should raise ValueError."""
        proc = TerrainProcessor()
        assert proc.utm_crs is None  # not set yet

        mf = _make_wgs84_tif(-9.0, 38.5, -6.5, 41.0)
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tf:
            tmp_path = tf.name
        try:
            with mf.open() as src:
                data = src.read(1)
                profile = src.profile
            with rasterio.open(tmp_path, 'w', **profile) as dst:
                dst.write(data, 1)

            with pytest.raises(ValueError, match="[Gg]eographic|UTM"):
                proc.crop_and_rotate_raster(
                    raster_path=tmp_path,
                    center_utm=(self.cx, self.cy),
                    crop_size_km=5,
                    rotation_deg=0,
                )
        finally:
            os.unlink(tmp_path)
