"""Tests for grid dimension fixes (Fix 1, Fix 2, Fix 3).

Fix 1 (critical):  Raise ValueError when the valid cell count falls below the
    configured target after NaN-edge trimming, so that users know to increase
    crop_size_km.

Fix 2 (important):  Symmetrically trim NaN border vertices from the sampled
    grid so the output shape is deterministic regardless of which side has the
    larger rounding error at the rotated-crop boundary.

Fix 3 (nice-to-have):  Save cell-centre arrays (``elevation_cc``, ``x_cc``,
    ``y_cc``, and ``z0_cc`` for roughness) alongside the vertex arrays in the
    NPZ maps, so post-processing pipelines do not need to interpolate.
"""

import pathlib
import tempfile

import numpy as np
import pytest
import pyvista as pv
from rasterio.transform import Affine

from terrain_mesh.grid_generator import StructuredGridGenerator
from terrain_mesh.pipeline import TerrainMeshPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_dem(nrows=12, ncols=12, value=100.0):
    """Return a flat elevation array with no NaN."""
    return np.full((nrows, ncols), value, dtype=float)


def _simple_transform(nrows, ncols, pixel_size=10.0):
    """North-up affine transform with origin at (0, nrows*pixel_size)."""
    return Affine(pixel_size, 0, 0, 0, -pixel_size, nrows * pixel_size)


def _make_pyvista_grid(ny=5, nx=5, elevation=50.0):
    """Create a minimal PyVista StructuredGrid with uniform elevation."""
    X, Y = np.meshgrid(np.linspace(0.0, 100.0, nx), np.linspace(0.0, 100.0, ny))
    Z = np.full((ny, nx), elevation, dtype=float)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    grid = pv.StructuredGrid()
    grid.points = pts
    grid.dimensions = (nx, ny, 1)
    return grid


# ---------------------------------------------------------------------------
# Fix 2 + Fix 1: NaN trimming and cell-count validation
# ---------------------------------------------------------------------------

class TestNanTrimming:
    """Fix 2: Symmetrically trim NaN border vertices."""

    def test_clean_dem_returns_target_dimensions(self):
        """A DEM fully covering the terrain extent should return exact target dimensions."""
        # Use a 20×20 DEM with only the central 10×10 region marked as terrain.
        # Grid vertices land at pixel indices [5, 14], well inside the DEM extent,
        # which avoids the tiny floating-point out-of-bounds artefacts that occur
        # when a grid is created to span exactly the DEM boundary.
        nrows, ncols = 20, 20
        elevation = _make_flat_dem(nrows, ncols)
        transform = _simple_transform(nrows, ncols)
        crop_mask = np.zeros((nrows, ncols), bool)
        crop_mask[5:15, 5:15] = True  # central 10×10 region

        gen = StructuredGridGenerator()
        grid = gen.create_structured_grid(
            elevation, transform,
            target_rows=5, target_cols=5,
            rotation_deg=0,
            crop_mask=crop_mask,
            centre_utm=(0.0, 0.0),
            center_coordinates=False,
        )
        nx, ny, _ = grid.dimensions
        assert nx == 5
        assert ny == 5

    def test_border_nan_raises_fix1_error(self):
        """Border NaN causes trimming → fewer cells than target → ValueError (Fix 1)."""
        nrows, ncols = 12, 12
        elevation = _make_flat_dem(nrows, ncols)
        # Set entire border to NaN (simulates vertices landing just outside the
        # rotated crop boundary).
        elevation[0, :] = np.nan
        elevation[-1, :] = np.nan
        elevation[:, 0] = np.nan
        elevation[:, -1] = np.nan

        transform = _simple_transform(nrows, ncols)
        crop_mask = np.ones((nrows, ncols), bool)

        gen = StructuredGridGenerator()
        # target_cols=12 → expects 11 cells; after trimming we get fewer.
        with pytest.raises(ValueError, match="crop_size_km"):
            gen.create_structured_grid(
                elevation, transform,
                target_rows=12, target_cols=12,
                rotation_deg=0,
                crop_mask=crop_mask,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
            )

    def test_interior_nan_raises_dem_gap_error(self):
        """Interior NaN after edge trimming signals a DEM coverage gap (Fix 2)."""
        # Use a 20×20 DEM with a central crop_mask so grid vertices sit inside the
        # DEM extent without FP boundary artefacts.  One NaN pixel inside the sampled
        # region produces an interior NaN cell that cannot be removed by edge trimming.
        nrows, ncols = 20, 20
        elevation = _make_flat_dem(nrows, ncols)
        # Pixel (10, 10) is inside the crop region; bilinear interpolation at the
        # target vertex near (row≈9.5, col≈9.5) in pixel space will sample it.
        elevation[10, 10] = np.nan
        transform = _simple_transform(nrows, ncols)
        crop_mask = np.zeros((nrows, ncols), bool)
        crop_mask[5:15, 5:15] = True  # central 10×10 region

        gen = StructuredGridGenerator()
        with pytest.raises(ValueError, match="interior"):
            gen.create_structured_grid(
                elevation, transform,
                target_rows=5, target_cols=5,
                rotation_deg=0,
                crop_mask=crop_mask,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
            )

    def test_fully_nan_grid_raises_clear_error(self):
        """An entirely NaN elevation array should raise a descriptive ValueError."""
        nrows, ncols = 8, 8
        elevation = np.full((nrows, ncols), np.nan)
        transform = _simple_transform(nrows, ncols)
        crop_mask = np.ones((nrows, ncols), bool)

        gen = StructuredGridGenerator()
        with pytest.raises(ValueError):
            gen.create_structured_grid(
                elevation, transform,
                target_rows=5, target_cols=5,
                rotation_deg=0,
                crop_mask=crop_mask,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
            )

    def test_error_message_mentions_crop_size(self):
        """The Fix 1 ValueError must guide the user toward increasing crop_size_km."""
        nrows, ncols = 10, 10
        elevation = _make_flat_dem(nrows, ncols)
        elevation[0, :] = np.nan
        elevation[-1, :] = np.nan

        transform = _simple_transform(nrows, ncols)
        crop_mask = np.ones((nrows, ncols), bool)

        gen = StructuredGridGenerator()
        with pytest.raises(ValueError, match="crop_size_km"):
            gen.create_structured_grid(
                elevation, transform,
                target_rows=10, target_cols=5,
                rotation_deg=0,
                crop_mask=crop_mask,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
            )


# ---------------------------------------------------------------------------
# Fix 3: Cell-centre arrays in NPZ maps
# ---------------------------------------------------------------------------

class TestCellCentreNpz:
    """Fix 3: Cell-centre arrays (*_cc) saved alongside vertex arrays in NPZ."""

    def test_terrain_map_contains_cc_keys(self):
        """terrain_map.npz must include elevation_cc, x_cc, y_cc keys."""
        grid = _make_pyvista_grid(ny=5, nx=5)
        pipeline = TerrainMeshPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            terrain_path, _ = pipeline._save_maps(
                grid=grid,
                roughness_data=None,
                roughness_transform=None,
                centre_utm=(50.0, 50.0),
                center_coordinates=False,
                output_dir=pathlib.Path(tmpdir),
            )
            data = np.load(terrain_path)

        # Vertex arrays must still be present (needed by blockMeshDict generator)
        assert 'elevation' in data
        assert 'x' in data
        assert 'y' in data

        # Cell-centre arrays must also be present
        assert 'elevation_cc' in data, "elevation_cc missing from terrain_map.npz"
        assert 'x_cc' in data, "x_cc missing from terrain_map.npz"
        assert 'y_cc' in data, "y_cc missing from terrain_map.npz"

    def test_cc_arrays_have_correct_shape(self):
        """Cell-centre arrays must have shape (ny-1, nx-1)."""
        ny, nx = 6, 8
        grid = _make_pyvista_grid(ny=ny, nx=nx)
        pipeline = TerrainMeshPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            terrain_path, _ = pipeline._save_maps(
                grid=grid,
                roughness_data=None,
                roughness_transform=None,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
                output_dir=pathlib.Path(tmpdir),
            )
            data = np.load(terrain_path)

        assert data['elevation'].shape == (ny, nx), "vertex elevation shape wrong"
        assert data['elevation_cc'].shape == (ny - 1, nx - 1), "cc elevation shape wrong"
        assert data['x_cc'].shape == (ny - 1, nx - 1), "cc x shape wrong"
        assert data['y_cc'].shape == (ny - 1, nx - 1), "cc y shape wrong"

    def test_cc_elevation_is_4vertex_average(self):
        """elevation_cc must equal the average of the four surrounding vertices."""
        ny, nx = 3, 3
        # Build a 3×3 grid with known Z values
        Z_vertex = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])
        X = np.zeros((ny, nx))
        Y = np.zeros((ny, nx))
        pts = np.column_stack([X.ravel(), Y.ravel(), Z_vertex.ravel()])
        grid = pv.StructuredGrid()
        grid.points = pts
        grid.dimensions = (nx, ny, 1)

        pipeline = TerrainMeshPipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            terrain_path, _ = pipeline._save_maps(
                grid=grid,
                roughness_data=None,
                roughness_transform=None,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
                output_dir=pathlib.Path(tmpdir),
            )
            data = np.load(terrain_path)

        # Expected 2×2 cell-centre averages:
        # cell (0,0): avg(1,2,4,5)=3.0   cell (0,1): avg(2,3,5,6)=4.0
        # cell (1,0): avg(4,5,7,8)=6.0   cell (1,1): avg(5,6,8,9)=7.0
        expected = np.array([[3.0, 4.0],
                              [6.0, 7.0]])
        np.testing.assert_array_almost_equal(data['elevation_cc'], expected)

    def test_roughness_map_contains_cc_keys(self):
        """roughness_map.npz must include z0_cc, x_cc, y_cc keys."""
        from rasterio.transform import from_bounds

        ny, nx = 4, 4
        grid = _make_pyvista_grid(ny=ny, nx=nx)

        # Minimal roughness raster covering the same domain
        roughness_data = np.full((10, 10), 0.1)
        roughness_transform = from_bounds(0, 0, 100, 100, 10, 10)

        pipeline = TerrainMeshPipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            _, roughness_path = pipeline._save_maps(
                grid=grid,
                roughness_data=roughness_data,
                roughness_transform=roughness_transform,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
                output_dir=pathlib.Path(tmpdir),
            )
            assert roughness_path is not None
            data = np.load(roughness_path)

        assert 'z0' in data
        assert 'z0_cc' in data, "z0_cc missing from roughness_map.npz"
        assert 'x_cc' in data, "x_cc missing from roughness_map.npz"
        assert 'y_cc' in data, "y_cc missing from roughness_map.npz"
        assert data['z0_cc'].shape == (ny - 1, nx - 1)

    def test_vertex_arrays_still_present_for_blockmesh(self):
        """Vertex arrays (elevation, x, y) must remain for blockMeshDict generation."""
        ny, nx = 5, 5
        grid = _make_pyvista_grid(ny=ny, nx=nx)
        pipeline = TerrainMeshPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            terrain_path, _ = pipeline._save_maps(
                grid=grid,
                roughness_data=None,
                roughness_transform=None,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
                output_dir=pathlib.Path(tmpdir),
            )
            data = np.load(terrain_path)

        # Vertex arrays needed by load_terrain_points / blockMeshDict generator
        assert data['elevation'].shape == (ny, nx)
        assert data['x'].shape == (ny, nx)
        assert data['y'].shape == (ny, nx)
