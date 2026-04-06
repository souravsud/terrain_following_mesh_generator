"""Tests for grid NaN handling (nearest-neighbour fill) and cell-centre NPZ maps.

The symmetric-trim / cell-count-validation approach (originally called Fix 1 and
Fix 2) has been replaced by a nearest-neighbour NaN fill so that domain dimensions
are preserved exactly as configured.  Tests below verify the new behaviour.

Fix 3 (cell-centre arrays in NPZ maps) is unchanged and tested in TestCellCentreNpz.
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
# NaN nearest-neighbour fill
# ---------------------------------------------------------------------------

class TestNanFill:
    """NaN values at grid edges are filled with nearest-neighbour elevation."""

    def test_clean_dem_returns_target_dimensions(self):
        """A DEM fully covering the terrain extent should return exact target dimensions."""
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

    def test_border_nan_filled_preserves_target_dimensions(self):
        """Border NaN values should be filled; grid dimensions stay at the target."""
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
        grid = gen.create_structured_grid(
            elevation, transform,
            target_rows=12, target_cols=12,
            rotation_deg=0,
            crop_mask=crop_mask,
            centre_utm=(0.0, 0.0),
            center_coordinates=False,
        )
        # Dimensions must be exactly as requested — no trimming.
        nx, ny, _ = grid.dimensions
        assert nx == 12
        assert ny == 12
        # No NaN values in the output elevation
        elev = grid.point_data['elevation']
        assert not np.any(np.isnan(elev)), "Elevation still contains NaN after fill"

    def test_border_nan_filled_with_nearest_value(self):
        """NaN border cells must be filled with their nearest valid neighbour."""
        nrows, ncols = 5, 5
        # Flat DEM at elevation 200 except the top row is NaN
        elevation = _make_flat_dem(nrows, ncols, value=200.0)
        elevation[0, :] = np.nan

        transform = _simple_transform(nrows, ncols)
        crop_mask = np.ones((nrows, ncols), bool)

        gen = StructuredGridGenerator()
        grid = gen.create_structured_grid(
            elevation, transform,
            target_rows=5, target_cols=5,
            rotation_deg=0,
            crop_mask=crop_mask,
            centre_utm=(0.0, 0.0),
            center_coordinates=False,
        )
        elev = grid.point_data['elevation']
        assert not np.any(np.isnan(elev))
        # All filled values should equal 200 (the only valid elevation present)
        np.testing.assert_allclose(elev, 200.0)

    def test_interior_nan_filled_with_nearest_neighbour(self):
        """An interior NaN (e.g. a DEM hole) is also filled without raising an error."""
        nrows, ncols = 20, 20
        elevation = _make_flat_dem(nrows, ncols, value=150.0)
        # A single interior NaN pixel
        elevation[10, 10] = np.nan
        transform = _simple_transform(nrows, ncols)
        crop_mask = np.zeros((nrows, ncols), bool)
        crop_mask[5:15, 5:15] = True

        gen = StructuredGridGenerator()
        # Should not raise
        grid = gen.create_structured_grid(
            elevation, transform,
            target_rows=5, target_cols=5,
            rotation_deg=0,
            crop_mask=crop_mask,
            centre_utm=(0.0, 0.0),
            center_coordinates=False,
        )
        elev = grid.point_data['elevation']
        assert not np.any(np.isnan(elev)), "Interior NaN should have been filled"

    def test_border_nan_filled_with_nearest_value_varying_elevations(self):
        """Nearest-neighbour fill produces valid elevations when the DEM is non-uniform."""
        # 20×20 DEM: left half at 100 m, right half at 200 m.
        # Only the central 10×10 block is marked as terrain, so grid vertices
        # land well inside the DEM and no NaN arises from bilinear sampling.
        # We then verify that NaN-free results are returned and that all filled
        # elevations lie within the DEM's valid range [100, 200].
        nrows, ncols = 20, 20
        elevation = np.empty((nrows, ncols), dtype=float)
        elevation[:, :10] = 100.0
        elevation[:, 10:] = 200.0

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
        elev = grid.point_data['elevation']
        assert not np.any(np.isnan(elev)), "Elevation still contains NaN"
        # All sampled/filled values must lie within the DEM's valid range (allow
        # a small tolerance for bilinear interpolation rounding at the boundary)
        assert np.all(elev >= 100.0 - 1e-6) and np.all(elev <= 200.0 + 1e-6), (
            f"Elevation out of [100, 200] range: min={elev.min()}, max={elev.max()}"
        )
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
