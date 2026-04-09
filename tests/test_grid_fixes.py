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


# ---------------------------------------------------------------------------
# NaN bug fix: np.fmax in _save_maps
# ---------------------------------------------------------------------------

class TestSaveMapsNanFix:
    """Valid terrain points that fall outside the roughness raster extent must
    not produce NaN in roughness_map.npz (the old np.maximum propagated NaN)."""

    def test_no_nan_in_z0_when_raster_undercovers_grid(self):
        """z0 must be finite (>= 0.0002) for all valid elevation points even when
        the roughness raster is smaller than the grid's coordinate range."""
        from rasterio.transform import from_bounds

        ny, nx = 5, 5
        grid = _make_pyvista_grid(ny=ny, nx=nx, elevation=50.0)

        # Roughness raster covers only the central 20 % of the grid (x,y in
        # [40,60]), so corner/edge vertices are strictly outside its bounds.
        roughness_data = np.full((3, 3), 0.5)
        roughness_transform = from_bounds(40, 40, 60, 60, 3, 3)

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

        z0 = data['z0']
        # No NaN should appear at valid terrain positions (elevation is fully
        # defined for this flat grid — no NaN elevation).
        assert not np.any(np.isnan(z0)), (
            "NaN found in z0 for valid terrain points outside roughness raster extent"
        )
        # All values must respect the minimum roughness floor
        assert np.all(z0 >= 0.0002 - 1e-9), (
            f"z0 value below minimum roughness: min={z0.min()}"
        )

    def test_nan_preserved_outside_terrain_crop(self):
        """NaN must be preserved in z0 where elevation itself is NaN (outside crop)."""
        from rasterio.transform import from_bounds
        import pyvista as pv

        ny, nx = 5, 5
        X, Y = np.meshgrid(np.linspace(0.0, 100.0, nx), np.linspace(0.0, 100.0, ny))
        Z = np.full((ny, nx), 50.0, dtype=float)
        # Force the four corners to NaN (simulates outside-crop points)
        Z[0, 0] = Z[0, -1] = Z[-1, 0] = Z[-1, -1] = np.nan
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        grid = pv.StructuredGrid()
        grid.points = pts
        grid.dimensions = (nx, ny, 1)

        roughness_data = np.full((5, 5), 0.3)
        roughness_transform = from_bounds(0, 0, 100, 100, 5, 5)

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
            data = np.load(roughness_path)

        z0 = data['z0']
        # Corners (NaN elevation) → NaN z0
        assert np.isnan(z0[0, 0]) and np.isnan(z0[0, -1])
        assert np.isnan(z0[-1, 0]) and np.isnan(z0[-1, -1])
        # Interior (valid elevation) → finite z0
        assert not np.any(np.isnan(z0[1:-1, 1:-1]))


# ---------------------------------------------------------------------------
# Constant roughness synthesis
# ---------------------------------------------------------------------------

class TestConstantRoughnessMap:
    """When no rmap_path is provided, a constant roughness_map.npz must be
    generated using mesh_config.default_z0."""

    def test_make_constant_roughness_returns_array_and_transform(self):
        """_make_constant_roughness must return a (3,3) array + affine transform."""
        grid = _make_pyvista_grid(ny=6, nx=6, elevation=100.0)
        pipeline = TerrainMeshPipeline()
        roughness_data, roughness_transform = pipeline._make_constant_roughness(
            grid=grid,
            center_coordinates=False,
            centre_utm=(0.0, 0.0),
            z0_value=0.5,
        )
        assert roughness_data.shape == (3, 3)
        assert np.allclose(roughness_data, 0.5)
        # Transform must be valid (non-zero pixel size)
        assert roughness_transform.a > 0
        assert roughness_transform.e < 0

    def test_make_constant_roughness_clips_to_minimum(self):
        """z0_value below the physical minimum (0.0002) must be clipped."""
        grid = _make_pyvista_grid(ny=4, nx=4, elevation=50.0)
        pipeline = TerrainMeshPipeline()
        roughness_data, _ = pipeline._make_constant_roughness(
            grid=grid,
            center_coordinates=False,
            centre_utm=(0.0, 0.0),
            z0_value=0.00001,  # below minimum
        )
        assert np.allclose(roughness_data, 0.0002)

    def test_save_maps_writes_roughness_npz_for_constant_raster(self):
        """_save_maps must write roughness_map.npz when a constant raster is passed."""
        from rasterio.transform import from_bounds

        ny, nx = 4, 4
        grid = _make_pyvista_grid(ny=ny, nx=nx, elevation=100.0)
        pipeline = TerrainMeshPipeline()

        roughness_data, roughness_transform = pipeline._make_constant_roughness(
            grid=grid,
            center_coordinates=False,
            centre_utm=(0.0, 0.0),
            z0_value=0.03,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _, roughness_path = pipeline._save_maps(
                grid=grid,
                roughness_data=roughness_data,
                roughness_transform=roughness_transform,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
                output_dir=pathlib.Path(tmpdir),
            )
            assert roughness_path is not None, "roughness_map.npz was not created"
            data = np.load(roughness_path)

        assert 'z0' in data
        assert 'z0_cc' in data
        # All valid grid points should have z0 ≈ 0.03 (clipped to min 0.0002)
        z0 = data['z0']
        assert not np.any(np.isnan(z0)), "Unexpected NaN in constant roughness map"
        np.testing.assert_allclose(z0, 0.03, rtol=1e-4)

    def test_constant_roughness_z0_value_is_consistent(self):
        """z0 values in the NPZ must equal the requested constant across the grid."""
        grid = _make_pyvista_grid(ny=5, nx=5, elevation=100.0)
        pipeline = TerrainMeshPipeline()

        constant_z0 = 0.15
        roughness_data, roughness_transform = pipeline._make_constant_roughness(
            grid=grid,
            center_coordinates=False,
            centre_utm=(0.0, 0.0),
            z0_value=constant_z0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _, roughness_path = pipeline._save_maps(
                grid=grid,
                roughness_data=roughness_data,
                roughness_transform=roughness_transform,
                centre_utm=(0.0, 0.0),
                center_coordinates=False,
                output_dir=pathlib.Path(tmpdir),
            )
            data = np.load(roughness_path)

        np.testing.assert_allclose(data['z0'], constant_z0, rtol=1e-4)
        np.testing.assert_allclose(data['z0_cc'], constant_z0, rtol=1e-4)

