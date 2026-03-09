"""Tests for pipeline metadata generation (FAIR data improvements).

Verifies that:
- Absolute filesystem paths are NOT present in the metadata output.
- The ``environment`` section exists and contains the expected keys.
- Python version is recorded correctly.
- ``openfoam_version`` from ToolsConfig is propagated into the metadata.
- ``input_data`` uses filenames only (no directory component).
- ``output_files`` uses filenames / output-relative paths only.
- Old top-level keys (``input_files``, ``output_directory``) are gone.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from terrain_mesh.utils import write_metadata
from terrain_mesh.config import (
    TerrainConfig,
    GridConfig,
    MeshConfig,
    BoundaryConfig,
    VisualizationConfig,
    ToolsConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_mock_vtk_grid():
    """Return a minimal VTK-like mock grid."""
    grid = MagicMock()
    grid.GetNumberOfPoints.return_value = 100
    grid.GetNumberOfCells.return_value = 81
    grid.GetBounds.return_value = (0.0, 1000.0, 0.0, 1000.0, 100.0, 500.0)
    return grid


def _write_and_read(tmp_path, openfoam_version=None, dem_path=None, rmap_path=None,
                    blockmesh_path=None):
    """Call write_metadata with minimal valid arguments and return parsed JSON."""
    dem_path = dem_path or Path("/some/deep/system/path/terrain_data.tif")
    output_dir = tmp_path
    vtk_path = output_dir / "terrain_structured.vtk"
    metadata_path = output_dir / "pipeline_metadata.json"

    if blockmesh_path is None:
        blockmesh_path = output_dir / "system" / "blockMeshDict"

    elevation = np.random.rand(10, 10) * 100 + 200

    write_metadata(
        dem_path=dem_path,
        rmap_path=rmap_path,
        terrain_config=TerrainConfig(center_lat=39.7, center_lon=-7.7,
                                     crop_size_km=20, rotation_deg=0),
        grid_config=GridConfig(nx=10, ny=10),
        mesh_config=MeshConfig(),
        boundary_config=BoundaryConfig(),
        visualization_config=VisualizationConfig(),
        tools_config=ToolsConfig(openfoam_version=openfoam_version),
        elevation_data=elevation,
        treated_elevation=elevation,
        transform=(0, 1, 0, 0, 0, 1),
        crs="EPSG:32629",
        min_elevation=float(elevation.min()),
        pixel_res=30.0,
        grid=_create_mock_vtk_grid(),
        vtk_path=vtk_path,
        blockmesh_path=blockmesh_path,
        output_dir=output_dir,
        metadata_path=metadata_path,
    )

    with open(metadata_path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoAbsolutePaths:
    """Absolute local filesystem paths must not appear in the metadata."""

    def test_no_absolute_path_in_input_data(self, tmp_path):
        data = _write_and_read(tmp_path,
                               dem_path=Path("/some/deep/system/path/terrain_data.tif"))
        dem_val = data["input_data"]["dem_filename"]
        assert dem_val == "terrain_data.tif", (
            f"Expected only the filename, got: {dem_val!r}"
        )
        assert "/" not in dem_val, "dem_filename should not contain path separators"

    def test_no_absolute_path_in_vtk(self, tmp_path):
        data = _write_and_read(tmp_path)
        vtk_val = data["output_files"]["vtk_mesh"]
        assert "/" not in vtk_val, (
            f"vtk_mesh should not contain '/', got: {vtk_val!r}"
        )

    def test_no_absolute_path_in_metadata_file(self, tmp_path):
        data = _write_and_read(tmp_path)
        meta_val = data["output_files"]["metadata_file"]
        assert "/" not in meta_val, (
            f"metadata_file should not contain '/', got: {meta_val!r}"
        )

    def test_blockmesh_is_relative_path(self, tmp_path):
        data = _write_and_read(tmp_path)
        bm_val = data["output_files"]["blockmesh_dict"]
        assert bm_val is not None
        # Should be a relative path like "system/blockMeshDict", not absolute
        assert not Path(bm_val).is_absolute(), (
            f"blockmesh_dict should be relative, got: {bm_val!r}"
        )

    def test_old_input_files_key_absent(self, tmp_path):
        data = _write_and_read(tmp_path)
        assert "input_files" not in data, (
            "Old 'input_files' key (with full paths) must not be present"
        )

    def test_old_output_directory_key_absent(self, tmp_path):
        data = _write_and_read(tmp_path)
        assert "output_directory" not in data.get("output_files", {}), (
            "Absolute 'output_directory' key must not be in output_files"
        )


class TestEnvironmentSection:
    """The 'environment' section must exist and contain reproducibility info."""

    def test_environment_section_present(self, tmp_path):
        data = _write_and_read(tmp_path)
        assert "environment" in data, "Metadata must contain an 'environment' section"

    def test_python_version_present_and_correct(self, tmp_path):
        data = _write_and_read(tmp_path)
        recorded = data["environment"]["python_version"]
        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert recorded == expected, (
            f"python_version mismatch: recorded={recorded!r}, expected={expected!r}"
        )

    def test_openfoam_version_null_by_default(self, tmp_path):
        data = _write_and_read(tmp_path, openfoam_version=None)
        assert data["environment"]["openfoam_version"] is None

    def test_openfoam_version_propagated(self, tmp_path):
        data = _write_and_read(tmp_path, openfoam_version="v2312")
        assert data["environment"]["openfoam_version"] == "v2312"

    def test_package_versions_dict_present(self, tmp_path):
        data = _write_and_read(tmp_path)
        pkg_ver = data["environment"]["package_versions"]
        assert isinstance(pkg_ver, dict), "package_versions must be a dict"
        # numpy must be detectable at test time
        assert "numpy" in pkg_ver, "numpy version must be recorded"


class TestInputDataSection:
    """input_data section should record filenames, not full paths."""

    def test_input_data_section_present(self, tmp_path):
        data = _write_and_read(tmp_path)
        assert "input_data" in data

    def test_roughness_null_when_not_provided(self, tmp_path):
        data = _write_and_read(tmp_path, rmap_path=None)
        assert data["input_data"]["roughness_filename"] is None

    def test_roughness_filename_only_when_provided(self, tmp_path):
        rmap = Path("/another/deep/path/roughness_map.tif")
        data = _write_and_read(tmp_path, rmap_path=rmap)
        rval = data["input_data"]["roughness_filename"]
        assert rval == "roughness_map.tif"
        assert "/" not in rval


class TestToolsConfig:
    """ToolsConfig dataclass and load_config integration."""

    def test_tools_config_default_openfoam_none(self):
        cfg = ToolsConfig()
        assert cfg.openfoam_version is None

    def test_tools_config_stores_version(self):
        cfg = ToolsConfig(openfoam_version="v2306")
        assert cfg.openfoam_version == "v2306"

    def test_load_config_includes_tools_config(self, tmp_path):
        from terrain_mesh.config import load_config

        config_text = """
terrain:
  center_lat: 51.5
  center_lon: -0.1
  crop_size_km: 10
  rotation_deg: 270
grid:
  nx: 10
  ny: 10
mesh:
  domain_height: 1000.0
boundary: {}
visualization:
  create_plots: false
tools:
  openfoam_version: "v2312"
"""
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(config_text)
        configs = load_config(str(cfg_file))
        assert "tools_config" in configs
        assert isinstance(configs["tools_config"], ToolsConfig)
        assert configs["tools_config"].openfoam_version == "v2312"

    def test_load_config_tools_section_optional(self, tmp_path):
        """YAML configs without a 'tools' section should still work."""
        from terrain_mesh.config import load_config

        config_text = """
terrain:
  center_lat: 51.5
  center_lon: -0.1
  crop_size_km: 10
  rotation_deg: 270
grid:
  nx: 10
  ny: 10
mesh:
  domain_height: 1000.0
boundary: {}
visualization:
  create_plots: false
"""
        cfg_file = tmp_path / "test_config_no_tools.yaml"
        cfg_file.write_text(config_text)
        configs = load_config(str(cfg_file))
        assert "tools_config" in configs
        assert configs["tools_config"].openfoam_version is None
