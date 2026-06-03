"""Configuration classes for terrain mesh generation.

This module provides dataclass-based configuration for:
- Terrain extraction and processing
- Grid generation with multi-block grading
- OpenFOAM mesh output
- Boundary treatment and smoothing
- Visualization options
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Union
import yaml

# Constants
DEFAULT_GAUSSIAN_SMOOTHING_SIGMA = 2.0  # Default sigma for Gaussian terrain smoothing
MIN_GRID_DIMENSION = 2
GRADING_TOLERANCE = 1e-6  # Tolerance for floating-point comparison in grading validation
DEFAULT_DOMAIN_HEIGHT = 4000.0
DEFAULT_Z_CELLS = 10


def _validate_grading(grading: List[Tuple[float, float, float]], name: str) -> None:
    """Validate that grading fractions sum to 1.0.

    Args:
        grading: List of (length_fraction, cell_fraction, expansion_ratio) tuples
        name: Name of the grading parameter for error messages

    Raises:
        ValueError: If fractions don't sum to 1.0 within tolerance
    """
    length_sum = sum(spec[0] for spec in grading)
    cell_sum = sum(spec[1] for spec in grading)

    if abs(length_sum - 1.0) > GRADING_TOLERANCE:
        raise ValueError(f"{name} length fractions must sum to 1.0, got {length_sum}")
    if abs(cell_sum - 1.0) > GRADING_TOLERANCE:
        raise ValueError(f"{name} cell fractions must sum to 1.0, got {cell_sum}")


DEFAULT_AOI_FRACTION = 0.4
DEFAULT_FLAT_BOUNDARY_THICKNESS = 0.1
DEFAULT_PLOT_DPI = 150


@dataclass
class TerrainConfig:
    """Configuration for terrain extraction and processing.
    
    Attributes:
        crop_size_km: Size of terrain region to extract (kilometers)
        rotation_deg: Rotation angle clockwise from North (degrees, 0-360)
        center_lat: Center latitude of terrain region (decimal degrees).
                    If omitted, the center is automatically derived from the
                    geographic extent of the DEM file (GeoTIFF only).
        center_lon: Center longitude of terrain region (decimal degrees).
                    If omitted, the center is automatically derived from the
                    geographic extent of the DEM file (GeoTIFF only).
        smoothing_sigma: Gaussian smoothing sigma for DEM (0 = no smoothing)
        roughness_smoothing_sigma: Gaussian smoothing sigma for roughness map (0 = no smoothing)
        center_coordinates: If True, the coordinate system is transformed such that the center is (0,0)
        
    Raises:
        ValueError: If crop_size_km is not positive
        ValueError: If center_lat is outside [-90, 90] (when provided)
        ValueError: If center_lon is outside [-180, 180] (when provided)
    """

    crop_size_km: float
    rotation_deg: float
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    smoothing_sigma: float = DEFAULT_GAUSSIAN_SMOOTHING_SIGMA
    roughness_smoothing_sigma: float = 0
    center_coordinates: bool = False

    def __post_init__(self):
        if self.crop_size_km <= 0:
            raise ValueError(f"Crop size must be positive, got {self.crop_size_km}")
        if self.center_lat is not None and not (-90.0 <= self.center_lat <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90, got {self.center_lat}")
        if self.center_lon is not None and not (-180.0 <= self.center_lon <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180, got {self.center_lon}")


@dataclass
class GridConfig:
    """Configuration for structured grid generation.
    
    Attributes:
        nx: Number of grid vertices (sampling points) in the x-direction (minimum 2).
            The mesh will have ``nx - 1`` cells per horizontal row.  To obtain a
            target of *C* cells in the x-direction, set ``nx = C + 1``.
        ny: Number of grid vertices (sampling points) in the y-direction (minimum 2).
            The mesh will have ``ny - 1`` cells per horizontal column.
        x_grading: Optional multi-block grading for x-direction.
                  Format: [(length_fraction, cell_fraction, expansion_ratio), ...]
                  All fractions must sum to 1.0
        y_grading: Optional multi-block grading for y-direction.
                  Format: [(length_fraction, cell_fraction, expansion_ratio), ...]
                  All fractions must sum to 1.0
                  
    Raises:
        ValueError: If grid dimensions are less than 2x2
        ValueError: If grading fractions don't sum to 1.0
    """

    nx: int
    ny: int
    x_grading: Optional[List[Tuple[float, float, float]]] = None
    y_grading: Optional[List[Tuple[float, float, float]]] = None

    def __post_init__(self):
        if self.nx < MIN_GRID_DIMENSION or self.ny < MIN_GRID_DIMENSION:
            raise ValueError(
                f"Grid dimensions must be at least {MIN_GRID_DIMENSION}x{MIN_GRID_DIMENSION}, "
                f"got {self.nx}x{self.ny}"
            )

        if self.x_grading:
            _validate_grading(self.x_grading, "x_grading")
        if self.y_grading:
            _validate_grading(self.y_grading, "y_grading")


@dataclass
class MeshConfig:
    """Configuration for OpenFOAM mesh generation and vertical extrusion.
    
    Attributes:
        domain_height: Height of computational domain in meters
        z_grading: Optional vertical grading specification.
                  Format: [(length_fraction, cell_fraction, expansion_ratio), ...]
        total_z_cells: Number of cells in vertical direction
        terrain_normal_first_layer: If True, first layer follows terrain normal
        adjust_ceiling_for_terrain: If True, raises the domain ceiling by the minimum terrain
                    elevation so the effective air-column height above terrain remains constant
                    regardless of the overall ASL altitude of the site (AGL behaviour)
        patch_types: Dictionary mapping boundary names to OpenFOAM patch types.
                    Defaults: ground=wall, sky=patch, inlet=patch, outlet=patch, sides=patch
        extract_inlet_face_info: If True, extract inlet face information for ABL setup
        
    Raises:
        ValueError: If z_grading fractions don't sum to 1.0
    """
    domain_height: float = DEFAULT_DOMAIN_HEIGHT
    
    # Z-direction configuration
    z_grading: Optional[List[Tuple[float, float, float]]] = None
    total_z_cells: Optional[int] = None
    terrain_normal_first_layer: bool = False
    adjust_ceiling_for_terrain: bool = False
    
    patch_types: Optional[Dict[str, str]] = None
    extract_inlet_face_info: bool = True

    # Default surface roughness length used when no roughness map is supplied.
    # When ``rmap_path`` is not passed to the pipeline, a constant roughness map
    # equal to this value is synthesised and saved alongside the terrain map so
    # that ML training datasets always contain a paired roughness map.
    default_z0: float = 0.1

    def __post_init__(self):
        if self.patch_types is None:
            self.patch_types = {
                "ground": "wall",
                "sky": "patch",
                "inlet": "patch",
                "outlet": "patch",
                "sides": "patch",
            }
        
        if self.z_grading:
            _validate_grading(self.z_grading, "z_grading")


@dataclass
class VisualizationConfig:
    """Configuration for mesh visualization and plotting.
    
    Attributes:
        create_plots: If True, generate visualization plots
        show_grid_lines: If True, display grid lines on plots
        save_high_res: If True, save high-resolution versions of plots
        plot_format: Output format for plots ('png', 'pdf', 'svg')
        dpi: Resolution in dots per inch for raster formats
    """

    create_plots: bool = True
    show_grid_lines: bool = True
    save_high_res: bool = True
    plot_format: str = "png"
    dpi: int = DEFAULT_PLOT_DPI


@dataclass
class BoundaryConfig:
    """Configuration for 3-zone smooth-step boundary treatment.

    Outside the AOI, terrain is divided into two zones per enabled face:

    - Smooth-step transition: the terrain is blended from 100 % real at the
      AOI edge toward a computed target elevation at the flat-zone boundary,
      using the C¹-continuous kernel w = 3t²−2t³.  No additional DEM
      smoothing is applied.
    - Flat zone: a thin strip at each enabled boundary face set to a constant
      target elevation so ABL inlet/outlet profiles can be applied on flat
      ground.

    The target elevation is always computed by fitting a robust (Theil-Sen)
    line through the 1-D median profile of the transition zone and
    extrapolating to the flat-zone boundary.  The ``clamp_target`` flag
    controls whether this extrapolated value is capped at the median of
    the actual flat-zone terrain — preventing the flat zone from being set
    higher than the real terrain on upward-sloping domains.

    Attributes:
        aoi_fraction: Fraction of domain size for the central AOI (0–1).
        boundary_mode: 'uniform' (radial, all sides) or 'directional'
            (selected faces only, flow-aligned).
        flat_boundary_thickness_fraction: Fraction of domain width used as
            the flat zone on each enabled face.  The inner half is sampled
            for the clamp; the outer half is set to the target elevation.
        enabled_boundaries: Faces to treat, e.g. ['east', 'west'].
        clamp_target: If True, cap the extrapolated target at the median of
            the real flat-zone terrain, ensuring the flat zone never sits
            above the actual terrain on that face.  Recommended for diverse
            global datasets.  Default True.

    Raises:
        ValueError: If aoi_fraction or flat_boundary_thickness_fraction is
            not strictly between 0 and 1.
    """

    aoi_fraction: float = DEFAULT_AOI_FRACTION
    boundary_mode: str = "uniform"
    flat_boundary_thickness_fraction: float = DEFAULT_FLAT_BOUNDARY_THICKNESS
    enabled_boundaries: List[str] = None
    clamp_target: bool = True

    def __post_init__(self):
        if self.enabled_boundaries is None:
            self.enabled_boundaries = (
                ["east", "west"] if self.boundary_mode == "directional" else ["uniform"]
            )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file and return instantiated config objects.
    
    This function reads a YAML configuration file and creates typed configuration
    objects for all components of the terrain mesh generation pipeline.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with config object instances ready for pipeline.run(**configs).
        Keys: terrain_config, grid_config, mesh_config, boundary_config, visualization_config
        
    Raises:
        FileNotFoundError: If config_path doesn't exist
        yaml.YAMLError: If YAML file is malformed
        ValueError: If configuration values are invalid
        
    Example:
        >>> configs = load_config("terrain_config.yaml")
        >>> pipeline = TerrainMeshPipeline()
        >>> results = pipeline.run(dem_path="terrain.tif", output_dir="output", **configs)
    """
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file) or {}

    def _to_tuples(data: dict, *keys) -> None:
        """Convert list-of-lists grading specs to list-of-tuples in-place."""
        for key in keys:
            if key in data:
                data[key] = [tuple(spec) for spec in data[key]]

    # Terrain configuration
    configs = {}
    configs["terrain_config"] = TerrainConfig(**config_data.get("terrain", {}))

    # Grid configuration — convert grading arrays from lists to tuples
    grid_data = config_data.get("grid", {})
    _to_tuples(grid_data, "x_grading", "y_grading")
    configs["grid_config"] = GridConfig(**grid_data)

    # Mesh configuration — convert grading array from lists to tuples
    mesh_data = config_data.get("mesh", {})
    _to_tuples(mesh_data, "z_grading")
    configs["mesh_config"] = MeshConfig(**mesh_data)

    configs["boundary_config"] = BoundaryConfig(**config_data.get("boundary", {}))
    configs["visualization_config"] = VisualizationConfig(**config_data.get("visualization", {}))

    return configs
