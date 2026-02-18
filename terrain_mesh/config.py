"""Configuration classes for terrain mesh generation.

This module provides dataclass-based configuration for:
- Terrain extraction and processing
- Grid generation with multi-block grading
- OpenFOAM mesh output
- Boundary treatment and smoothing
- Visualization options
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import yaml

# Constants
DEFAULT_GAUSSIAN_SMOOTHING_SIGMA = 2.0  # Default sigma for Gaussian terrain smoothing
MIN_GRID_DIMENSION = 2
GRADING_TOLERANCE = 1e-6  # Tolerance for floating-point comparison in grading validation
DEFAULT_DOMAIN_HEIGHT = 4000.0
DEFAULT_Z_CELLS = 10
DEFAULT_AOI_FRACTION = 0.4
DEFAULT_FLAT_BOUNDARY_THICKNESS = 0.1
DEFAULT_PROGRESSION_RATE = 1.5
DEFAULT_PLOT_DPI = 150


@dataclass
class TerrainConfig:
    """Configuration for terrain extraction and processing.
    
    Attributes:
        center_lat: Center latitude of terrain region (decimal degrees)
        center_lon: Center longitude of terrain region (decimal degrees)
        crop_size_km: Size of terrain region to extract (kilometers)
        rotation_deg: Rotation angle clockwise from North (degrees, 0-360)
        smoothing_sigma: Gaussian smoothing sigma (0 = no smoothing)
        center_coordinates: If True, the coordinate system is transformed such that the centre is (0,0)
        
    Raises:
        ValueError: If crop_size_km is not positive
    """

    center_lat: float
    center_lon: float
    crop_size_km: float
    rotation_deg: float
    smoothing_sigma: float = DEFAULT_GAUSSIAN_SMOOTHING_SIGMA
    center_coordinates: bool = False

    def __post_init__(self):
        if self.crop_size_km <= 0:
            raise ValueError(f"Crop size must be positive, got {self.crop_size_km}")


@dataclass
class GridConfig:
    """Configuration for structured grid generation.
    
    Attributes:
        nx: Number of cells in x-direction (minimum 2)
        ny: Number of cells in y-direction (minimum 2)
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
            self._validate_grading(self.x_grading, "x_grading")
        if self.y_grading:
            self._validate_grading(self.y_grading, "y_grading")

    @staticmethod
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
            raise ValueError(
                f"{name} length fractions must sum to 1.0, got {length_sum}"
            )
        if abs(cell_sum - 1.0) > GRADING_TOLERANCE:
            raise ValueError(
                f"{name} cell fractions must sum to 1.0, got {cell_sum}"
            )


@dataclass
class MeshConfig:
    """Configuration for OpenFOAM mesh generation and vertical extrusion.
    
    Attributes:
        domain_height: Height of computational domain in meters
        z_grading: Optional vertical grading specification.
                  Format: [(length_fraction, cell_fraction, expansion_ratio), ...]
        total_z_cells: Number of cells in vertical direction
        terrain_normal_first_layer: If True, first layer follows terrain normal
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
    
    patch_types: Optional[Dict[str, str]] = None
    extract_inlet_face_info: bool = True

    def __post_init__(self):
        if self.patch_types is None:
            self.patch_types = {
                "ground": "wall",
                "sky": "patch",
                "inlet": "patch",
                "outlet": "patch",
                "sides": "patch",
            }
        
        # Validate z_grading if specified
        if self.z_grading:
            self._validate_z_grading()
        
    
    def _validate_z_grading(self) -> None:
        """Validate z-direction grading specification.
        
        Raises:
            ValueError: If fractions don't sum to 1.0 within tolerance
        """
        length_sum = sum(spec[0] for spec in self.z_grading)
        cell_sum = sum(spec[1] for spec in self.z_grading)

        if abs(length_sum - 1.0) > GRADING_TOLERANCE:
            raise ValueError(
                f"z_grading length fractions must sum to 1.0, got {length_sum}"
            )
        if abs(cell_sum - 1.0) > GRADING_TOLERANCE:
            raise ValueError(
                f"z_grading cell fractions must sum to 1.0, got {cell_sum}"
            )

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
    """Configuration for boundary treatment with progressive smoothing.
    
    This configuration controls the 4-zone boundary treatment:
    - Area of Interest (AOI): Central region with original terrain
    - Transition Zone: Progressive smoothing from AOI to boundary
    - Blend Zone: Smooth blending to target elevation
    - Flat Zone: Constant elevation at boundary
    
    Attributes:
        aoi_fraction: Fraction of domain considered as AOI (0-1)
        boundary_mode: Treatment mode - 'uniform' (all sides) or 'directional' (selected sides)
        flat_boundary_thickness_fraction: Thickness of flat boundary region (0-1)
        enabled_boundaries: List of boundaries to treat: ['east', 'west', 'north', 'south']
        smoothing_method: Smoothing kernel type - 'mean', 'gaussian', or 'median'
        kernel_progression: How smoothing increases - 'exponential' or 'linear'
        base_kernel_size: Initial smoothing kernel size (auto-calculated if None)
        max_kernel_size: Maximum smoothing kernel size (auto-calculated if None)
        progression_rate: Rate of kernel size increase (for exponential progression)
        boundary_flatness_mode: Flattening method - 'heavy_smooth' or 'blend_target'
        uniform_elevation: Override boundary height in meters (auto-calculated if None)
        
    Raises:
        ValueError: If aoi_fraction or flat_boundary_thickness_fraction is not in (0, 1)
        ValueError: If progression_rate is not > 1
    """

    # Zone definition
    aoi_fraction: float = DEFAULT_AOI_FRACTION

    # Treatment mode
    boundary_mode: str = "uniform"  # 'uniform' or 'directional'

    # Boundary sampling parameters
    flat_boundary_thickness_fraction: float = DEFAULT_FLAT_BOUNDARY_THICKNESS
    enabled_boundaries: List[str] = None  # For directional mode ['east', 'west']

    # Progressive smoothing parameters
    smoothing_method: str = "mean"  # 'gaussian', 'mean', 'median'
    kernel_progression: str = "exponential"  # 'exponential', 'linear'
    base_kernel_size: int = None
    max_kernel_size: int = None
    progression_rate: float = DEFAULT_PROGRESSION_RATE  # For exponential progression

    # Boundary flatness treatment
    boundary_flatness_mode: str = "heavy_smooth"  # 'heavy_smooth', 'blend_target'
    uniform_elevation: Optional[float] = None  # Override calculated boundary height

    # Legacy parameters (for backward compatibility if needed)
    flat_fraction: float = 0.05  # Not used in progressive smoothing
    flat_boundaries: List[str] = None  # Not used in progressive smoothing
    transition_smoothing_sigma: float = 3.0  # Not used in progressive smoothing
    transition_iterations: int = 10  # Not used in progressive smoothing

    def __post_init__(self):
        if self.enabled_boundaries is None:
            if self.boundary_mode == "directional":
                self.enabled_boundaries = ["east", "west"]
            else:
                self.enabled_boundaries = ["uniform"]

        # Legacy compatibility
        if self.flat_boundaries is None:
            self.flat_boundaries = ["north", "south", "east", "west"]


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

    # Create config objects with defaults, override with YAML values
    configs = {}

    # Terrain configuration
    terrain_data = config_data.get("terrain", {})
    configs["terrain_config"] = TerrainConfig(**terrain_data)

    # Grid configuration
    grid_data = config_data.get("grid", {})
    # Handle grading arrays - convert lists of lists to lists of tuples
    if "x_grading" in grid_data:
        grid_data["x_grading"] = [tuple(spec) for spec in grid_data["x_grading"]]
    if "y_grading" in grid_data:
        grid_data["y_grading"] = [tuple(spec) for spec in grid_data["y_grading"]]
    configs["grid_config"] = GridConfig(**grid_data)

    # Mesh configuration
    mesh_data = config_data.get("mesh", {})
    # Handle z_grading array - convert lists of lists to lists of tuples
    if "z_grading" in mesh_data:
        mesh_data["z_grading"] = [tuple(spec) for spec in mesh_data["z_grading"]]
    configs["mesh_config"] = MeshConfig(**mesh_data)

    # Boundary configuration
    boundary_data = config_data.get("boundary", {})
    configs["boundary_config"] = BoundaryConfig(**boundary_data)

    # Visualization configuration
    viz_data = config_data.get("visualization", {})
    configs["visualization_config"] = VisualizationConfig(**viz_data)

    return configs
