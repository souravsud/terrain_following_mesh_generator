"""Configuration classes for terrain mesh generation"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import yaml
from .ogrid_config import OGridConfig
from pathlib import Path


@dataclass
class TerrainConfig:
    """Configuration for terrain processing"""

    center_lat: float
    center_lon: float
    crop_size_km: float
    domain_shape: str = 'rectangular'  #'rectangular' or 'circular'
    rotation_deg: float = 0.0
    smoothing_sigma: float = 2.0
    center_coordinates: bool = False

    def __post_init__(self):
        if self.crop_size_km <= 0:
            raise ValueError("Crop size must be positive")


@dataclass
class GridConfig:
    """Configuration for grid generation"""

    nx: int
    ny: int
    x_grading: Optional[List[Tuple[float, float, float]]] = None
    y_grading: Optional[List[Tuple[float, float, float]]] = None

    def __post_init__(self):
        if self.nx < 2 or self.ny < 2:
            raise ValueError("Grid dimensions must be at least 2x2")

        if self.x_grading:
            self._validate_grading(self.x_grading, "x_grading")
        if self.y_grading:
            self._validate_grading(self.y_grading, "y_grading")

    @staticmethod
    def _validate_grading(grading: List[Tuple[float, float, float]], name: str):
        length_sum = sum(spec[0] for spec in grading)
        cell_sum = sum(spec[1] for spec in grading)

        if abs(length_sum - 1.0) > 1e-6:
            raise ValueError(
                f"{name} length fractions must sum to 1.0, got {length_sum}"
            )
        if abs(cell_sum - 1.0) > 1e-6:
            raise ValueError(f"{name} cell fractions must sum to 1.0, got {cell_sum}")


@dataclass
class MeshConfig:
    """Configuration for OpenFOAM mesh generation"""
    domain_height: float = 4000.0
    
    # Z-direction configuration
    z_grading: Optional[List[Tuple[float, float, float]]] = None
    total_z_cells: Optional[int] = 20
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
        
    
    def _validate_z_grading(self):
        """Validate z-direction grading specification"""
        length_sum = sum(spec[0] for spec in self.z_grading)
        cell_sum = sum(spec[1] for spec in self.z_grading)

        if abs(length_sum - 1.0) > 1e-6:
            raise ValueError(
                f"z_grading length fractions must sum to 1.0, got {length_sum}"
            )
        if abs(cell_sum - 1.0) > 1e-6:
            raise ValueError(f"z_grading cell fractions must sum to 1.0, got {cell_sum}")

@dataclass
class VisualizationConfig:
    """Configuration for visualization options"""

    create_plots: bool = True
    show_grid_lines: bool = True
    save_high_res: bool = True
    plot_format: str = "png"
    dpi: int = 150


@dataclass
class BoundaryConfig:
    """Configuration for boundary treatment with progressive smoothing"""

    # Zone definition
    aoi_fraction: float = 0.4

    # Treatment mode
    boundary_mode: str = "directional"  # 'uniform' or 'directional'

    # Boundary sampling parameters
    flat_boundary_thickness_fraction: float = 0.1
    enabled_boundaries: List[str] = None  # For directional mode ['east', 'west']

    # Progressive smoothing parameters
    smoothing_method: str = "mean"  # 'gaussian', 'mean', 'median'
    kernel_progression: str = "exponential"  # 'exponential', 'linear'
    base_kernel_size: int = None
    max_kernel_size: int = None
    progression_rate: float = 1.5  # For exponential progression

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


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    Automatically detects domain shape and loads appropriate configs.
    
    Returns:
        dict with config objects (terrain_config, mesh_config, boundary_config, 
        visualization_config, and either grid_config OR ogrid_config)
    """
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create terrain config
    terrain_config = TerrainConfig(**config_dict['terrain'])
    
    # Detect domain shape and create appropriate grid config
    domain_shape = config_dict['terrain'].get('domain_shape', 'rectangular')
    
    if domain_shape == 'circular':
        # Load O-grid config
        from .ogrid_config import OGridConfig
        ogrid_config = OGridConfig(**config_dict['ogrid'])
        grid_config = None
        print(f"Loaded circular domain configuration with {ogrid_config.n_sectors} sectors")
    else:
        # Load rectangular grid config
        grid_config = GridConfig(**config_dict['grid'])
        ogrid_config = None
        print(f"Loaded rectangular domain configuration ({grid_config.nx}x{grid_config.ny})")
    
    # Load other configs (same for both domain types)
    mesh_config = MeshConfig(**config_dict['mesh'])
    boundary_config = BoundaryConfig(**config_dict['boundary'])
    visualization_config = VisualizationConfig(**config_dict.get('visualization', {}))
    
    # Return all configs
    return {
        'terrain_config': terrain_config,
        'grid_config': grid_config,        # None for circular
        'ogrid_config': ogrid_config,      # None for rectangular
        'mesh_config': mesh_config,
        'boundary_config': boundary_config,
        'visualization_config': visualization_config
    }
