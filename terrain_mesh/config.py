"""Configuration classes for terrain mesh generation"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import yaml

@dataclass
class TerrainConfig:
    """Configuration for terrain processing"""
    center_lat: float
    center_lon: float
    crop_size_km: float
    rotation_deg: float = 0.0
    smoothing_sigma: float = 2.0
    center_coordinates: bool = False
    
    def __post_init__(self):
        if not (-90 <= self.center_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= self.center_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
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
            raise ValueError(f"{name} length fractions must sum to 1.0, got {length_sum}")
        if abs(cell_sum - 1.0) > 1e-6:
            raise ValueError(f"{name} cell fractions must sum to 1.0, got {cell_sum}")

@dataclass
class MeshConfig:
    """Configuration for OpenFOAM mesh generation"""
    domain_height: float = 4000.0
    
    # Z-direction configuration
    z_grading: Optional[List[Tuple[float, float, float]]] = None
    total_z_cells: Optional[int] = None
    terrain_normal_first_layer: bool = False
    
    patch_types: Optional[Dict[str, str]] = None
    extract_inlet_face_info: bool = True
    
    def __post_init__(self):
        if self.patch_types is None:
            self.patch_types = {
                'ground': 'wall',
                'sky': 'patch', 
                'inlet': 'patch',
                'outlet': 'patch',
                'sides': 'patch'
            }
        
        # Validate z_grading if specified
        if self.z_grading:
            self._validate_z_grading()
        
    
    def _validate_z_grading(self):
        """Validate z-direction grading specification"""
        length_sum = sum(spec[0] for spec in self.z_grading)
        cell_sum = sum(spec[1] for spec in self.z_grading)
        
        if abs(length_sum - 1.0) > 1e-6:
            raise ValueError(f"z_grading length fractions must sum to 1.0, got {length_sum}")
        if abs(cell_sum - 1.0) > 1e-6:
            raise ValueError(f"z_grading cell fractions must sum to 1.0, got {cell_sum}")

@dataclass
class VisualizationConfig:
    """Configuration for visualization options"""
    create_plots: bool = True
    show_grid_lines: bool = True
    save_high_res: bool = True
    plot_format: str = 'png'
    dpi: int = 150

@dataclass
class BoundaryConfig:
    """Configuration for boundary treatment with progressive smoothing"""
    
    # Zone definition
    aoi_fraction: float = 0.4
    
    # Treatment mode
    boundary_mode: str = 'uniform'  # 'uniform' or 'directional'
    
    # Boundary sampling parameters
    flat_boundary_thickness_fraction: float = 0.1
    enabled_boundaries: List[str] = None  # For directional mode ['east', 'west']
    
    # Progressive smoothing parameters
    smoothing_method: str = 'mean'  # 'gaussian', 'mean', 'median'
    kernel_progression: str = 'exponential'  # 'exponential', 'linear'
    base_kernel_size: int = None
    max_kernel_size: int = None
    progression_rate: float = 1.5  # For exponential progression
    
    # Boundary flatness treatment
    boundary_flatness_mode: str = 'heavy_smooth'  # 'heavy_smooth', 'blend_target'
    uniform_elevation: Optional[float] = None  # Override calculated boundary height
    
    # Legacy parameters (for backward compatibility if needed)
    flat_fraction: float = 0.05  # Not used in progressive smoothing
    flat_boundaries: List[str] = None  # Not used in progressive smoothing
    transition_smoothing_sigma: float = 3.0  # Not used in progressive smoothing
    transition_iterations: int = 10  # Not used in progressive smoothing
    
    def __post_init__(self):
        if self.enabled_boundaries is None:
            if self.boundary_mode == 'directional':
                self.enabled_boundaries = ['east', 'west']
            else:
                self.enabled_boundaries = ['uniform']
        
        # Legacy compatibility
        if self.flat_boundaries is None:
            self.flat_boundaries = ['north', 'south', 'east', 'west']


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file and return instantiated config objects.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with config object instances ready for pipeline.run(**configs)
    """
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file) or {}
    
    # Create config objects with defaults, override with YAML values
    configs = {}
    
    # Terrain configuration
    terrain_data = config_data.get('terrain', {})
    configs['terrain_config'] = TerrainConfig(**terrain_data)
    
    # Grid configuration
    grid_data = config_data.get('grid', {})
    # Handle grading arrays - convert lists of lists to lists of tuples
    if 'x_grading' in grid_data:
        grid_data['x_grading'] = [tuple(spec) for spec in grid_data['x_grading']]
    if 'y_grading' in grid_data:
        grid_data['y_grading'] = [tuple(spec) for spec in grid_data['y_grading']]
    configs['grid_config'] = GridConfig(**grid_data)
    
    # Mesh configuration
    mesh_data = config_data.get('mesh', {})
    # Handle z_grading array - convert lists of lists to lists of tuples  
    if 'z_grading' in mesh_data:
        mesh_data['z_grading'] = [tuple(spec) for spec in mesh_data['z_grading']]
    configs['mesh_config'] = MeshConfig(**mesh_data)
    
    # Boundary configuration
    boundary_data = config_data.get('boundary', {})
    configs['boundary_config'] = BoundaryConfig(**boundary_data)
    
    # Visualization configuration
    viz_data = config_data.get('visualization', {})
    configs['visualization_config'] = VisualizationConfig(**viz_data)
    
    return configs