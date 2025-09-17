"""
Terrain-to-CFD Mesh Generation Package
"""

from .config import TerrainConfig, GridConfig, MeshConfig, BoundaryConfig, VisualizationConfig, load_config, save_config_template
from .pipeline import TerrainMeshPipeline

__version__ = "1.0.0"

def create_uniform_grid_config(nx: int, ny: int):
    """Create grid config with uniform spacing"""
    from .config import GridConfig
    return GridConfig(nx=nx, ny=ny)

def create_cfd_grading_config(nx: int, ny: int, boundary_refinement: float = 0.05):
    """Create grid config with typical CFD boundary layer refinement"""
    from .config import GridConfig
    
    expansion_ratio = 1.0 / boundary_refinement
    grading = [
        (0.3, 0.2, boundary_refinement),
        (0.4, 0.6, 1.0),
        (0.3, 0.2, expansion_ratio)
    ]
    
    return GridConfig(nx=nx, ny=ny, x_grading=grading, y_grading=grading)