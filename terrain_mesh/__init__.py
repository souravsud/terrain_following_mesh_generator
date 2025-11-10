"""
Terrain-to-CFD Mesh Generation Package
"""
from .config import (
    TerrainConfig, 
    GridConfig, 
    MeshConfig, 
    BoundaryConfig, 
    VisualizationConfig, 
    load_config
)
from .pipeline import TerrainMeshPipeline
from .ogrid_config import OGridConfig
from .ogrid_pipeline import OGridPipeline

__version__ = "1.1.0"  # Bumped for O-grid support

def create_uniform_grid_config(nx: int, ny: int):
    """Create grid config with uniform spacing (rectangular domain)"""
    from .config import GridConfig
    return GridConfig(nx=nx, ny=ny)

def create_cfd_grading_config(nx: int, ny: int, boundary_refinement: float = 0.05):
    """Create grid config with typical CFD boundary layer refinement (rectangular domain)"""
    from .config import GridConfig
    expansion_ratio = 1.0 / boundary_refinement
    grading = [
        (0.3, 0.2, boundary_refinement),
        (0.4, 0.6, 1.0),
        (0.3, 0.2, expansion_ratio)
    ]
    return GridConfig(nx=nx, ny=ny, x_grading=grading, y_grading=grading)

def create_ogrid_config(nr: int = 50, 
                       n_sectors: int = 16,
                       n_circumferential_per_sector: int = 20,
                       aoi_cell_fraction: float = 0.4,
                       radial_expansion_ratio: float = 20.0):
    """
    Create O-grid config for circular domains.
    
    Args:
        nr: Total radial cells from center to circle
        n_sectors: Number of circumferential sectors (must be multiple of 4)
        n_circumferential_per_sector: Cells along each sector arc
        aoi_cell_fraction: Fraction of nr in center square (0-1)
        radial_expansion_ratio: Cell size ratio from square to circle
    """
    return OGridConfig(
        nr=nr,
        n_sectors=n_sectors,
        n_circumferential_per_sector=n_circumferential_per_sector,
        aoi_cell_fraction=aoi_cell_fraction,
        radial_expansion_ratio=radial_expansion_ratio
    )