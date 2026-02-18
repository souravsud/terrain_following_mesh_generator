"""
Terrain-to-CFD Mesh Generation Package

This package provides tools for generating structured, terrain-following meshes
for Computational Fluid Dynamics (CFD) simulations, specifically targeting
OpenFOAM atmospheric boundary layer (ABL) studies.

Main Components:
    - TerrainMeshPipeline: Main orchestration class for the complete workflow
    - Configuration classes: Type-safe configuration management
    - TerrainProcessor: DEM extraction and processing
    - GridGenerator: Structured grid generation with grading
    - BoundaryTreatment: Boundary smoothing and conditioning
    - BlockMeshGenerator: OpenFOAM mesh export

Basic Usage:
    >>> import terrain_mesh as tm
    >>> 
    >>> # Load configuration
    >>> configs = tm.load_config("config.yaml")
    >>> 
    >>> # Run pipeline
    >>> pipeline = tm.TerrainMeshPipeline()
    >>> results = pipeline.run(
    ...     dem_path="terrain.tif",
    ...     output_dir="output",
    ...     **configs
    ... )

For more information, see the README.md file.
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

__version__ = "1.0.0"
__all__ = [
    "TerrainMeshPipeline",
    "TerrainConfig",
    "GridConfig", 
    "MeshConfig",
    "BoundaryConfig",
    "VisualizationConfig",
    "load_config",
    "create_uniform_grid_config",
    "create_cfd_grading_config",
]

def create_uniform_grid_config(nx: int, ny: int) -> GridConfig:
    """Create grid configuration with uniform spacing.
    
    Args:
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction
        
    Returns:
        GridConfig with uniform spacing (no grading)
        
    Example:
        >>> grid_config = create_uniform_grid_config(200, 200)
    """
    return GridConfig(nx=nx, ny=ny)

def create_cfd_grading_config(nx: int, ny: int, boundary_refinement: float = 0.05) -> GridConfig:
    """Create grid configuration with typical CFD boundary layer refinement.
    
    This creates a grid with refined cells at domain boundaries and coarser
    cells in the center, which is common for atmospheric flow simulations.
    
    Args:
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction
        boundary_refinement: Expansion ratio at boundaries (smaller = finer)
            Default 0.05 gives 20x refinement at boundaries
            
    Returns:
        GridConfig with 3-zone grading (refined-uniform-refined)
        
    Example:
        >>> # Create grid with fine cells at boundaries
        >>> grid_config = create_cfd_grading_config(384, 384, boundary_refinement=0.05)
    """
    
    expansion_ratio = 1.0 / boundary_refinement
    grading = [
        (0.3, 0.2, boundary_refinement),  # First 30% of domain, 20% of cells, refined
        (0.4, 0.6, 1.0),                  # Middle 40%, 60% of cells, uniform
        (0.3, 0.2, expansion_ratio)       # Last 30%, 20% of cells, refined
    ]
    
    return GridConfig(nx=nx, ny=ny, x_grading=grading, y_grading=grading)