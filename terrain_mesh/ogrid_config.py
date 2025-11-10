"""Configuration for O-grid circular domain meshing"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class OGridConfig:
    """Configuration for O-grid circular domain generation"""
    
    # Grid topology
    nr: int = 50  # Total radial cells from center to circle
    n_sectors: int = 16  # Number of circumferential sectors (must be multiple of 4)
    n_circumferential_per_sector: int = 20  # Cells along each sector arc
    
    # Radial distribution
    aoi_cell_fraction: float = 0.4  # Fraction of nr in center square (rest in radial expansion)
    radial_expansion_ratio: float = 20.0  # Expansion ratio from square to circle
    
    # Center square options
    subdivide_center: bool = False  # Subdivide center square into 4 blocks (like reference)
    
    def __post_init__(self):
        """Validate O-grid configuration"""
        
        # Validate n_sectors
        if self.n_sectors % 4 != 0:
            raise ValueError(f"n_sectors must be multiple of 4, got {self.n_sectors}")
        
        if self.n_sectors < 4:
            raise ValueError(f"n_sectors must be >= 4, got {self.n_sectors}")
        
        # Validate fractions
        if not 0 < self.aoi_cell_fraction < 1:
            raise ValueError(f"aoi_cell_fraction must be between 0 and 1, got {self.aoi_cell_fraction}")
        
        if self.radial_expansion_ratio <= 1:
            raise ValueError(f"radial_expansion_ratio must be > 1, got {self.radial_expansion_ratio}")
        
        # Validate cell counts
        if self.nr < 10:
            raise ValueError(f"nr must be >= 10 for reasonable mesh, got {self.nr}")
        
        if self.n_circumferential_per_sector < 5:
            raise ValueError(f"n_circumferential_per_sector must be >= 5, got {self.n_circumferential_per_sector}")
    
    def get_n_aoi_cells(self) -> int:
        """Calculate number of cells in AOI (center square)"""
        return int(self.nr * self.aoi_cell_fraction)
    
    def get_n_radial_cells(self) -> int:
        """Calculate number of cells in radial expansion zone"""
        return self.nr - self.get_n_aoi_cells()
    
    def get_total_blocks(self) -> int:
        """Calculate total number of blocks in O-grid"""
        if self.subdivide_center:
            # 16 outer + 16 inner + 4 center
            return self.n_sectors * 2 + 4
        else:
            # 16 outer + 16 inner
            return self.n_sectors * 2
    
    def print_summary(self):
        """Print O-grid configuration summary"""
        print("\nO-Grid Configuration:")
        print(f"  Sectors: {self.n_sectors}")
        print(f"  Total radial cells: {self.nr}")
        print(f"    - AOI cells: {self.get_n_aoi_cells()}")
        print(f"    - Radial expansion cells: {self.get_n_radial_cells()}")
        print(f"  Circumferential cells per sector: {self.n_circumferential_per_sector}")
        print(f"  Radial expansion ratio: {self.radial_expansion_ratio}")
        print(f"  Total blocks: {self.get_total_blocks()}")
        print(f"  Center square subdivided: {self.subdivide_center}")