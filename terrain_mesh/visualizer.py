"""Visualization tools for terrain and grid data"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
from typing import Optional, Dict

from .config import VisualizationConfig

class TerrainVisualizer:
    """Handle visualization of input and output data"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def create_overview_plots(self, original_dem: np.ndarray, zones: Dict, treated_elevation: np.ndarray, 
                         final_elevation: np.ndarray, output_dir: Path,pv_grid=None):
        """Create diagnostic plots for boundary treatment"""
        
        if not self.config.create_plots:
            return
        
        print("Creating boundary treatment visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Boundary Treatment Analysis', fontsize=16)
        
        # 1. Original terrain (before cropping)
        ax = axes[0, 0]
        im1 = ax.imshow(original_dem, cmap='terrain', origin='upper')
        ax.set_title('Original DEM (Before Cropping)')
        plt.colorbar(im1, ax=ax, label='Elevation (m)')
        
        # 2. Zone mask visualization
        ax = axes[0, 1]
        zone_mask = np.zeros_like(treated_elevation)
        if 'aoi' in zones:
            zone_mask[zones['aoi']] = 1  # AOI = 1
        if 'transition' in zones:
            zone_mask[zones['transition']] = 2  # Transition = 2
        if 'blend' in zones:
            zone_mask[zones['blend']] = 3  # Flat = 3
        if 'flat' in zones:
            zone_mask[zones['flat']] = 4  # Flat = 3
        
        im2 = ax.imshow(zone_mask, cmap='viridis', origin='upper')
        ax.set_title('Smoothing Zones\n(AOI=1, Transition=2, Blend=3, Flat=4)')
        plt.colorbar(im2, ax=ax, label='Zone Type')
        
        # 3. Treated terrain (after boundary processing)
        ax = axes[1, 0]
        im3 = ax.imshow(treated_elevation, cmap='terrain', origin='upper')
        ax.set_title('Treated Terrain (After Boundary Processing)')
        plt.colorbar(im3, ax=ax, label='Elevation (m)')
        
        # 4. Final output using actual grid coordinates
        if pv_grid is not None:
            ax = axes[1, 1]
            points = pv_grid.points
            
            # Downsample
            step = 20
            sample_indices = np.arange(0, len(points), step)
            
            x_coords = points[sample_indices, 0]  # Real X coordinates
            y_coords = points[sample_indices, 1]  # Real Y coordinates  
            elevations = points[sample_indices, 2] # Real Z coordinates
            
            scatter = ax.scatter(x_coords, y_coords, c=elevations, cmap='terrain', s=1)
            ax.set_title('Grid Point Distribution (Downsampled)')
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax, label='Elevation (m)')
            
            plt.tight_layout()
            
            output_path = output_dir / f'boundary_treatment.{self.config.plot_format}'
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
        
        print(f"Boundary treatment plots saved: {output_path}")