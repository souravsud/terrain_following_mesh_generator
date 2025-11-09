"""Visualization tools for terrain and grid data"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyvista as pv
from pathlib import Path
from typing import Optional, Dict

from .config import VisualizationConfig
from .utils import rotate_coordinates

class TerrainVisualizer:
    """Handle visualization of input and output data"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def create_overview_plots(self, original_dem: np.ndarray, 
                              zones: Dict, 
                              treated_elevation: np.ndarray, 
                              output_dir: Path,
                              grid=None,
                              rotation_deg=None, 
                              crop_mask=None):
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
        
        if rotation_deg is not None and crop_mask is not None:
            
            rows, cols = np.where(crop_mask)
            center_row, center_col = np.mean(rows), np.mean(cols)
            
            y_grid, x_grid = np.mgrid[0:crop_mask.shape[0], 0:crop_mask.shape[1]]
            rel_x = x_grid - center_col
            rel_y = y_grid - center_row
            
            flow_x, flow_y = rotate_coordinates(rel_x, rel_y, 0, 0, rotation_deg)
            
            valid_flow_x = flow_x[crop_mask]
            min_x, max_x = valid_flow_x.min(), valid_flow_x.max()
            
            # Find inlet (west boundary - upwind)
            inlet_mask = (flow_x <= (min_x + 50)) & crop_mask
            if np.any(inlet_mask):
                inlet_points = np.where(inlet_mask)
                inlet_row = np.mean(inlet_points[0])
                inlet_col = np.mean(inlet_points[1])
                ax.plot(inlet_col, inlet_row, 'r^', markersize=15, label='INLET (West/Upwind)')
                ax.text(inlet_col, inlet_row - 30, 'Inlet', 
                    color='red', fontsize=12, ha='center', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Find outlet (east boundary - downwind)
            outlet_mask = (flow_x >= (max_x - 50)) & crop_mask
            if np.any(outlet_mask):
                outlet_points = np.where(outlet_mask)
                outlet_row = np.mean(outlet_points[0])
                outlet_col = np.mean(outlet_points[1])
                ax.plot(outlet_col, outlet_row, 'bv', markersize=15, label='OUTLET (East/Downwind)')
                ax.text(outlet_col, outlet_row + 30, 'Outlet', 
                    color='blue', fontsize=12, ha='center', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend(loc='upper right')
        
        # 3. Treated terrain (after boundary processing)
        ax = axes[1, 0]
        im3 = ax.imshow(treated_elevation, cmap='terrain', origin='upper')
        ax.set_title('Treated Terrain (After Boundary Processing)')
        plt.colorbar(im3, ax=ax, label='Elevation (m)')
        
        # 4. Final output using actual grid coordinates
        if grid is not None:
            ax = axes[1, 1]
            points = grid.points
            
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
    
    def create_roughness_plots(self, roughness_data: np.ndarray, roughness_transform: object,
                           z0_stats: dict, output_dir: Path, vtk_file: Optional[str] = None):
        """Create diagnostic plots for roughness map and interpolated z0"""
        if not self.config.create_plots:
            return
        
        print("Creating roughness visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Surface Roughness (z0) Analysis', fontsize=16)
        
        # Get coordinate extent
        nrows, ncols = roughness_data.shape
        x_min = roughness_transform.c
        y_max = roughness_transform.f
        x_res = roughness_transform.a
        y_res = -roughness_transform.e
        extent = [x_min, x_min + ncols * x_res, y_max - nrows * y_res, y_max]
        
        # 1. Original roughness map (cropped & rotated)
        ax = axes[0]
        roughness_masked = np.ma.masked_invalid(roughness_data)
        im1 = ax.imshow(roughness_masked, extent=extent, origin='upper', cmap='YlOrBr',
                        norm=LogNorm(vmin=max(0.0001, np.nanmin(roughness_data)), 
                                    vmax=np.nanmax(roughness_data)))
        ax.set_title('Roughness Map (Cropped & Rotated)')
        ax.set_xlabel('UTM Easting (m)')
        ax.set_ylabel('UTM Northing (m)')
        ax.set_aspect('equal')
        plt.colorbar(im1, ax=ax, label='z0 (m)')
        
        stats_text = f"Min: {np.nanmin(roughness_data):.4f} m\n"
        stats_text += f"Max: {np.nanmax(roughness_data):.4f} m\n"
        stats_text += f"Mean: {np.nanmean(roughness_data):.4f} m"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Interpolated z0 on mesh faces
        ax = axes[1]
        if vtk_file and Path(vtk_file).exists():
            mesh = pv.read(vtk_file)
            nx, ny, _ = mesh.dimensions
            points = mesh.points.reshape((ny, nx, 3))
            valid_mask = ~np.isnan(points[:, :, 2])
            
            # Read z0 values - FIXED PARSING
            z0_file = output_dir / '0' / 'include' / 'z0Values'
            if z0_file.exists():
                z0_values = []
                with open(z0_file, 'r') as f:
                    in_values = False
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith('//'):
                            continue
                        # Found opening parenthesis - start reading values
                        if line == '(':
                            in_values = True
                            continue
                        # Found closing parenthesis - stop
                        if line == ')':
                            break
                        # Read numeric values
                        if in_values:
                            try:
                                z0_values.append(float(line))
                            except ValueError:
                                continue  # Skip non-numeric lines
                
                z0_values = np.array(z0_values)
                print(f"Read {len(z0_values)} z0 values from file")
                
                # Get face centers
                face_x, face_y, face_z0 = [], [], []
                idx = 0
                for j in range(ny - 1):
                    for i in range(nx - 1):
                        if (valid_mask[j, i] and valid_mask[j, i+1] and 
                            valid_mask[j+1, i+1] and valid_mask[j+1, i]):
                            center = (points[j, i, :2] + points[j, i+1, :2] + 
                                    points[j+1, i+1, :2] + points[j+1, i, :2]) / 4.0
                            face_x.append(center[0])
                            face_y.append(center[1])
                            face_z0.append(z0_values[idx])
                            idx += 1
                
                scatter = ax.scatter(face_x, face_y, c=face_z0, cmap='YlOrBr', s=5, alpha=0.8,
                                norm=LogNorm(vmin=max(0.0001, min(face_z0)), vmax=max(face_z0)))
                ax.set_title('Interpolated z0 on Ground Faces')
                ax.set_xlabel('UTM Easting (m)')
                ax.set_ylabel('UTM Northing (m)')
                ax.set_aspect('equal')
                plt.colorbar(scatter, ax=ax, label='z0 (m)')
                
                stats_text2 = f"Faces: {z0_stats['n_faces']:,}\n"
                stats_text2 += f"Min: {z0_stats['z0_min']:.4f} m\n"
                stats_text2 += f"Max: {z0_stats['z0_max']:.4f} m\n"
                stats_text2 += f"Mean: {z0_stats['z0_mean']:.4f} m"
                ax.text(0.02, 0.98, stats_text2, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'z0Values file not found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'VTK file not available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f'roughness_analysis.{self.config.plot_format}'
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
        print(f"Roughness plots saved: {output_path}")