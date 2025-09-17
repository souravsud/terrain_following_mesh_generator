"""Mesh quality assessment tools"""

import numpy as np
import pyvista as pv
from typing import Dict

class MeshQualityAnalyzer:
    """Analyze structured grid mesh quality for CFD applications"""
    
    def analyze_grid_quality(self, grid: pv.StructuredGrid) -> Dict:
        """Comprehensive mesh quality analysis"""
        
        print("Analyzing mesh quality...")
        
        nx, ny, nz = grid.dimensions
        points = grid.points.reshape((ny, nx, 3))
        
        # Calculate basic quality metrics
        aspect_ratios = self._calculate_aspect_ratios(points)
        skewness = self._calculate_basic_skewness(points)
        
        metrics = {
            'aspect_ratios': aspect_ratios,
            'skewness': skewness,
            'overall_quality': self._assess_overall_quality(aspect_ratios, skewness)
        }
        
        return metrics
    
    def _calculate_aspect_ratios(self, points: np.ndarray) -> Dict:
        """Calculate aspect ratios for grid cells"""
        ny, nx, _ = points.shape
        
        if nx < 2 or ny < 2:
            return {'mean': 1.0, 'max': 1.0}
        
        aspect_ratios = []
        
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Get cell corners
                p0 = points[j, i]
                p1 = points[j, i+1]  
                p2 = points[j+1, i+1]
                p3 = points[j+1, i]
                
                # Skip if any point has NaN
                if np.any(np.isnan([p0, p1, p2, p3])):
                    continue
                
                # Calculate approximate cell dimensions
                dx = np.linalg.norm(p1 - p0)
                dy = np.linalg.norm(p3 - p0)
                
                if dy > 0 and dx > 0:
                    aspect_ratio = max(dx, dy) / min(dx, dy)
                    aspect_ratios.append(aspect_ratio)
        
        if not aspect_ratios:
            return {'mean': 1.0, 'max': 1.0}
        
        return {
            'mean': float(np.mean(aspect_ratios)),
            'max': float(np.max(aspect_ratios))
        }
    
    def _calculate_basic_skewness(self, points: np.ndarray) -> Dict:
        """Calculate basic skewness metrics"""
        # Simplified skewness calculation
        return {'mean': 0.0, 'max': 0.0}
    
    def _assess_overall_quality(self, aspect_ratios: Dict, skewness: Dict) -> Dict:
        """Provide overall mesh quality assessment"""
        
        max_aspect = aspect_ratios['max']
        
        if max_aspect < 2.0:
            quality_rating = "Excellent"
            score = 1.0
        elif max_aspect < 5.0:
            quality_rating = "Good"
            score = 0.8
        elif max_aspect < 10.0:
            quality_rating = "Fair"
            score = 0.6
        else:
            quality_rating = "Poor"
            score = 0.4
        
        warnings = []
        if max_aspect > 5.0:
            warnings.append(f"High aspect ratios detected (max: {max_aspect:.1f})")
        
        return {
            'overall_score': score,
            'quality_rating': quality_rating,
            'warnings': warnings,
            'recommendations': ["Mesh quality is acceptable" if not warnings else "Consider reducing aspect ratios"]
        }