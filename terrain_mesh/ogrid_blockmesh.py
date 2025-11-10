"""Generate OpenFOAM blockMeshDict for O-grid circular domains"""

import os
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import pyvista as pv

from .ogrid_config import OGridConfig


class OGridBlockMeshGenerator:
    """Generate blockMeshDict for O-grid topology"""
    
    def generate_blockMeshDict(self,
                          mesh_config: object,
                          ogrid_config: OGridConfig,
                          input_vtk_file: str,
                          output_dict_file: str = "system/blockMeshDict"):
        """
        Generate O-grid blockMeshDict from VTK file.
        
        Reads full O-grid terrain mesh and extracts vertices for block corners.
        """
        
        print("\n" + "="*60)
        print("Generating O-Grid blockMeshDict")
        print("="*60)
        
        import pyvista as pv
        
        # Read VTK file
        mesh = pv.read(input_vtk_file)
        nr, n_sectors, _ = mesh.dimensions
        
        print(f"Read O-grid VTK: {n_sectors} sectors × {nr} radial points")
        
        # Reshape points to grid structure
        points = mesh.points.reshape((n_sectors, nr, 3))
        
        # Extract coordinates
        x_grid = points[:, :, 0]  # (n_sectors, nr)
        y_grid = points[:, :, 1]
        z_grid = points[:, :, 2]  # Terrain elevations
        
        # Select vertices for blockMesh blocks
        # We need specific radial positions for the O-grid topology:
        # - Center (r=0)
        # - Square corners (r = aoi radius)
        # - Inner ring (r = transition)
        # - Outer circle (r = max)
        
        n_aoi_cells = ogrid_config.get_n_aoi_cells()
        n_radial_cells = ogrid_config.get_n_radial_cells()
        
        # Radial indices for block corners
        idx_center = 0
        idx_square = n_aoi_cells  # Inner edge of radial expansion
        idx_inner_ring = n_aoi_cells  # Same as square for simplicity
        idx_outer = nr - 1  # Outer circle
        
        # Extract vertices at these radial positions
        vertices_ground = []
        vertices_elevations = []
        
        # 1. Outer circle vertices (n_sectors points)
        for i in range(n_sectors):
            x = x_grid[i, idx_outer]
            y = y_grid[i, idx_outer]
            z = z_grid[i, idx_outer]
            vertices_ground.append([x, y])
            vertices_elevations.append(z)
        
        # 2. Inner ring vertices (n_sectors points)
        for i in range(n_sectors):
            x = x_grid[i, idx_inner_ring]
            y = y_grid[i, idx_inner_ring]
            z = z_grid[i, idx_inner_ring]
            vertices_ground.append([x, y])
            vertices_elevations.append(z)
        
        # 3. Square corner vertices (4 points)
        # These should be at sectors aligned with square (0, 90, 180, 270 degrees)
        corner_sectors = [0, n_sectors//4, n_sectors//2, 3*n_sectors//4]
        for i in corner_sectors:
            x = x_grid[i, idx_square]
            y = y_grid[i, idx_square]
            z = z_grid[i, idx_square]
            vertices_ground.append([x, y])
            vertices_elevations.append(z)
        
        # 4. Center point (if subdividing)
        if ogrid_config.subdivide_center:
            x = x_grid[0, idx_center]
            y = y_grid[0, idx_center]
            z = z_grid[0, idx_center]
            vertices_ground.append([x, y])
            vertices_elevations.append(z)
        
        vertices_ground = np.array(vertices_ground)
        vertices_elevations = np.array(vertices_elevations)
        
        print(f"Extracted {len(vertices_ground)} vertices for blockMesh")
        print(f"  Outer circle: {n_sectors}")
        print(f"  Inner ring: {n_sectors}")
        print(f"  Square corners: 4")
        if ogrid_config.subdivide_center:
            print(f"  Center: 1")
        
        # Calculate z-grading
        domain_height = mesh_config.domain_height
        total_z_cells = mesh_config.total_z_cells
        terrain_normal_first_layer = mesh_config.terrain_normal_first_layer
        
        z_grading_spec, first_cell_height = self._calculate_z_grading_spec(
            domain_height, mesh_config.z_grading, total_z_cells, terrain_normal_first_layer)
        
        # Calculate radial grading
        radial_expansion = ogrid_config.radial_expansion_ratio
        
        # Create output directory
        os.makedirs(os.path.dirname(output_dict_file), exist_ok=True)
        
        # Write blockMeshDict
        with open(output_dict_file, 'w') as f:
            self._write_header(f)
            
            # Vertices (using extracted vertices)
            self._write_vertices(f, vertices_ground, vertices_elevations, domain_height,
                            first_cell_height, terrain_normal_first_layer)
            
            # Blocks
            self._write_blocks(f, ogrid_config, n_aoi_cells, n_radial_cells,
                            radial_expansion, z_grading_spec, total_z_cells,
                            terrain_normal_first_layer)
            
            # Edges (arcs for smooth circle)
            self._write_edges(f, ogrid_config, len(vertices_ground))
            
            # Boundaries
            self._write_boundaries(f, ogrid_config, terrain_normal_first_layer)
            
            f.write("\nmergePatchPairs\n(\n);\n")
            f.write("\n// ************************************************************************* //\n")
        
        print(f"\n✓ O-Grid blockMeshDict saved to: {output_dict_file}")
        print(f"  Total blocks: {ogrid_config.get_total_blocks()}")
        print(f"  Sectors: {ogrid_config.n_sectors}")
        print(f"  Radial cells: AOI={n_aoi_cells}, Expansion={n_radial_cells}")
        print(f"  Z-cells: {total_z_cells}")
        print("="*60)
    
    def _write_header(self, f):
        """Write OpenFOAM header"""
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("| =========                 |                                                 |\n")
        f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
        f.write("|  \\\\    /   O peration     | Version:  v2312                                 |\n")
        f.write("|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |\n")
        f.write("|    \\\\/     M anipulation  |                                                 |\n")
        f.write("\\*---------------------------------------------------------------------------*/\n")
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n")
        f.write("    class       dictionary;\n    object      blockMeshDict;\n}\n")
        f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
        f.write("// O-Grid circular domain generated by TerrainMeshPipeline\n\n")
        f.write("convertToMeters 1;\n\n")
    
    def _write_vertices(self, f, vertices_ground, elevations, domain_height,
                       first_cell_height, terrain_normal_first_layer):
        """Write vertices section"""
        
        n_ground = len(vertices_ground)
        
        f.write("vertices\n(\n")
        f.write("    // Ground level vertices\n")
        
        # Ground vertices
        for i, ((x, y), z) in enumerate(zip(vertices_ground, elevations)):
            if np.isnan(z):
                z = 0.0  # Fallback for NaN
            f.write(f"    ({x:.6f} {y:.6f} {z:.6f})  // v{i} ground\n")
        
        # First layer top vertices (if terrain_normal_first_layer)
        if terrain_normal_first_layer:
            f.write("\n    // First layer top vertices (terrain normal offset)\n")
            for i, ((x, y), z) in enumerate(zip(vertices_ground, elevations)):
                if np.isnan(z):
                    z = 0.0
                # For now, simple vertical offset (normals would need more work)
                z_top = z + first_cell_height
                f.write(f"    ({x:.6f} {y:.6f} {z_top:.6f})  // v{i + n_ground} first_layer_top\n")
            
            # Sky vertices
            f.write("\n    // Sky level vertices\n")
            for i, ((x, y), z) in enumerate(zip(vertices_ground, elevations)):
                f.write(f"    ({x:.6f} {y:.6f} {domain_height:.6f})  // v{i + 2*n_ground} sky\n")
        else:
            # Sky vertices
            f.write("\n    // Sky level vertices\n")
            for i, ((x, y), z) in enumerate(zip(vertices_ground, elevations)):
                f.write(f"    ({x:.6f} {y:.6f} {domain_height:.6f})  // v{i + n_ground} sky\n")
        
        f.write(");\n\n")
    
    def _write_blocks(self, f, ogrid_config, n_aoi_cells, n_radial_cells,
                     radial_expansion, z_grading_spec, total_z_cells,
                     terrain_normal_first_layer):
        """Write blocks section"""
        
        n_sectors = ogrid_config.n_sectors
        n_circ = ogrid_config.n_circumferential_per_sector
        n_ground = self._get_vertex_count(ogrid_config)
        
        outer_start = 0
        inner_start = n_sectors
        square_start = 2 * n_sectors
        center_idx = 2 * n_sectors + 4 if ogrid_config.subdivide_center else None
        
        f.write("blocks\n(\n")
        
        # Outer radial blocks
        f.write("    // Outer radial blocks (square edge to circle)\n")
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            
            v0 = outer_start + i
            v1 = outer_start + i_next
            v2 = inner_start + i_next
            v3 = inner_start + i
            
            if terrain_normal_first_layer:
                v4 = v0 + n_ground
                v5 = v1 + n_ground
                v6 = v2 + n_ground
                v7 = v3 + n_ground
                v8 = v0 + 2 * n_ground
                v9 = v1 + 2 * n_ground
                v10 = v2 + 2 * n_ground
                v11 = v3 + 2 * n_ground
                
                # First layer (uniform)
                f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                f.write(f"({n_circ} {n_radial_cells} 1) simpleGrading ({radial_expansion} 1 1)\n")
                
                # Upper layers (graded)
                f.write(f"    hex ({v4} {v5} {v6} {v7} {v8} {v9} {v10} {v11}) ")
                f.write(f"({n_circ} {n_radial_cells} {total_z_cells-1}) {z_grading_spec}\n")
            else:
                v4 = v0 + n_ground
                v5 = v1 + n_ground
                v6 = v2 + n_ground
                v7 = v3 + n_ground
                
                f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                f.write(f"({n_circ} {n_radial_cells} {total_z_cells}) {z_grading_spec}\n")
        
        # Inner transition blocks
        f.write("\n    // Inner transition blocks (inner ring to square)\n")
        corner_sectors = [0, n_sectors//4, n_sectors//2, 3*n_sectors//4]
        
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            
            # Simplified topology: connect to nearest square corners
            corner_idx = min(range(4), key=lambda j: abs((i - corner_sectors[j]) % n_sectors))
            next_corner_idx = (corner_idx + 1) % 4
            
            v0 = inner_start + i
            v1 = inner_start + i_next
            v2 = square_start + next_corner_idx
            v3 = square_start + corner_idx
            
            if terrain_normal_first_layer:
                v4 = v0 + n_ground
                v5 = v1 + n_ground
                v6 = v2 + n_ground
                v7 = v3 + n_ground
                v8 = v0 + 2 * n_ground
                v9 = v1 + 2 * n_ground
                v10 = v2 + 2 * n_ground
                v11 = v3 + 2 * n_ground
                
                f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                f.write(f"({n_circ} {n_aoi_cells} 1) simpleGrading (1 1 1)\n")
                
                f.write(f"    hex ({v4} {v5} {v6} {v7} {v8} {v9} {v10} {v11}) ")
                f.write(f"({n_circ} {n_aoi_cells} {total_z_cells-1}) {z_grading_spec}\n")
            else:
                v4 = v0 + n_ground
                v5 = v1 + n_ground
                v6 = v2 + n_ground
                v7 = v3 + n_ground
                
                f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                f.write(f"({n_circ} {n_aoi_cells} {total_z_cells}) {z_grading_spec}\n")
        
        # Center square blocks (if subdivided)
        if ogrid_config.subdivide_center:
            f.write("\n    // Center square blocks\n")
            for i in range(4):
                i_next = (i + 1) % 4
                
                v0 = square_start + i
                v1 = square_start + i_next
                v2 = center_idx
                v3 = center_idx
                
                if terrain_normal_first_layer:
                    v4 = v0 + n_ground
                    v5 = v1 + n_ground
                    v6 = v2 + n_ground
                    v7 = v3 + n_ground
                    v8 = v0 + 2 * n_ground
                    v9 = v1 + 2 * n_ground
                    v10 = v2 + 2 * n_ground
                    v11 = v3 + 2 * n_ground
                    
                    f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                    f.write(f"({n_circ} {n_aoi_cells} 1) simpleGrading (1 1 1)\n")
                    
                    f.write(f"    hex ({v4} {v5} {v6} {v7} {v8} {v9} {v10} {v11}) ")
                    f.write(f"({n_circ} {n_aoi_cells} {total_z_cells-1}) {z_grading_spec}\n")
                else:
                    v4 = v0 + n_ground
                    v5 = v1 + n_ground
                    v6 = v2 + n_ground
                    v7 = v3 + n_ground
                    
                    f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                    f.write(f"({n_circ} {n_aoi_cells} {total_z_cells}) {z_grading_spec}\n")
        
        f.write(");\n\n")
    
    def _write_edges(self, f, ogrid_config, n_vertices):
        """Write edges section (arcs for circular outer boundary)"""
        
        f.write("edges\n(\n")
        f.write("    // Arcs for smooth circular outer boundary\n")
        
        n_sectors = ogrid_config.n_sectors
        outer_start = 0
        n_ground = self._get_vertex_count(ogrid_config)
        
        # Add arcs between outer circle vertices
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            i_mid = (2 * i + 1) % (2 * n_sectors)  # Midpoint angle
            
            # Ground arcs
            v0 = outer_start + i
            v1 = outer_start + i_next
            
            # Sky arcs
            v0_sky = v0 + n_ground
            v1_sky = v1 + n_ground
            
            f.write(f"    // arc {v0} {v1} (midpoint)  // Ground arc sector {i}\n")
            f.write(f"    // arc {v0_sky} {v1_sky} (midpoint)  // Sky arc sector {i}\n")
        
        f.write(");\n\n")
    
    def _write_boundaries(self, f, ogrid_config, terrain_normal_first_layer):
        """Write boundaries section"""
        
        n_sectors = ogrid_config.n_sectors
        outer_start = 0
        n_ground = self._get_vertex_count(ogrid_config)
        
        f.write("boundary\n(\n")
        
        # Ground patch (all bottom faces)
        f.write("    ground\n    {\n        type wall;\n        faces\n        (\n")
        # Write all ground faces (outer blocks)
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            v0 = outer_start + i
            v1 = outer_start + i_next
            v2 = n_sectors + i_next
            v3 = n_sectors + i
            f.write(f"            ({v0} {v3} {v2} {v1})\n")
        
        # Inner blocks ground faces
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            corner_idx = min(range(4), key=lambda j: abs((i - [0, n_sectors//4, n_sectors//2, 3*n_sectors//4][j]) % n_sectors))
            next_corner_idx = (corner_idx + 1) % 4
            
            v0 = n_sectors + i
            v1 = n_sectors + i_next
            v2 = 2*n_sectors + next_corner_idx
            v3 = 2*n_sectors + corner_idx
            f.write(f"            ({v0} {v3} {v2} {v1})\n")
        
        f.write("        );\n    }\n\n")
        
        # Sky patch
        f.write("    sky\n    {\n        type patch;\n        faces\n        (\n")
        
        if terrain_normal_first_layer:
            sky_offset = 2 * n_ground
        else:
            sky_offset = n_ground
        
        # Outer blocks sky faces
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            v0 = outer_start + i + sky_offset
            v1 = outer_start + i_next + sky_offset
            v2 = n_sectors + i_next + sky_offset
            v3 = n_sectors + i + sky_offset
            f.write(f"            ({v0} {v1} {v2} {v3})\n")
        
        # Inner blocks sky faces
        for i in range(n_sectors):
            i_next = (i + 1) % n_sectors
            corner_idx = min(range(4), key=lambda j: abs((i - [0, n_sectors//4, n_sectors//2, 3*n_sectors//4][j]) % n_sectors))
            next_corner_idx = (corner_idx + 1) % 4
            
            v0 = n_sectors + i + sky_offset
            v1 = n_sectors + i_next + sky_offset
            v2 = 2*n_sectors + next_corner_idx + sky_offset
            v3 = 2*n_sectors + corner_idx + sky_offset
            f.write(f"            ({v0} {v1} {v2} {v3})\n")
        
        f.write("        );\n    }\n\n")
        
        # Lateral patches (one per sector)
        for sector_idx in range(n_sectors):
            i_next = (sector_idx + 1) % n_sectors
            
            f.write(f"    lateral{sector_idx+1:02d}\n    {{\n        type patch;\n")
            f.write(f"        inGroups (lateral);\n        faces\n        (\n")
            
            if terrain_normal_first_layer:
                # Two faces per sector (first layer + upper layers)
                v0 = outer_start + sector_idx
                v1 = outer_start + i_next
                v4 = v0 + n_ground
                v5 = v1 + n_ground
                v8 = v0 + 2 * n_ground
                v9 = v1 + 2 * n_ground
                
                f.write(f"            ({v0} {v1} {v5} {v4})\n")
                f.write(f"            ({v4} {v5} {v9} {v8})\n")
            else:
                v0 = outer_start + sector_idx
                v1 = outer_start + i_next
                v4 = v0 + n_ground
                v5 = v1 + n_ground
                
                f.write(f"            ({v0} {v1} {v5} {v4})\n")
            
            f.write(f"        );\n    }}\n\n")
        
        f.write(");\n")
    
    def _get_vertex_count(self, ogrid_config: OGridConfig) -> int:
        """Calculate total number of ground vertices"""
        n_sectors = ogrid_config.n_sectors
        count = 2 * n_sectors + 4  # outer + inner + 4 corners
        if ogrid_config.subdivide_center:
            count += 1  # center point
        return count
    
    def _calculate_z_grading_spec(self, domain_height, z_grading, total_z_cells, 
                                 terrain_normal_first_layer):
        """Calculate z-direction grading specification (reuse from existing code)"""
        
        # Simplified version - can import from BlockMeshGenerator if needed
        if len(z_grading) == 1:
            expansion_ratio = z_grading[0][2]
            first_cell_height = 0.0  # Placeholder
            return f"simpleGrading (1 1 {expansion_ratio})", first_cell_height
        else:
            # Multi-grading
            grading_parts = []
            for length_frac, cell_frac, expansion_ratio in z_grading:
                grading_parts.append(f"({length_frac} {cell_frac} {expansion_ratio})")
            grading_str = " ".join(grading_parts)
            first_cell_height = 0.0  # Placeholder
            return f"multiGrading (1 1 ({grading_str}))", first_cell_height