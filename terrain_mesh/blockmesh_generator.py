import os
import numpy as np
import pyvista as pv
from typing import Dict, List
from pathlib import Path

from .config import MeshConfig

class BlockMeshGenerator:
    """Generate OpenFOAM blockMeshDict from structured grid"""

    def generate_blockMeshDict(self, config: MeshConfig, input_vtk_file: str = "terrain_structured.vtk", output_dict_file: str = "system/blockMeshDict"):
        """Wrapper method that uses MeshConfig"""
        return self._blockMeshDictCreator(
            input_vtk_file=input_vtk_file,
            output_dict_file=output_dict_file,
            domain_height=config.domain_height,
            num_cells_z=config.num_cells_z,
            expansion_ratio_R=config.expansion_ratio_z,  # Note: your config uses expansion_ratio_z
            patch_types=config.patch_types
        )

    def _blockMeshDictCreator(self,
            input_vtk_file="terrain_structured.vtk",
            output_dict_file="system/blockMeshDict",
            domain_height=4000.0,
            num_cells_z=20,
            expansion_ratio_R=20.0,
            patch_types=None
        ):
        """
        Generates an OpenFOAM blockMeshDict for terrain-following mesh, skipping NaN regions.
        
        Args:
            input_vtk_file (str): Path to the input VTK file.
            output_dict_file (str): Path for the output blockMeshDict file.
            domain_height (float): Height of the computational domain (sky patch).
            num_cells_z (int): Number of cells in the vertical (z) direction.
            expansion_ratio_R (float): The simpleGrading expansion ratio in z.
            patch_types (dict): Patch types for boundaries. Default:
                {
                    'ground': 'wall',
                    'sky': 'patch', 
                    'inlet': 'patch',
                    'outlet': 'patch',
                    'sides': 'patch'
                }
        """
        
        # Set default patch types if not provided
        if patch_types is None:
            patch_types = {
                'ground': 'wall',
                'sky': 'patch',
                'inlet': 'patch', 
                'outlet': 'patch',
                'sides': 'patch'
            }
        
        try:
            # Read VTK file
            mesh = pv.read(input_vtk_file)
            nx, ny, nz = mesh.dimensions
            points = mesh.points.reshape((ny, nx, 3))
            print(f"Read structured grid: {nx}x{ny} with {points.shape} points")
            
            # Extract coordinates
            x_coords = points[:, :, 0]  # (ny, nx)
            y_coords = points[:, :, 1]  # (ny, nx)  
            z_coords = points[:, :, 2]  # (ny, nx)
            
            # Create validity mask (not NaN)
            valid_mask = ~np.isnan(z_coords)
            nan_count = np.sum(~valid_mask)
            print(f"Found {nan_count}/{valid_mask.size} NaN points ({100*nan_count/valid_mask.size:.1f}%)")
            
            # Create vertex mapping (only for valid points)
            vertex_map = np.full((ny, nx), -1, dtype=int)  # -1 for invalid
            valid_vertices = []
            vertex_counter = 0
            
            # Map valid ground vertices
            for j in range(ny):
                for i in range(nx):
                    if valid_mask[j, i]:
                        vertex_map[j, i] = vertex_counter
                        x, y, z = points[j, i]
                        valid_vertices.append((x, y, z))  # Ground vertex
                        vertex_counter += 1
            
            num_ground_vertices = vertex_counter
            
            # Map valid sky vertices  
            for j in range(ny):
                for i in range(nx):
                    if valid_mask[j, i]:
                        x, y, z = points[j, i]
                        valid_vertices.append((x, y, domain_height))  # Sky vertex
                        vertex_counter += 1
            
            print(f"Created {num_ground_vertices} ground + {num_ground_vertices} sky = {len(valid_vertices)} vertices")
            
            # Find valid blocks and store positions
            valid_blocks = []
            block_positions = {}  # Store (i,j) -> block_vertices mapping
            
            for j in range(ny - 1):
                for i in range(nx - 1):
                    # Check if all 4 corners are valid
                    corners_valid = (valid_mask[j, i] and valid_mask[j, i+1] and 
                                valid_mask[j+1, i+1] and valid_mask[j+1, i])
                    
                    if corners_valid:
                        # Get vertex indices
                        v0 = vertex_map[j, i]
                        v1 = vertex_map[j, i+1] 
                        v2 = vertex_map[j+1, i+1]
                        v3 = vertex_map[j+1, i]
                        v4 = v0 + num_ground_vertices  # Sky vertices
                        v5 = v1 + num_ground_vertices
                        v6 = v2 + num_ground_vertices
                        v7 = v3 + num_ground_vertices
                        
                        block_vertices = (v0, v1, v2, v3, v4, v5, v6, v7)
                        valid_blocks.append(block_vertices)
                        block_positions[(i, j)] = block_vertices
            
            print(f"Created {len(valid_blocks)} valid blocks (skipped {(nx-1)*(ny-1) - len(valid_blocks)} blocks with NaN)")
            
            # Detect boundary patches by direction
            boundary_patches = self.detect_boundary_patches(block_positions, nx, ny)
            
            # Create output directory
            output_dir = os.path.dirname(output_dict_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            
            # Write blockMeshDict
            with open(output_dict_file, 'w') as f:
                # Header
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
                f.write("convertToMeters 1;\n\n")
                
                # Vertices (only valid ones)
                f.write("vertices\n(\n")
                for i, (x, y, z) in enumerate(valid_vertices):
                    f.write(f"    ({x:.4f} {y:.4f} {z:.4f}) // v{i}\n")
                f.write(");\n\n")
                
                # Blocks (only valid ones)
                f.write("blocks\n(\n")
                for v0, v1, v2, v3, v4, v5, v6, v7 in valid_blocks:
                    f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                    f.write(f"(1 1 {num_cells_z}) simpleGrading (1 1 {expansion_ratio_R})\n")
                f.write(");\n\n")
                
                # Boundaries with configurable patch types
                f.write("boundary\n(\n")
                
                # Ground patch
                f.write(f"    ground\n    {{\n        type {patch_types['ground']};\n        faces\n        (\n")
                for v0, v1, v2, v3, v4, v5, v6, v7 in valid_blocks:
                    f.write(f"            ({v0} {v3} {v2} {v1})\n")
                f.write("        );\n    }\n\n")
                
                # Sky patch  
                f.write(f"    sky\n    {{\n        type {patch_types['sky']};\n        faces\n        (\n")
                for v0, v1, v2, v3, v4, v5, v6, v7 in valid_blocks:
                    f.write(f"            ({v4} {v5} {v6} {v7})\n")
                f.write("        );\n    }\n\n")
                
                # Inlet patch (left boundary)
                if boundary_patches['inlet']:
                    f.write(f"    inlet\n    {{\n        type {patch_types['inlet']};\n        faces\n        (\n")
                    for face in boundary_patches['inlet']:
                        f.write(f"            {face}\n")
                    f.write("        );\n    }\n\n")
                
                # Outlet patch (right boundary)
                if boundary_patches['outlet']:
                    f.write(f"    outlet\n    {{\n        type {patch_types['outlet']};\n        faces\n        (\n")
                    for face in boundary_patches['outlet']:
                        f.write(f"            {face}\n")
                    f.write("        );\n    }\n\n")
                
                # Sides patch (front/back boundaries)
                if boundary_patches['sides']:
                    f.write(f"    sides\n    {{\n        type {patch_types['sides']};\n        faces\n        (\n")
                    for face in boundary_patches['sides']:
                        f.write(f"            {face}\n")
                    f.write("        );\n    }\n\n")
                
                f.write(");\n\n")
                f.write("// ************************************************************************* //\n")
            
            print(f"\nSuccessfully generated blockMeshDict at '{output_dict_file}'")
            total_cells = len(valid_blocks) * num_cells_z
            print(f"Total blocks: {len(valid_blocks)}, Total cells: {total_cells}")
            
            # Print boundary summary
            print(f"\nBoundary patches created:")
            print(f"  ground ({patch_types['ground']}): {len(valid_blocks)} faces")
            print(f"  sky ({patch_types['sky']}): {len(valid_blocks)} faces")
            for patch_name in ['inlet', 'outlet', 'sides']:
                if boundary_patches[patch_name]:
                    print(f"  {patch_name} ({patch_types[patch_name]}): {len(boundary_patches[patch_name])} faces")
            
        except Exception as e:
            print(f"Error: {e}")



    def detect_boundary_patches(self, block_positions, nx, ny):
        """
        Detect directional boundary patches based on grid position.
        
        Args:
            block_positions: dict mapping (i,j) -> block_vertices
            nx, ny: grid dimensions
        
        Returns:
            dict: {'inlet': [faces], 'outlet': [faces], 'sides': [faces]}
        """
        boundary_patches = {
            'inlet': [],    # Front boundary (j-1 neighbor missing) - wind inlet
            'outlet': [],   # Back boundary (j+1 neighbor missing) - wind outlet  
            'sides': []     # Left/right boundaries (i±1 neighbors missing)
        }
        
        # Create set for fast neighbor lookup
        block_set = set(block_positions.keys())
        
        for (i, j), (v0, v1, v2, v3, v4, v5, v6, v7) in block_positions.items():
            # Front boundary (inlet) - missing front neighbor
            if (i, j-1) not in block_set:
                boundary_patches['inlet'].append(f"({v0} {v1} {v5} {v4})")
            
            # Back boundary (outlet) - missing back neighbor
            if (i, j+1) not in block_set:
                boundary_patches['outlet'].append(f"({v3} {v7} {v6} {v2})")
            
            # Left boundary (sides) - missing left neighbor
            if (i-1, j) not in block_set:
                boundary_patches['sides'].append(f"({v0} {v4} {v7} {v3})")
            
            # Right boundary (sides) - missing right neighbor
            if (i+1, j) not in block_set:
                boundary_patches['sides'].append(f"({v1} {v2} {v6} {v5})")
        
        return boundary_patches