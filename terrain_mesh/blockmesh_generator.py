import os
import numpy as np
import pyvista as pv
from typing import Dict, List, Tuple
from pathlib import Path

from .config import MeshConfig


class BlockMeshGenerator:
    """Generate OpenFOAM blockMeshDict from structured grid"""

    def generate_blockMeshDict(
        self,
        config: MeshConfig,
        input_vtk_file: str = "terrain_structured.vtk",
        output_dict_file: str = "system/blockMeshDict",
        inlet_face_file: str = "0/include/inletFaceInfo.txt",
    ):
        """Wrapper method that uses MeshConfig"""

        return self._blockMeshDictCreator(
            input_vtk_file=input_vtk_file,
            output_dict_file=output_dict_file,
            inlet_face_file=inlet_face_file,
            domain_height=config.domain_height,
            z_grading=config.z_grading,
            total_z_cells=config.total_z_cells,
            terrain_normal_first_layer=config.terrain_normal_first_layer,
            patch_types=config.patch_types,
            extract_inlet_face_info=config.extract_inlet_face_info,
        )

    def _blockMeshDictCreator(
        self,
        input_vtk_file="terrain_structured.vtk",
        output_dict_file="system/blockMeshDict",
        inlet_face_file="0/include/inletFaceInfo.txt",
        domain_height=4000.0,
        z_grading=None,
        total_z_cells=20,
        patch_types=None,
        extract_inlet_face_info=True,
        terrain_normal_first_layer = False,
    ):
        """
        Generates an OpenFOAM blockMeshDict with flexible z-direction grading.
        """

        class MockConfig:
            def __init__(self, z_grading, total_z_cells, terrain_normal_first_layer):
                self.z_grading = z_grading
                self.total_z_cells = total_z_cells
                self.terrain_normal_first_layer = terrain_normal_first_layer

        mock_config = MockConfig(z_grading, total_z_cells, terrain_normal_first_layer)

        # Set default patch types if not provided
        if patch_types is None:
            patch_types = {
                "ground": "wall",
                "sky": "patch",
                "inlet": "patch",
                "outlet": "patch",
                "sides": "patch",
            }

        # Calculate z-direction grading specification
        z_grading_spec,first_cell_height  = self._calculate_z_grading_spec(
            domain_height, z_grading, total_z_cells,terrain_normal_first_layer
        )

        try:
            # Read VTK file
            mesh = pv.read(input_vtk_file)
            nx, ny, _ = mesh.dimensions
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

            # Calculate terrain normals
            normals = None
            if terrain_normal_first_layer:
                print("Calculating terrain normals...")
                normals = self.calculate_vertex_normals(points, valid_mask, nx, ny)
            
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
            
            if terrain_normal_first_layer:
                # Map first layer top vertices (offset by first_cell_height along normal)
                for j in range(ny):
                    for i in range(nx):
                        if valid_mask[j, i]:
                            x, y, z = points[j, i]
                            normal = normals[j, i]
                            # Offset along normal
                            x_offset = x + normal[0] * first_cell_height
                            y_offset = y + normal[1] * first_cell_height
                            z_offset = z + normal[2] * first_cell_height
                            valid_vertices.append((x_offset, y_offset, z_offset))
                            vertex_counter += 1
                
                num_first_layer_vertices = vertex_counter
                
                # Map valid sky vertices  
                for j in range(ny):
                    for i in range(nx):
                        if valid_mask[j, i]:
                            x, y, z = points[j, i]
                            valid_vertices.append((x, y, domain_height))  # Sky vertex
                            vertex_counter += 1
                
                
                print(f"Created {num_ground_vertices} ground + {num_ground_vertices} first_layer + {num_ground_vertices} sky = {len(valid_vertices)} vertices")
            else:

                # Map valid sky vertices
                for j in range(ny):
                    for i in range(nx):
                        if valid_mask[j, i]:
                            x, y, z = points[j, i]
                            valid_vertices.append((x, y, domain_height))  # Sky vertex
                            vertex_counter += 1
                print("valid vertices:", len(valid_vertices))
                print(f"Created {num_ground_vertices} ground + {num_ground_vertices} sky = {len(valid_vertices)} vertices")

            # Find valid blocks and store positions
            valid_blocks_layer1 = []
            valid_blocks_layer2plus = []
            block_positions = {}  # Store (i,j) -> block_vertices mapping

            if terrain_normal_first_layer:
                
                for j in range(ny - 1):
                    for i in range(nx - 1):
                        corners_valid = (valid_mask[j, i] and valid_mask[j, i+1] and 
                                    valid_mask[j+1, i+1] and valid_mask[j+1, i])
                        
                        if corners_valid:
                            # Ground layer vertex indices
                            v0 = vertex_map[j, i]
                            v1 = vertex_map[j, i+1] 
                            v2 = vertex_map[j+1, i+1]
                            v3 = vertex_map[j+1, i]
                            
                            # First layer top vertex indices
                            v4 = v0 + num_ground_vertices
                            v5 = v1 + num_ground_vertices
                            v6 = v2 + num_ground_vertices
                            v7 = v3 + num_ground_vertices
                            
                            # First layer block (ground to first_layer_top)
                            block_layer1 = (v0, v1, v2, v3, v4, v5, v6, v7, 1)
                            valid_blocks_layer1.append(block_layer1)
                            
                            # Sky vertex indices
                            v8 = v0 + 2 * num_ground_vertices
                            v9 = v1 + 2 * num_ground_vertices
                            v10 = v2 + 2 * num_ground_vertices
                            v11 = v3 + 2 * num_ground_vertices
                            
                            # Remaining layers block (first_layer_top to sky)
                            num_cells_remaining = total_z_cells - 1
                            block_layer2plus = (v4, v5, v6, v7, v8, v9, v10, v11, num_cells_remaining)
                            valid_blocks_layer2plus.append(block_layer2plus)
                            
                            block_positions[(i, j)] = (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)
                
                print(f"Created {len(valid_blocks_layer1)} first layer blocks + {len(valid_blocks_layer2plus)} upper layer blocks")
                print(f"Skipped {(nx-1)*(ny-1) - len(valid_blocks_layer1)} blocks with NaN")
                  
            else:
                for j in range(ny - 1):
                    for i in range(nx - 1):
                        # Check if all 4 corners are valid
                        corners_valid = (
                            valid_mask[j, i]
                            and valid_mask[j, i + 1]
                            and valid_mask[j + 1, i + 1]
                            and valid_mask[j + 1, i]
                        )

                        if corners_valid:
                            # Get vertex indices
                            v0 = vertex_map[j, i]
                            v1 = vertex_map[j, i + 1]
                            v2 = vertex_map[j + 1, i + 1]
                            v3 = vertex_map[j + 1, i]
                            v4 = v0 + num_ground_vertices  # Sky vertices
                            v5 = v1 + num_ground_vertices
                            v6 = v2 + num_ground_vertices
                            v7 = v3 + num_ground_vertices

                            block = (v0, v1, v2, v3, v4, v5, v6, v7, total_z_cells)
                            valid_blocks_layer1.append(block)
                            block_positions[(i, j)] = (v0, v1, v2, v3, v4, v5, v6, v7)

                print(f"Created {len(valid_blocks_layer1)} valid blocks (skipped {(nx-1)*(ny-1) - len(valid_blocks_layer1)} blocks with NaN)")

            # Detect boundary patches
            boundary_patches = self.detect_boundary_patches(
                block_positions, nx, ny, num_ground_vertices, terrain_normal_first_layer
            )

            if extract_inlet_face_info:
                inlet_face_info = self.save_inlet_face_info(
                    block_positions,
                    boundary_patches,
                    points,
                    inlet_face_file,
                    domain_height,
                    mock_config,
                )

            # Write blockMeshDict
            with open(output_dict_file, "w") as f:
                # Header
                f.write("/*--------------------------------*- C++ -*------------------------------------*\\\n")
                f.write("| ===========                 |                                                 |\n")
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

                # Blocks (only valid ones) with new grading
                f.write("blocks\n(\n")
                if terrain_normal_first_layer:
                    # First layer blocks (uniform, 1 cell)
                    for v0, v1, v2, v3, v4, v5, v6, v7, nz in valid_blocks_layer1:
                        f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                        f.write(f"(1 1 1) simpleGrading (1 1 1)\n")
                    
                    # Upper layer blocks (with expansion ratio)
                    for v0, v1, v2, v3, v4, v5, v6, v7, nz in valid_blocks_layer2plus:
                        f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                        f.write(f"(1 1 {nz}) {z_grading_spec}\n")
                else:
                    for v0, v1, v2, v3, v4, v5, v6, v7, nz in valid_blocks_layer1:
                        f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                        f.write(f"(1 1 {nz}) {z_grading_spec}\n")
                f.write(");\n\n")

                # Boundaries with configurable patch types
                f.write("boundary\n(\n")

                # Ground patch
                f.write(f"    ground\n    {{\n        type {patch_types['ground']};\n        faces\n        (\n")
                for v0, v1, v2, v3, v4, v5, v6, v7, nz in valid_blocks_layer1:
                    f.write(f"            ({v0} {v3} {v2} {v1})\n")
                f.write("        );\n    }\n\n")

                # Sky patch
                f.write(f"    sky\n    {{\n        type {patch_types['sky']};\n        faces\n        (\n")
                if terrain_normal_first_layer:
                    for v0, v1, v2, v3, v4, v5, v6, v7, nz in valid_blocks_layer2plus:
                        f.write(f"            ({v4} {v5} {v6} {v7})\n")
                else:
                    for v0, v1, v2, v3, v4, v5, v6, v7, nz in valid_blocks_layer1:
                        f.write(f"            ({v4} {v5} {v6} {v7})\n")
                f.write("        );\n    }\n\n")

                # Inlet, outlet, sides patches
                for patch_name in ['inlet', 'outlet', 'sides']:
                    if boundary_patches[patch_name]:
                        f.write(f"    {patch_name}\n    {{\n        type {patch_types[patch_name]};\n        faces\n        (\n")
                        for face in boundary_patches[patch_name]:
                            f.write(f"            {face}\n")
                        f.write("        );\n    }\n\n")

                f.write(");\n\n")
                f.write("// ************************************************************************* //\n")

            print(f"\nSuccessfully generated blockMeshDict at '{output_dict_file}'")
            if terrain_normal_first_layer:
                total_cells = len(valid_blocks_layer1) + len(valid_blocks_layer2plus) * (total_z_cells - 1)
                print(f"Total cells: {total_cells} (first layer: {len(valid_blocks_layer1)}, upper layers: {len(valid_blocks_layer2plus) * (total_z_cells - 1)})")
            else:
                total_cells = len(valid_blocks_layer1) * total_z_cells
                print(f"Total cells: {total_cells}")

            # Print boundary summary
            print(f"\nBoundary patches created:")
            print(f"  ground ({patch_types['ground']}): {len(valid_blocks_layer1)} faces")
            if terrain_normal_first_layer:
                print(f"  sky ({patch_types['sky']}): {len(valid_blocks_layer2plus)} faces")
            else:
                print(f"  sky ({patch_types['sky']}): {len(valid_blocks_layer1)} faces")
            for patch_name in ['inlet', 'outlet', 'sides']:
                if boundary_patches[patch_name]:
                    print(f"  {patch_name} ({patch_types[patch_name]}): {len(boundary_patches[patch_name])} faces")

            print(f"\nZ-direction configuration:")
            print(f"  Total z-cells: {total_z_cells}")
            print(f"  Z-grading: {z_grading}")
            print(f"  Generated grading spec: {z_grading_spec}")

        except Exception as e:
            print(f"Error: {e}")

    def calculate_vertex_normals(self, points, valid_mask, nx, ny):
        """
        Calculate terrain normals at each valid vertex using neighboring cells.

        Args:
            points: array of shape (ny, nx, 3) with terrain coordinates
            valid_mask: boolean array of shape (ny, nx) indicating valid points
            nx, ny: grid dimensions

        Returns:
            normals: array of shape (ny, nx, 3) with unit normals (NaN for invalid points)
        """
        normals = np.full((ny, nx, 3), np.nan)

        for j in range(ny):
            for i in range(nx):
                if not valid_mask[j, i]:
                    continue

                # Collect valid neighboring cell normals
                cell_normals = []

                # Check 4 cells around this vertex (if they exist and are valid)
                cells = [
                    (i - 1, j - 1),  # bottom-left cell
                    (i, j - 1),  # bottom-right cell
                    (i - 1, j),  # top-left cell
                    (i, j),  # top-right cell
                ]

                for ci, cj in cells:
                    # Check if cell exists and all 4 corners are valid
                    if (
                        0 <= ci < nx - 1
                        and 0 <= cj < ny - 1
                        and valid_mask[cj, ci]
                        and valid_mask[cj, ci + 1]
                        and valid_mask[cj + 1, ci + 1]
                        and valid_mask[cj + 1, ci]
                    ):

                        # Get cell corner points
                        p0 = points[cj, ci]
                        p1 = points[cj, ci + 1]
                        p2 = points[cj + 1, ci + 1]
                        p3 = points[cj + 1, ci]

                        # Calculate normal from cross product of diagonals
                        diag1 = p2 - p0
                        diag2 = p3 - p1
                        normal = np.cross(diag1, diag2)

                        # Normalize
                        norm_length = np.linalg.norm(normal)
                        if norm_length > 1e-10:
                            normal = normal / norm_length
                            # Ensure upward pointing (positive z component)
                            if normal[2] < 0:
                                normal = -normal
                            cell_normals.append(normal)

                # Average the normals from surrounding cells
                if cell_normals:
                    avg_normal = np.mean(cell_normals, axis=0)
                    norm_length = np.linalg.norm(avg_normal)
                    if norm_length > 1e-10:
                        normals[j, i] = avg_normal / norm_length

        return normals

    def detect_boundary_patches(self, block_positions, nx, ny, num_ground_vertices, terrain_normal_first_layer):
        """
        Detect directional boundary patches based on grid position.

        Args:
            block_positions: dict mapping (i,j) -> block_vertices
            nx, ny: grid dimensions

        Returns:
            dict: {'inlet': [faces], 'outlet': [faces], 'sides': [faces]}
        """
        boundary_patches = {
            "inlet": [],  # Front boundary (j-1 neighbor missing) - wind inlet
            "outlet": [],  # Back boundary (j+1 neighbor missing) - wind outlet
            "sides": [],  # Left/right boundaries (i±1 neighbors missing)
        }

        # Create set for fast neighbor lookup
        block_set = set(block_positions.keys())

        if terrain_normal_first_layer:
            for (i, j), vertices in block_positions.items():
                v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11 = vertices
                
                # Front boundary (inlet) - 2 faces stacked
                if (i, j-1) not in block_set:
                    boundary_patches['inlet'].append(f"({v0} {v1} {v5} {v4})")  # Layer 1
                    boundary_patches['inlet'].append(f"({v4} {v5} {v9} {v8})")  # Layer 2+
                
                # Back boundary (outlet)
                if (i, j+1) not in block_set:
                    boundary_patches['outlet'].append(f"({v3} {v7} {v6} {v2})")  # Layer 1
                    boundary_patches['outlet'].append(f"({v7} {v11} {v10} {v6})")  # Layer 2+
                
                # Left boundary (sides)
                if (i-1, j) not in block_set:
                    boundary_patches['sides'].append(f"({v0} {v4} {v7} {v3})")  # Layer 1
                    boundary_patches['sides'].append(f"({v4} {v8} {v11} {v7})")  # Layer 2+
                
                # Right boundary (sides)
                if (i+1, j) not in block_set:
                    boundary_patches['sides'].append(f"({v1} {v2} {v6} {v5})")  # Layer 1
                    boundary_patches['sides'].append(f"({v5} {v6} {v10} {v9})")  # Layer 2+
        else:
            for (i, j), (v0, v1, v2, v3, v4, v5, v6, v7) in block_positions.items():
                # Front boundary (inlet) - missing front neighbor
                if (i, j - 1) not in block_set:
                    boundary_patches["inlet"].append(f"({v0} {v1} {v5} {v4})")

                # Back boundary (outlet) - missing back neighbor
                if (i, j + 1) not in block_set:
                    boundary_patches["outlet"].append(f"({v3} {v7} {v6} {v2})")

                # Left boundary (sides) - missing left neighbor
                if (i - 1, j) not in block_set:
                    boundary_patches["sides"].append(f"({v0} {v4} {v7} {v3})")

                # Right boundary (sides) - missing right neighbor
                if (i + 1, j) not in block_set:
                    boundary_patches["sides"].append(f"({v1} {v2} {v6} {v5})")

        return boundary_patches

    def save_inlet_face_info(
        self,
        block_positions,
        boundary_patches,
        points,
        output_file,
        domain_height,
        mesh_config,
        roughness_data=None,
        roughness_transform=None,
        default_z0=0.1
    ):
        """
        Saves inlet face information to a file with mesh parameters AND z0 values.
        """
        from scipy.interpolate import RegularGridInterpolator
        
        print("Saving inlet face information...")

        # Create the directory for the output file if it doesn't exist.
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        inlet_faces = []
        block_set = set(block_positions.keys())
        
        # Prepare z0 interpolator if roughness data provided
        z0_interpolator = None
        if roughness_data is not None and roughness_transform is not None:
            print("Preparing z0 interpolation for inlet faces...")
            nrows, ncols = roughness_data.shape
            x_min = roughness_transform.c
            y_max = roughness_transform.f
            x_res = roughness_transform.a
            y_res = -roughness_transform.e
            
            x_coords = np.arange(ncols) * x_res + x_min
            y_coords = np.arange(nrows) * (-y_res) + y_max
            
            # Fill NaN with nearest neighbor
            roughness_filled = roughness_data.copy()
            if np.any(np.isnan(roughness_filled)):
                from scipy.ndimage import distance_transform_edt
                invalid_mask = np.isnan(roughness_filled)
                indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
                roughness_filled[invalid_mask] = roughness_data[tuple(indices[:, invalid_mask])]
            
            z0_interpolator = RegularGridInterpolator(
                (y_coords, x_coords),
                roughness_filled,
                method='linear',
                bounds_error=False,
                fill_value=default_z0
            )

        # Collect inlet faces
        if mesh_config.terrain_normal_first_layer:
            for (i, j), (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) in block_positions.items():
                if (i, j - 1) not in block_set:
                    x_ground = points[j, i, 0]
                    y_ground = points[j, i, 1]
                    z_ground = points[j, i, 2]
                    
                    # Interpolate z0 at this location
                    if z0_interpolator:
                        z0_val = float(z0_interpolator([y_ground, x_ground])[0])
                    else:
                        z0_val = default_z0
                    
                    inlet_faces.append({
                        "block_i": i,
                        "block_j": j,
                        "x_ground": x_ground,
                        "y_ground": y_ground,
                        "z_ground": z_ground,
                        "z0": z0_val,
                        "vertices": (v0, v1, v2, v3, v8, v9, v10, v11),
                    })
        else:
            for (i, j), (v0, v1, v2, v3, v4, v5, v6, v7) in block_positions.items():
                if (i, j - 1) not in block_set:
                    x_ground = points[j, i, 0]
                    y_ground = points[j, i, 1]
                    z_ground = points[j, i, 2]
                    
                    # Interpolate z0 at this location
                    if z0_interpolator:
                        z0_val = float(z0_interpolator([y_ground, x_ground])[0])
                    else:
                        z0_val = default_z0
                    
                    inlet_faces.append({
                        "block_i": i,
                        "block_j": j,
                        "x_ground": x_ground,
                        "y_ground": y_ground,
                        "z_ground": z_ground,
                        "z0": z0_val,
                        "vertices": (v0, v1, v2, v3, v4, v5, v6, v7),
                    })

        # Calculate statistics
        avg_inlet_height = sum(face["z_ground"] for face in inlet_faces) / len(inlet_faces)
        if z0_interpolator:
            z0_values = [face["z0"] for face in inlet_faces]
            z0_stats = f"min={min(z0_values):.4f}, max={max(z0_values):.4f}, mean={np.mean(z0_values):.4f}"
            print(f"Inlet z0 statistics: {z0_stats}")

        # Save to file
        with open(output_file, "w") as f:
            f.write("# Inlet face information with mesh parameters and z0 roughness\n")
            f.write("# Generated by BlockMeshGenerator\n")
            f.write("#\n")

            # Mesh parameters section
            f.write("# MESH_PARAMETERS_START\n")
            f.write(f"domain_height={domain_height}\n")
            f.write(f"avg_inlet_height={avg_inlet_height:.6f}\n")
            f.write("mesh_type=graded\n")
            f.write(f"total_z_cells={mesh_config.total_z_cells}\n")

            # Write z_grading
            f.write("z_grading=")
            grading_str = ";".join(
                [f"{spec[0]},{spec[1]},{spec[2]}" for spec in mesh_config.z_grading]
            )
            f.write(f"{grading_str}\n")

            f.write("# MESH_PARAMETERS_END\n")
            f.write("#\n")

            # Face data section
            f.write("# FACE_DATA_START\n")
            f.write("# Format: block_i, block_j, x_ground, y_ground, z_ground, z0\n")
            f.write(f"# Total inlet blocks: {len(inlet_faces)}\n")
            if z0_interpolator:
                f.write(f"# Z0 statistics: {z0_stats}\n")

            for face in inlet_faces:
                f.write(f"{face['block_i']}, {face['block_j']}, ")
                f.write(f"{face['x_ground']:.6f}, {face['y_ground']:.6f}, {face['z_ground']:.6f}, ")
                f.write(f"{face['z0']:.6f}\n")

            f.write("# FACE_DATA_END\n")

        print(f"Saved inlet face info: {len(inlet_faces)} blocks to {output_file}")

        return inlet_faces

    def create_blockMesh_spacing(self, n_points, grading_spec):
        """
        Create variable spacing coordinates from 0 to 1 using blockMesh-style grading.

        Parameters:
        -----------
        n_points : int
            Total number of points
        grading_spec : list of tuples
            [(length_fraction, cell_fraction, expansion_ratio), ...]
            - length_fraction: fraction of domain length for this region
            - cell_fraction: fraction of total cells for this region
            - expansion_ratio: last_cell_size/first_cell_size in this region

        Returns:
        --------
        np.ndarray
            Coordinate array from 0 to 1 with blockMesh-style spacing
        """

        total_cells = n_points - 1
        n_regions = len(grading_spec)
        print(f"Creating blockMesh spacing with {n_regions} regions, total cells: {total_cells}")

        # Extract specifications
        length_fractions = np.array([spec[0] for spec in grading_spec])
        cell_fractions = np.array([spec[1] for spec in grading_spec])
        expansion_ratios = np.array([spec[2] for spec in grading_spec])

        # Validate inputs
        if abs(length_fractions.sum() - 1.0) > 1e-6:
            raise ValueError(
                f"Length fractions sum to {length_fractions.sum():.6f}, must sum to 1.0"
            )

        if abs(cell_fractions.sum() - 1.0) > 1e-6:
            raise ValueError(
                f"Cell fractions sum to {cell_fractions.sum():.6f}, must sum to 1.0"
            )

        # Calculate target cell counts (may not be integers)
        target_cells = cell_fractions * total_cells

        # Round to integers and adjust to maintain total
        actual_cells = np.round(target_cells).astype(int)

        # Adjust for rounding errors
        cell_diff = total_cells - actual_cells.sum()
        if cell_diff != 0:
            # Add/subtract cells from regions with largest rounding errors
            errors = target_cells - actual_cells
            if cell_diff > 0:
                # Need to add cells - add to regions with most positive error
                indices = np.argsort(errors)[::-1]
            else:
                # Need to remove cells - remove from regions with most negative error
                indices = np.argsort(errors)

            for i in range(abs(cell_diff)):
                actual_cells[indices[i]] += np.sign(cell_diff)

        # Generate coordinates for each region
        coords = [0.0]  # Start at 0
        current_pos = 0.0

        for i, (length_frac, actual_cell_count, expansion_ratio) in enumerate(
            zip(length_fractions, actual_cells, expansion_ratios)
        ):
            region_length = length_frac

            if actual_cell_count == 0:
                continue

            # Generate spacing within this region
            region_coords = self.generate_region_coordinates(
                actual_cell_count, expansion_ratio
            )

            # Scale to region length and add to current position
            region_coords_scaled = region_coords * region_length + current_pos

            # Add coordinates (skip the first one as it's already included)
            coords.extend(region_coords_scaled[1:])

            current_pos += region_length

        return np.array(coords)

    def generate_region_coordinates(self, n_cells, expansion_ratio):
        """
        Generate coordinates within a single region [0,1] with given expansion ratio.

        Parameters:
        -----------
        n_cells : int
            Number of cells in this region
        expansion_ratio : float
            Ratio of last_cell_size/first_cell_size

        Returns:
        --------
        np.ndarray
            Coordinates from 0 to 1 for this region
        """

        if n_cells == 0:
            return np.array([0.0, 1.0])

        if n_cells == 1:
            return np.array([0.0, 1.0])

        # For uniform spacing (expansion_ratio ≈ 1)
        if abs(expansion_ratio - 1.0) < 1e-6:
            return np.linspace(0.0, 1.0, n_cells + 1)

        # For geometric progression
        r = expansion_ratio ** (
            1.0 / (n_cells - 1)
        )  # Common ratio between adjacent cells

        # Calculate first cell size
        if abs(r - 1.0) < 1e-6:
            ds = 1.0 / n_cells
        else:
            ds = (r - 1.0) / (r**n_cells - 1.0)

        # Generate cell sizes
        cell_sizes = ds * r ** np.arange(n_cells)

        # Generate coordinates
        coords = np.zeros(n_cells + 1)
        coords[1:] = np.cumsum(cell_sizes)

        return coords

    def _calculate_z_grading_spec(
        self,
        domain_height: float,
        z_grading: List[Tuple[float, float, float]],
        total_z_cells: int,
        terrain_normal_first_layer: bool,
    ) -> str:
        """
        Calculate the z-direction grading specification string for OpenFOAM blockMeshDict.

        Args:
            domain_height: Total domain height
            z_grading: BlockMesh-style grading specification
            total_z_cells: Total number of cells in z-direction

        Returns:
            str: OpenFOAM grading specification (e.g., "simpleGrading (1 1 20)" or multiGrading spec)
        """
        first_cell_size = 0
        if terrain_normal_first_layer:
            # Only need first 3 points for cell size calculation
            z_coords = self.create_blockMesh_spacing(total_z_cells, z_grading)
            first_cell_size = (z_coords[1] - z_coords[0]) * domain_height
            second_cell_size = (z_coords[2] - z_coords[1]) * domain_height
            third_cell_size = (z_coords[3] - z_coords[2]) * domain_height
            print(f"First cell size: {first_cell_size}, Second cell size: {second_cell_size}, Third cell size: {third_cell_size}")
            cell_ratio = first_cell_size / second_cell_size
        else:
            z_coords = self.create_blockMesh_spacing(total_z_cells + 1, z_grading)
            
        if len(z_grading) == 1:
            # Single region - calculate expansion ratio from blockMesh spacing
            expansion_ratio = z_grading[0][2]
            if terrain_normal_first_layer:
                expansion_ratio = cell_ratio*expansion_ratio
                
            return f"simpleGrading (1 1 {expansion_ratio})", first_cell_size

        else:
            grading_parts = []
            first_region = True
            for length_frac, cell_frac, expansion_ratio in z_grading:
                if terrain_normal_first_layer:
                    region_height = length_frac * domain_height
                    region_cell_count = cell_frac * total_z_cells
                    if first_region:
                        length_frac = (region_height - first_cell_size)/(domain_height - first_cell_size)
                        cell_frac = (region_cell_count - 1)/ (total_z_cells -1)
                        expansion_ratio = expansion_ratio * cell_ratio
                        first_region = False
                    else:
                        length_frac = (region_height)/(domain_height - first_cell_size)
                        cell_frac = (region_cell_count)/ (total_z_cells -1)
                        
                grading_parts.append(f"({length_frac} {cell_frac} {expansion_ratio})")
            grading_str = " ".join(grading_parts)
            return f"multiGrading (1 1 ({grading_str}))", first_cell_size

    """ def _calculate_expansion_ratio_from_coords(self, coords):

        if len(coords) < 3:
            return 1.0

        # Calculate cell sizes
        cell_sizes = np.diff(coords)

        if len(cell_sizes) < 2:
            return 1.0

        # Expansion ratio is last_cell / first_cell
        expansion_ratio = cell_sizes[-1] / cell_sizes[0]
        return expansion_ratio """
    
    def generate_z0_field(
        self,
        vtk_file: str,
        roughness_data: np.ndarray,
        roughness_transform: object,
        output_file: str,
        default_z0: float = 0.1
    ):
        """
        Generate OpenFOAM z0 field file from roughness map and VTK mesh.
        
        Args:
            vtk_file: Path to VTK terrain file
            roughness_data: 2D roughness array (with NaN outside rotated crop)
            roughness_transform: Affine transform for roughness grid
            output_file: Output path for z0 field file (e.g. '0/include/z0Values')
            default_z0: Default roughness for points outside roughness coverage (fallback only)
        """
        from scipy.interpolate import RegularGridInterpolator
        
        print("\n" + "="*60)
        print("Generating z0 field for OpenFOAM")
        print("="*60)
        
        # Read VTK mesh
        mesh = pv.read(vtk_file)
        nx, ny, _ = mesh.dimensions
        points = mesh.points.reshape((ny, nx, 3))
        
        # Extract ground face centers (same logic as blockMeshDict generation)
        z_coords = points[:, :, 2]
        valid_mask = ~np.isnan(z_coords)
        
        ground_face_centers = []
        
        # Iterate through cells (same order as blockMeshDict ground faces)
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Check if all 4 corners are valid (same as blockMeshDict logic)
                corners_valid = (
                    valid_mask[j, i] and valid_mask[j, i+1] and 
                    valid_mask[j+1, i+1] and valid_mask[j+1, i]
                )
                
                if corners_valid:
                    # Get 4 corner points of ground face
                    p0 = points[j, i, :2]       # (x, y) only
                    p1 = points[j, i+1, :2]
                    p2 = points[j+1, i+1, :2]
                    p3 = points[j+1, i, :2]
                    
                    # Calculate face center
                    face_center = (p0 + p1 + p2 + p3) / 4.0
                    ground_face_centers.append(face_center)
        
        ground_face_centers = np.array(ground_face_centers)
        n_faces = len(ground_face_centers)
        
        print(f"Found {n_faces} ground faces")
        print(f"Ground face center bounds: X[{ground_face_centers[:, 0].min():.2f}, {ground_face_centers[:, 0].max():.2f}], "
            f"Y[{ground_face_centers[:, 1].min():.2f}, {ground_face_centers[:, 1].max():.2f}]")
        
        # Prepare roughness grid for interpolation
        nrows, ncols = roughness_data.shape
        
        # Get coordinate arrays from transform
        x_min = roughness_transform.c
        y_max = roughness_transform.f
        x_res = roughness_transform.a
        y_res = -roughness_transform.e  # Usually negative
        
        x_coords_rough = np.arange(ncols) * x_res + x_min
        y_coords_rough = np.arange(nrows) * (-y_res) + y_max  # Descending
        
        print(f"Roughness grid: {nrows}x{ncols}")
        print(f"Roughness bounds: X[{x_coords_rough[0]:.2f}, {x_coords_rough[-1]:.2f}], "
            f"Y[{y_coords_rough[-1]:.2f}, {y_coords_rough[0]:.2f}]")
        print(f"Valid roughness pixels: {np.sum(~np.isnan(roughness_data))} / {roughness_data.size}")
        
        # For interpolation, we need to handle NaN values
        # Option 1: Use nearest-neighbor to fill NaN gaps first
        valid_mask_rough = ~np.isnan(roughness_data)
        
        if not np.any(valid_mask_rough):
            raise ValueError("No valid roughness data in cropped region")
        
        # Fill NaN using nearest valid value (for interpolation continuity)
        # This is better than using default_z0 everywhere
        from scipy.ndimage import distance_transform_edt
        
        roughness_data_filled = roughness_data.copy()
        
        # Find nearest valid pixel for each NaN pixel
        invalid_mask = np.isnan(roughness_data)
        if np.any(invalid_mask):
            # Distance transform to find nearest valid pixel
            indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
            roughness_data_filled[invalid_mask] = roughness_data[tuple(indices[:, invalid_mask])]
        
        print(f"Filled {np.sum(invalid_mask)} NaN pixels using nearest neighbor propagation")
        
        # Create interpolator (bilinear)
        interpolator = RegularGridInterpolator(
            (y_coords_rough, x_coords_rough),  # Note: (y, x) order for (rows, cols)
            roughness_data_filled,
            method='linear',
            bounds_error=False,
            fill_value=default_z0  # Fallback for points outside grid (shouldn't happen)
        )
        
        # Interpolate z0 at ground face centers
        face_z0_values = interpolator(ground_face_centers[:, [1, 0]])  # (y, x) order
        
        print("Setting minimum roughness to 0.0002 m to avoid zero values")
        face_z0_values = np.maximum(face_z0_values, 0.0002)
        # Check if any face centers fell outside roughness coverage
        outside_count = np.sum(face_z0_values == default_z0)
        if outside_count > 0:
            print(f"WARNING: {outside_count} face centers fell outside roughness coverage, using default z0={default_z0}")
        
        # Statistics
        print(f"\nZ0 statistics:")
        print(f"  Min: {face_z0_values.min():.4f}")
        print(f"  Max: {face_z0_values.max():.4f}")
        print(f"  Mean: {face_z0_values.mean():.4f}")
        print(f"  Median: {np.median(face_z0_values):.4f}")
        
        # Create output directory
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write OpenFOAM format
        with open(output_file, 'w') as f:
            f.write("// Surface roughness length z0 for ground patch\n")
            f.write("// Generated from roughness map\n")
            f.write("// Format: nonuniform List<scalar>\n")
            f.write("// Include in 0/nut using: z0 #include \"include/z0Values\";\n")
            f.write("//\n")
            f.write(f"nonuniform List<scalar>\n")
            f.write(f"{n_faces}\n")
            f.write("(\n")
            
            # Write values (one per line for readability)
            for z0_val in face_z0_values:
                f.write(f"    {z0_val:.6f}\n")
            
            f.write(")\n")
        
        print(f"\nSuccessfully wrote z0 field to: {output_file}")
        print(f"Total faces: {n_faces}")
        print("="*60)
        
        return {
        'n_faces': n_faces,
        'z0_min': float(face_z0_values.min()),
        'z0_max': float(face_z0_values.max()),
        'z0_mean': float(face_z0_values.mean()),
        'output_file': str(output_file)
    }