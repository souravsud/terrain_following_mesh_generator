import os
import logging
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from .config import MeshConfig
from .utils import create_blockMesh_spacing, generate_region_coordinates, build_roughness_interpolator, load_terrain_points

logger = logging.getLogger(__name__)


class BlockMeshGenerator:
    """Generate OpenFOAM blockMeshDict from structured grid"""

    def generate_blockMeshDict(
        self,
        config: MeshConfig,
        terrain_map: str = "maps/terrain_map.npz",
        output_dict_file: str = "system/blockMeshDict",
        inlet_face_file: str = "0/include/inletFaceInfo.txt",
        roughness_data: np.ndarray = None,
        roughness_transform: object = None,
    ):
        """Wrapper method that uses MeshConfig"""

        return self._blockMeshDictCreator(
            terrain_map=terrain_map,
            output_dict_file=output_dict_file,
            inlet_face_file=inlet_face_file,
            domain_height=config.domain_height,
            z_grading=config.z_grading,
            total_z_cells=config.total_z_cells,
            terrain_normal_first_layer=config.terrain_normal_first_layer,
            patch_types=config.patch_types,
            extract_inlet_face_info=config.extract_inlet_face_info,
            roughness_data=roughness_data,
            roughness_transform=roughness_transform,
        )

    def _blockMeshDictCreator(
        self,
        terrain_map="maps/terrain_map.npz",
        output_dict_file="system/blockMeshDict",
        inlet_face_file="0/include/inletFaceInfo.txt",
        domain_height=4000.0,
        z_grading=None,
        total_z_cells=20,
        patch_types=None,
        extract_inlet_face_info=True,
        terrain_normal_first_layer = False,
        roughness_data=None,
        roughness_transform=None,
    ):
        """
        Generates an OpenFOAM blockMeshDict with flexible z-direction grading.
        """

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
            # Read terrain map (NPZ)
            ny, nx, points = load_terrain_points(terrain_map)
            logger.debug(f"Grid: {nx}x{ny} ({points.shape[0]} points)")

            # Extract coordinates
            x_coords = points[:, :, 0]  # (ny, nx)
            y_coords = points[:, :, 1]  # (ny, nx)
            z_coords = points[:, :, 2]  # (ny, nx)

            # Create validity mask (not NaN)
            valid_mask = ~np.isnan(z_coords)
            nan_count = np.sum(~valid_mask)
            logger.debug(f"NaN points: {nan_count}/{valid_mask.size} ({100*nan_count/valid_mask.size:.1f}%)")

            # Calculate terrain normals
            normals = None
            if terrain_normal_first_layer:
                logger.debug("Calculating terrain normals...")
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
                
                
                logger.debug(f"Vertices: {num_ground_vertices} ground + {num_ground_vertices} first_layer + {num_ground_vertices} sky = {len(valid_vertices)}")
            else:

                # Map valid sky vertices
                for j in range(ny):
                    for i in range(nx):
                        if valid_mask[j, i]:
                            x, y, z = points[j, i]
                            valid_vertices.append((x, y, domain_height))  # Sky vertex
                            vertex_counter += 1
                logger.debug(f"Vertices: {num_ground_vertices} ground + {num_ground_vertices} sky = {len(valid_vertices)}")

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
                
                logger.debug(f"Blocks: {len(valid_blocks_layer1)} first layer + {len(valid_blocks_layer2plus)} upper layer (skipped {(nx-1)*(ny-1) - len(valid_blocks_layer1)} NaN blocks)")
                  
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

                logger.debug(f"Blocks: {len(valid_blocks_layer1)} valid (skipped {(nx-1)*(ny-1) - len(valid_blocks_layer1)} NaN blocks)")

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
                    z_grading,
                    total_z_cells,
                    terrain_normal_first_layer,
                    roughness_data,
                    roughness_transform,
                )
            os.makedirs(os.path.dirname(output_dict_file), exist_ok=True)
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
                f.write("scale 1;\n\n")

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

            logger.debug(f"blockMeshDict written to '{output_dict_file}'")
            if terrain_normal_first_layer:
                total_cells = len(valid_blocks_layer1) + len(valid_blocks_layer2plus) * (total_z_cells - 1)
            else:
                total_cells = len(valid_blocks_layer1) * total_z_cells
            logger.debug(f"Total cells: {total_cells}")

            # Log boundary summary
            sky_faces = len(valid_blocks_layer2plus) if terrain_normal_first_layer else len(valid_blocks_layer1)
            side_summary = ", ".join(
                f"{p}: {len(boundary_patches[p])}" for p in ['inlet', 'outlet', 'sides'] if boundary_patches[p]
            )
            logger.debug(
                f"Patches — ground: {len(valid_blocks_layer1)}, sky: {sky_faces}"
                + (f", {side_summary}" if side_summary else "")
            )
            logger.debug(f"Z-cells: {total_z_cells}, grading: {z_grading}")

        except Exception as e:
            logger.error(f"Failed to generate blockMeshDict: {e}")

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
        z_grading,
        total_z_cells,
        terrain_normal_first_layer,
        roughness_data=None,
        roughness_transform=None,
        default_z0=0.1
    ):
        """
        Saves inlet face information to a file with mesh parameters AND z0 values.
        """
        z0_min = 0.0002 # Minimum z0 to avoid zero or extremely small values that can cause numerical issues in log-law calculations- values corresponds to that of water
        
        logger.debug("Saving inlet face information...")

        # Create the directory for the output file if it doesn't exist.
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        inlet_faces = []
        block_set = set(block_positions.keys())
        
        # Prepare z0 interpolator if roughness data provided
        z0_interpolator = None
        if roughness_data is not None and roughness_transform is not None:
            logger.debug("Preparing z0 interpolation for inlet faces...")
            z0_interpolator = build_roughness_interpolator(
                roughness_data, roughness_transform, default_z0
            )

        # Collect inlet faces (both terrain_normal_first_layer modes share identical face data)
        for (i, j), vertices in block_positions.items():
            if (i, j - 1) not in block_set:
                x_ground = points[j, i, 0]
                y_ground = points[j, i, 1]
                z_ground = points[j, i, 2]
                
                z0_val = float(z0_interpolator([y_ground, x_ground])[0]) if z0_interpolator else default_z0
                z0_val = max(z0_val, z0_min)

                inlet_faces.append({
                    "block_i": i,
                    "block_j": j,
                    "x_ground": x_ground,
                    "y_ground": y_ground,
                    "z_ground": z_ground,
                    "z0": z0_val,
                })

        # Calculate statistics
        avg_inlet_height = sum(face["z_ground"] for face in inlet_faces) / len(inlet_faces)
        if z0_interpolator:
            z0_values = np.array([face["z0"] for face in inlet_faces])
            
            # Avoid zero or extremely small values
            z0_values = z0_values[z0_values > 0]
            
            # Geometric mean (physically correct for log-law)
            z0_eff = float(np.exp(np.mean(np.log(z0_values))))
            
            z0_stats = (
                f"min={z0_values.min():.4f}, "
                f"max={z0_values.max():.4f}, "
                f"arith_mean={np.mean(z0_values):.4f}, "
                f"geo_mean={z0_eff:.4f}"
            )
            
            logger.debug(f"Inlet z0 statistics: {z0_stats}")
        else:
            z0_eff = default_z0
            logger.debug(f"No roughness data provided, using default z0={default_z0}")

        # Save to file
        with open(output_file, "w") as f:
            f.write("# Inlet face information with mesh parameters and z0 roughness\n")
            f.write("# Generated by BlockMeshGenerator\n")
            f.write("#\n")

            # Mesh parameters section
            f.write("# MESH_PARAMETERS_START\n")
            f.write(f"domain_height={domain_height}\n")
            f.write(f"avg_inlet_height={avg_inlet_height:.6f}\n")
            f.write(f"z0_eff_atInlet={z0_eff:.6f}\n")
            f.write("mesh_type=graded\n")
            f.write(f"total_z_cells={total_z_cells}\n")

            # Write z_grading
            f.write("z_grading=")
            grading_str = ";".join(
                [f"{spec[0]},{spec[1]},{spec[2]}" for spec in z_grading]
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

        logger.debug(f"Saved inlet face info: {len(inlet_faces)} blocks to {output_file}")

        return inlet_faces

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
            z_coords = create_blockMesh_spacing(total_z_cells, z_grading)
            first_cell_size = (z_coords[1] - z_coords[0]) * domain_height
            second_cell_size = (z_coords[2] - z_coords[1]) * domain_height
            third_cell_size = (z_coords[3] - z_coords[2]) * domain_height
            logger.debug(f"Cell sizes — first: {first_cell_size:.3f}m, second: {second_cell_size:.3f}m, third: {third_cell_size:.3f}m")
            cell_ratio = first_cell_size / second_cell_size
        else:
            z_coords = create_blockMesh_spacing(total_z_cells + 1, z_grading)
            
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
    
    def generate_z0_field(
        self,
        terrain_map: str,
        roughness_data: np.ndarray,
        roughness_transform: object,
        output_file: str,
        default_z0: float = 0.1
    ):
        """
        Generate OpenFOAM z0 field file from roughness map and terrain map.
        
        Args:
            terrain_map: Path to terrain NPZ map (maps/terrain_map.npz)
            roughness_data: 2D roughness array (with NaN outside rotated crop)
            roughness_transform: Affine transform for roughness grid
            output_file: Output path for z0 field file (e.g. '0/include/z0Values')
            default_z0: Default roughness for points outside roughness coverage (fallback only)
        """
        logger.debug("Generating z0 field for OpenFOAM")
        
        # Read terrain map (NPZ)
        ny, nx, points = load_terrain_points(terrain_map)
        
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
        
        logger.debug(f"Found {n_faces} ground faces")
        logger.debug(f"Face center bounds: X[{ground_face_centers[:, 0].min():.2f}, {ground_face_centers[:, 0].max():.2f}], "
            f"Y[{ground_face_centers[:, 1].min():.2f}, {ground_face_centers[:, 1].max():.2f}]")

        if not np.any(~np.isnan(roughness_data)):
            raise ValueError("No valid roughness data in cropped region")

        logger.debug(f"Roughness grid: {roughness_data.shape[0]}x{roughness_data.shape[1]} ({np.sum(~np.isnan(roughness_data))} valid pixels)")

        # Build shared roughness interpolator (NaN-fill + bilinear)
        interpolator = build_roughness_interpolator(roughness_data, roughness_transform, default_z0)
        
        # Interpolate z0 at ground face centers
        face_z0_values = interpolator(ground_face_centers[:, [1, 0]])  # (y, x) order
        
        logger.debug("Applying minimum roughness of 0.0002 m")
        face_z0_values = np.maximum(face_z0_values, 0.0002)
        # Check if any face centers fell outside roughness coverage
        outside_count = np.sum(face_z0_values == default_z0)
        if outside_count > 0:
            logger.warning(f"{outside_count} face centers outside roughness coverage, using default z0={default_z0}")
        
        # Statistics
        logger.debug(
            f"Z0 — min: {face_z0_values.min():.4f}, max: {face_z0_values.max():.4f}, "
            f"mean: {face_z0_values.mean():.4f}, median: {np.median(face_z0_values):.4f}"
        )
        
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
        
        logger.debug(f"z0 field written to: {output_file} ({n_faces} faces)")
        
        return {
            'n_faces': n_faces,
            'z0_min': float(face_z0_values.min()),
            'z0_max': float(face_z0_values.max()),
            'z0_mean': float(face_z0_values.mean()),
            'output_file': str(output_file)
        }