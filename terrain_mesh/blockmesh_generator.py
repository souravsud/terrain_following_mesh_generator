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
    ):
        """
        Generates an OpenFOAM blockMeshDict with flexible z-direction grading.
        """

        class MockConfig:
            def __init__(self, z_grading, total_z_cells):
                self.z_grading = z_grading
                self.total_z_cells = total_z_cells

        mock_config = MockConfig(z_grading, total_z_cells)

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
        z_grading_spec = self._calculate_z_grading_spec(
            domain_height, z_grading, total_z_cells
        )

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
            print(
                f"Found {nan_count}/{valid_mask.size} NaN points ({100*nan_count/valid_mask.size:.1f}%)"
            )

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

            print(
                f"Created {num_ground_vertices} ground + {num_ground_vertices} sky = {len(valid_vertices)} vertices"
            )

            # Find valid blocks and store positions
            valid_blocks = []
            block_positions = {}  # Store (i,j) -> block_vertices mapping

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

                        block_vertices = (v0, v1, v2, v3, v4, v5, v6, v7)
                        valid_blocks.append(block_vertices)
                        block_positions[(i, j)] = block_vertices

            print(
                f"Created {len(valid_blocks)} valid blocks (skipped {(nx-1)*(ny-1) - len(valid_blocks)} blocks with NaN)"
            )

            # Detect boundary patches by direction
            boundary_patches = self.detect_boundary_patches(block_positions, nx, ny)
            print("inlet file")
            # Detect boundary patches by direction
            boundary_patches = self.detect_boundary_patches(block_positions, nx, ny)
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
                f.write(
                    "/*--------------------------------*- C++ -*----------------------------------*\\\n"
                )
                f.write(
                    "| =========                 |                                                 |\n"
                )
                f.write(
                    "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
                )
                f.write(
                    "|  \\\\    /   O peration     | Version:  v2312                                 |\n"
                )
                f.write(
                    "|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |\n"
                )
                f.write(
                    "|    \\\\/     M anipulation  |                                                 |\n"
                )
                f.write(
                    "\\*---------------------------------------------------------------------------*/\n"
                )
                f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n")
                f.write(
                    "    class       dictionary;\n    object      blockMeshDict;\n}\n"
                )
                f.write(
                    "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
                )
                f.write("convertToMeters 1;\n\n")

                # Vertices (only valid ones)
                f.write("vertices\n(\n")
                for i, (x, y, z) in enumerate(valid_vertices):
                    f.write(f"    ({x:.4f} {y:.4f} {z:.4f}) // v{i}\n")
                f.write(");\n\n")

                # Blocks (only valid ones) with new grading
                f.write("blocks\n(\n")
                for v0, v1, v2, v3, v4, v5, v6, v7 in valid_blocks:
                    f.write(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) ")
                    f.write(f"(1 1 {total_z_cells}) {z_grading_spec}\n")
                f.write(");\n\n")

                # Boundaries with configurable patch types
                f.write("boundary\n(\n")

                # Ground patch
                f.write(
                    f"    ground\n    {{\n        type {patch_types['ground']};\n        faces\n        (\n"
                )
                for v0, v1, v2, v3, v4, v5, v6, v7 in valid_blocks:
                    f.write(f"            ({v0} {v3} {v2} {v1})\n")
                f.write("        );\n    }\n\n")

                # Sky patch
                f.write(
                    f"    sky\n    {{\n        type {patch_types['sky']};\n        faces\n        (\n"
                )
                for v0, v1, v2, v3, v4, v5, v6, v7 in valid_blocks:
                    f.write(f"            ({v4} {v5} {v6} {v7})\n")
                f.write("        );\n    }\n\n")

                # Inlet patch (left boundary)
                if boundary_patches["inlet"]:
                    f.write(
                        f"    inlet\n    {{\n        type {patch_types['inlet']};\n        faces\n        (\n"
                    )
                    for face in boundary_patches["inlet"]:
                        f.write(f"            {face}\n")
                    f.write("        );\n    }\n\n")

                # Outlet patch (right boundary)
                if boundary_patches["outlet"]:
                    f.write(
                        f"    outlet\n    {{\n        type {patch_types['outlet']};\n        faces\n        (\n"
                    )
                    for face in boundary_patches["outlet"]:
                        f.write(f"            {face}\n")
                    f.write("        );\n    }\n\n")

                # Sides patch (front/back boundaries)
                if boundary_patches["sides"]:
                    f.write(
                        f"    sides\n    {{\n        type {patch_types['sides']};\n        faces\n        (\n"
                    )
                    for face in boundary_patches["sides"]:
                        f.write(f"            {face}\n")
                    f.write("        );\n    }\n\n")

                f.write(");\n\n")
                f.write(
                    "// ************************************************************************* //\n"
                )

            print(f"\nSuccessfully generated blockMeshDict at '{output_dict_file}'")
            total_cells = len(valid_blocks) * total_z_cells
            print(f"Total blocks: {len(valid_blocks)}, Total cells: {total_cells}")

            # Print boundary summary
            print(f"\nBoundary patches created:")
            print(f"  ground ({patch_types['ground']}): {len(valid_blocks)} faces")
            print(f"  sky ({patch_types['sky']}): {len(valid_blocks)} faces")
            for patch_name in ["inlet", "outlet", "sides"]:
                if boundary_patches[patch_name]:
                    print(
                        f"  {patch_name} ({patch_types[patch_name]}): {len(boundary_patches[patch_name])} faces"
                    )

            print(f"\nZ-direction configuration:")
            print(f"  Total z-cells: {total_z_cells}")
            print(f"  Z-grading: {z_grading}")
            print(f"  Generated grading spec: {z_grading_spec}")

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
            "inlet": [],  # Front boundary (j-1 neighbor missing) - wind inlet
            "outlet": [],  # Back boundary (j+1 neighbor missing) - wind outlet
            "sides": [],  # Left/right boundaries (i±1 neighbors missing)
        }

        # Create set for fast neighbor lookup
        block_set = set(block_positions.keys())

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
    ):
        """
        Saves inlet face information to a file with mesh parameters, creating the directory if it doesn't exist.

        Args:
            block_positions (dict): A dictionary mapping block indices (i, j) to vertex indices.
            boundary_patches (dict): A dictionary for boundary patches (passed as a placeholder).
            points (list or array): A data structure containing the point coordinates.
            output_file (str): The full path to the output file (e.g., '0/include/inletFaceInfo.txt').
            domain_height (float): Total domain height
            mesh_config: MeshConfig object with z-direction parameters
        """
        print("Saving inlet face information...")

        # Create the directory for the output file if it doesn't exist.
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        inlet_faces = []
        block_set = set(block_positions.keys())

        for (i, j), (v0, v1, v2, v3, v4, v5, v6, v7) in block_positions.items():
            # Check if this block has inlet face
            if (i, j - 1) not in block_set:
                # This block contributes to inlet
                # Get ground coordinates from vertex indices
                x_ground = points[j, i, 0]  # v0 x-coordinate
                y_ground = points[j, i, 1]  # v0 y-coordinate
                z_ground = points[j, i, 2]  # v0 z-coordinate

                inlet_faces.append(
                    {
                        "block_i": i,
                        "block_j": j,
                        "x_ground": x_ground,
                        "y_ground": y_ground,
                        "z_ground": z_ground,
                        "vertices": (v0, v1, v2, v3, v4, v5, v6, v7),
                    }
                )

        # Calculate average inlet height for reference
        avg_inlet_height = sum(face["z_ground"] for face in inlet_faces) / len(
            inlet_faces
        )

        # Save to file with mesh parameters
        with open(output_file, "w") as f:
            f.write("# Inlet face information with mesh parameters\n")
            f.write("# Generated by BlockMeshGenerator\n")
            f.write("#\n")

            # Mesh parameters section
            f.write("# MESH_PARAMETERS_START\n")
            f.write(f"domain_height={domain_height}\n")
            f.write(f"avg_inlet_height={avg_inlet_height:.6f}\n")
            f.write("mesh_type=graded\n")
            f.write(f"total_z_cells={mesh_config.total_z_cells}\n")

            # Write z_grading as parseable format
            f.write("z_grading=")
            grading_str = ";".join(
                [f"{spec[0]},{spec[1]},{spec[2]}" for spec in mesh_config.z_grading]
            )
            f.write(f"{grading_str}\n")

            f.write("# MESH_PARAMETERS_END\n")
            f.write("#\n")

            # Face data section
            f.write("# FACE_DATA_START\n")
            f.write("# Format: block_i, block_j, x_ground, y_ground, z_ground\n")
            f.write(f"# Total inlet blocks: {len(inlet_faces)}\n")

            for face in inlet_faces:
                f.write(f"{face['block_i']}, {face['block_j']}, ")
                f.write(
                    f"{face['x_ground']:.6f}, {face['y_ground']:.6f}, {face['z_ground']:.6f}\n"
                )

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

        if len(z_grading) == 1:
            # Single region - calculate expansion ratio from blockMesh spacing
            z_coords = self.create_blockMesh_spacing(total_z_cells + 1, z_grading)
            expansion_ratio = self._calculate_expansion_ratio_from_coords(z_coords)
            return f"simpleGrading (1 1 {expansion_ratio})"
        else:
            # Multiple regions - use multiGrading
            grading_parts = []
            for length_frac, cell_frac, expansion_ratio in z_grading:
                grading_parts.append(f"({length_frac} {cell_frac} {expansion_ratio})")
            grading_str = " ".join(grading_parts)
            return f"multiGrading (1 1 ({grading_str}))"

    def _calculate_expansion_ratio_from_coords(self, coords):
        """Calculate expansion ratio from coordinate array"""
        if len(coords) < 3:
            return 1.0

        # Calculate cell sizes
        cell_sizes = np.diff(coords)

        if len(cell_sizes) < 2:
            return 1.0

        # Expansion ratio is last_cell / first_cell
        expansion_ratio = cell_sizes[-1] / cell_sizes[0]
        return expansion_ratio
