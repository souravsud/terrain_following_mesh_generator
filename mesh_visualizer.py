"""
Visualize terrain mesh structure from config before actual meshing.
Shows grid spacing and grading in X, Y, and Z directions on a flat plane.

Usage:
    python mesh_visualizer.py terrain_config.yaml --ground-height 250 --output output_dir
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path
from typing import List, Tuple


def create_blockMesh_spacing(n_points, grading_spec):
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
        region_coords = generate_region_coordinates(
            actual_cell_count, expansion_ratio
        )

        # Scale to region length and add to current position
        region_coords_scaled = region_coords * region_length + current_pos

        # Add coordinates (skip the first one as it's already included)
        coords.extend(region_coords_scaled[1:])

        current_pos += region_length

    return np.array(coords)

def generate_region_coordinates(n_cells, expansion_ratio):
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


def calculate_z_coordinates(
    domain_height: float,
    ground_height: float,
    z_grading: List[Tuple[float, float, float]],
    total_z_cells: int,
) -> np.ndarray:
    
    available_height = domain_height - ground_height
    z_norm = create_blockMesh_spacing(total_z_cells + 1, z_grading)
    return ground_height + z_norm * available_height

def load_config(config_path: str):
    """Load and parse YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert grading lists to tuples
    if "x_grading" in config["grid"]:
        config["grid"]["x_grading"] = [
            tuple(spec) for spec in config["grid"]["x_grading"]
        ]
    if "y_grading" in config["grid"]:
        config["grid"]["y_grading"] = [
            tuple(spec) for spec in config["grid"]["y_grading"]
        ]
    if "z_grading" in config["mesh"]:
        config["mesh"]["z_grading"] = [
            tuple(spec) for spec in config["mesh"]["z_grading"]
        ]

    return config


def visualize_mesh(config_path: str, ground_height: float, output_dir: str = None):
    """Generate mesh visualization plots."""

    # Load config
    config = load_config(config_path)
    terrain = config["terrain"]
    grid = config["grid"]
    mesh = config["mesh"]

    logY = False
    fillPlot = True

    # Extract parameters
    domain_size = terrain["crop_size_km"] * 1000  # Convert to meters
    nx, ny = grid["nx"], grid["ny"]
    domain_height = mesh["domain_height"]

    # Handle z-direction configuration
    if "total_z_cells" in mesh:
        total_z_cells = mesh["total_z_cells"]
        z_grading = mesh["z_grading"]

    # Generate coordinates
    print("Generating mesh coordinates...")

    # X direction
    if grid.get("x_grading"):
        x_norm = create_blockMesh_spacing(nx, grid["x_grading"])
        x_coords = x_norm * domain_size
    else:
        x_coords = np.linspace(0, domain_size, nx)

    # Y direction
    if grid.get("y_grading"):
        y_norm = create_blockMesh_spacing(ny, grid["y_grading"])
        y_coords = y_norm * domain_size
    else:
        y_coords = np.linspace(0, domain_size, ny)

    # Z direction
    z_coords = calculate_z_coordinates(
        domain_height, ground_height, z_grading, total_z_cells
    )

    # Create meshgrids for slices
    X, Y = np.meshgrid(x_coords, y_coords)
    X_xz, Z_xz = np.meshgrid(x_coords, z_coords)

    # Calculate cell sizes
    x_cells = np.diff(x_coords)
    y_cells = np.diff(y_coords)
    z_cells = np.diff(z_coords)

    # Print statistics
    print(f"\nMesh Statistics:")
    print(
        f"  Domain: {domain_size/1000:.1f} x {domain_size/1000:.1f} x {domain_height:.0f} m"
    )
    print(f"  Ground height: {ground_height:.0f} m")
    print(f"  Grid: {nx} x {ny} x {total_z_cells}")
    print(
        f"  X cells: min={x_cells.min():.2f}m, max={x_cells.max():.2f}m, ratio={x_cells.max()/x_cells.min():.2f}"
    )
    print(
        f"  Y cells: min={y_cells.min():.2f}m, max={y_cells.max():.2f}m, ratio={y_cells.max()/y_cells.min():.2f}"
    )
    print(
        f"  Z cells: min={z_cells.min():.2f}m, max={z_cells.max():.2f}m, ratio={z_cells.max()/z_cells.min():.2f}"
    )
    print(f"  Total cell count = {nx*ny*total_z_cells/1000000:.1f}M cells")

    # Create Figure 1: Grid slices (XY and XZ)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # XY slice (plan view)
    ax1.set_aspect("equal")
    for i in range(ny):
        ax1.plot(X[i, :], Y[i, :], "b-", linewidth=0.5, alpha=0.6)
    for j in range(nx):
        ax1.plot(X[:, j], Y[:, j], "b-", linewidth=0.5, alpha=0.6)
    ax1.set_xlabel("X (m)", fontsize=11)
    ax1.set_ylabel("Y (m)", fontsize=11)
    ax1.set_title("XY Slice (Plan View) - Ground Level", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.ticklabel_format(style="plain")

    # XZ slice (elevation view through center)
    center_y_idx = ny // 2
    for i in range(total_z_cells + 1):
        ax2.plot(x_coords, [z_coords[i]] * nx, "b-", linewidth=0.5, alpha=0.6)
    for j in range(nx):
        ax2.plot(
            [x_coords[j]] * (total_z_cells + 1),
            z_coords,
            "b-",
            linewidth=0.5,
            alpha=0.6,
        )
    ax2.axhline(
        y=ground_height,
        color="brown",
        linestyle="--",
        linewidth=1.5,
        label="Ground",
        alpha=0.7,
    )
    ax2.set_xlabel("X (m)", fontsize=11)
    ax2.set_ylabel("Z (m)", fontsize=11)
    ax2.set_title(
        f"XZ Slice (Elevation View) - Y={y_coords[center_y_idx]:.0f}m",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right")
    ax2.ticklabel_format(style="plain")

    plt.tight_layout()

    # Create Figure 2: Cell spacing distributions
    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15, 5))

    # X spacing
    x_positions = (x_coords[:-1] + x_coords[1:]) / 2
    ax3.plot(x_positions, x_cells, "b-", linewidth=2)
    if fillPlot:
        ax3.fill_between(x_positions, x_cells, alpha=0.3)
    ax3.set_xlabel("X Position (m)", fontsize=11)
    ax3.set_ylabel("Cell Size (m)", fontsize=11)
    if logY:
        ax3.set_yscale("log")
    ax3.set_title("X-Direction Cell Spacing", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle="--", which="both")

    # Y spacing
    y_positions = (y_coords[:-1] + y_coords[1:]) / 2
    ax4.plot(y_positions, y_cells, "g-", linewidth=2)
    if fillPlot:
        ax4.fill_between(y_positions, y_cells, alpha=0.3, color="g")
    ax4.set_xlabel("Y Position (m)", fontsize=11)
    ax4.set_ylabel("Cell Size (m)", fontsize=11)
    if logY:
        ax4.set_yscale("log")
    ax4.set_title("Y-Direction Cell Spacing", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, linestyle="--", which="both")

    # Z spacing
    z_positions = (z_coords[:-1] + z_coords[1:]) / 2
    ax5.plot(z_cells, z_positions, "r-", linewidth=2)
    if fillPlot:
        ax5.fill_betweenx(z_positions, z_cells, alpha=0.3, color="r")
    ax5.axhline(
        y=ground_height, color="brown", linestyle="--", linewidth=1.5, alpha=0.7
    )

    ax5.set_xlabel("Cell Size (m)", fontsize=11)
    ax5.set_ylabel("Z Position (m)", fontsize=11)
    if logY:
        ax5.set_xscale("log")
    ax5.set_title("Z-Direction Cell Spacing", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3, linestyle="--", which="both")

    plt.tight_layout()

    # Save figures
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig1_path = output_path / "mesh_slices.png"
        fig2_path = output_path / "mesh_spacing.png"

        # fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")

        print(f"\nPlots saved:")
        print(f"  {fig1_path}")
        print(f"  {fig2_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize terrain mesh structure from config"
    )
    parser.add_argument("config", help="Path to terrain_config.yaml")
    parser.add_argument(
        "--ground-height",
        type=float,
        required=True,
        help="Average ground elevation in meters (e.g., 250)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for plots (default: current directory)",
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return

    visualize_mesh(args.config, args.ground_height, args.output)


if __name__ == "__main__":
    main()
