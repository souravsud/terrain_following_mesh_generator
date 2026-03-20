# Terrain-Following Mesh Generator for Atmospheric Simulations (OpenFOAM)

This tool generates a **structured, terrain-following orthogonal mesh** for atmospheric simulations in [OpenFOAM](https://www.openfoam.com/).  
It pre-processes elevation data (DEM) and optional surface roughness maps, then produces a `blockMeshDict` and aerodynamic roughness field (`z0`) for high-fidelity wind flow simulations.

---

## Features
- **Automatic surface roughness mapping** from ESA WorldCover land classification
- **Structured, graded surface mesh** generation with user-defined resolution and grading
- **Terrain-following vertical extrusion** to specified ceiling height
- **Boundary treatment** to minimize artificial terrain edge effects
- **Multi-block grading** support for refined regions of interest
- Outputs ready-to-use `system/blockMeshDict` and `0/include/z0Values` files
- Quick mesh sanity checks using mesh_visualiser

---

## Workflow

1. Load **UTM-projected DEM** (and optional roughness map)
2. **Extract and rotate** terrain region based on config
3. Apply **boundary treatment** (smoothing/flattening at edges)
4. Generate **structured terrain mesh** with specified resolution and grading
5. **Extrude vertically** to ceiling height with z-grading
6. Export `blockMeshDict` for OpenFOAM
7. Map **z0 roughness values** to mesh faces for ABL simulations

---

## Installation

We recommend using a [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment to manage dependencies.

### 1. Clone the repository

```bash
git clone https://github.com/souravsud/terrain_following_mesh_generator.git
cd terrain_following_mesh_generator

```

### 2. Create and activate conda environment
```bash
conda env create -f environment.yml
conda activate tfmesh
```

### 3. Install the package

Install as a package so it can be imported from any script or project:

```bash
# Standard install
pip install .

# Editable install (recommended during development -- changes to source take effect immediately)
pip install -e .
```

After installation, `terrain_mesh` is importable from anywhere:

```python
import terrain_mesh as tm

grid_config = tm.create_uniform_grid_config(nx=200, ny=200)
pipeline = tm.TerrainMeshPipeline()
```

## Usage

### Quick Start

The simplest way to use the tool is with the command-line interface:

```bash
# Show help and available options
python run.py --help

# Run with your DEM file and config (--dem is required)
python run.py --config terrain_config.yaml --dem terrain_data.tif --output ./output

# Include roughness map for z0 field generation
python run.py --dem terrain.tif --rmap roughness.tif --output ./output

# Enable verbose logging for debugging
python run.py --dem terrain.tif --verbose
```

> **Note**: `--dem` is required. There is no built-in DEM download — you must supply
> your own GeoTIFF (or DAT / NetCDF) file for the area of interest.

### Advanced Usage

For programmatic usage, see the examples in the `examples/` directory:
- `examples/simple_example.py` - Basic usage with uniform mesh
- `examples/advanced_example.py` - Advanced usage with grading and boundary treatment

### Inputs

Currently the tool supports DEM data in the following formats:
- **GeoTIFF files**: Supported directly (will be reprojected to UTM if needed)
- **DAT files**: Custom format with UTM coordinates (debug/testing)
- **NetCDF files**: With 2D coordinate arrays (debug/testing)

## Examples

See the `examples/` directory for ready-to-use example scripts:
- `simple_example.py` - Generate a uniform mesh
- `advanced_example.py` - Use multi-block grading and boundary treatment
- `README.md` - Guidelines and tips for mesh generation

## Configuration

### Configuration Parameters

#### **Terrain Section**
| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `center_lat` | float | Center latitude of terrain region | No | `null` (auto-detect from GeoTIFF when possible) |
| `center_lon` | float | Center longitude of terrain region | No | `null` (auto-detect from GeoTIFF when possible) |
| `center_coordinates` | bool | Centres the coordinate system at map center | No | `false` |
| `crop_size_km` | float | Size of terrain region to extract (km) | Yes | No default |
| `rotation_deg` | float | Rotation angle of domain (degrees, clockwise from North) | Yes | No default|
| `smoothing_sigma` | float | Gaussian smoothing applied to terrain (0=no smoothing) | No | `0.0` |

**Note**: If using downloaded GeoTIFFs, `center_coordinates` will be auto-loaded from metadata JSON.

#### **Grid Section**
| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `nx` | int | Number of cells in x-direction | Yes | No default |
| `ny` | int | Number of cells in y-direction | Yes | No default |
| `x_grading` | list | Multi-block grading in x-direction (see format below) | No | `null` (uniform) |
| `y_grading` | list | Multi-block grading in y-direction (see format below) | No | `null` (uniform) |

**Grading Format**: Each grading entry is `[length_fraction, cell_fraction, expansion_ratio]`
- `length_fraction`: Fraction of domain length for this block
- `cell_fraction`: Fraction of total cells in this block  
- `expansion_ratio`: Cell size ratio (last/first cell in block)
- All `length_fraction` values must sum to 1.0
- All `cell_fraction` values must sum to 1.0

**Example**: Refine center, coarsen edges
```yaml
x_grading:
  - [0.35, 0.12, 0.05]   # First 35% of domain, 12% of cells, compress toward center
  - [0.30, 0.76, 1.0]    # Middle 30%, 76% of cells, uniform
  - [0.35, 0.12, 20.0]   # Last 35%, 12% of cells, expand away from center
```

#### **Mesh Section**
| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `domain_height` | float | Height of computational domain (m) | No | `4000.0` |
| `total_z_cells` | int | Total cells in vertical direction | Yes | No default |
| `z_grading` | list | Vertical grading specification (same format as x/y grading) | No | `null` (uniform) |
| `patch_types` | dict | OpenFOAM patch type for each boundary | No | See below |
| `extract_inlet_face_info` | bool | Extract inlet face information | No | `true` |

**Default patch types**:
```yaml
patch_types:
  ground: wall
  sky: patch
  inlet: patch
  outlet: patch
  sides: patch
```

#### **Boundary Section** (Advanced)
Controls boundary treatment to reduce artificial terrain effects at domain edges.

| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `aoi_fraction` | float | Fraction of domain considered as area-of-interest (center region) | No | `0.4` |
| `boundary_mode` | str | Treatment mode: `uniform` or `directional` | No | `uniform` |
| `flat_boundary_thickness_fraction` | float | Thickness of flattened boundary region (fraction of domain) | No | `0.1` |
| `enabled_boundaries` | list | Boundaries to flatten: `[east, west, north, south]` | No | `[east, west]` |
| `smoothing_method` | str | Smoothing kernel: `mean`, `gaussian`, or `median` | No | `mean` |
| `kernel_progression` | str | How smoothing increases: `exponential` or `linear` | No | `exponential` |
| `base_kernel_size` | int | Initial smoothing kernel size | No | `null` (auto) |
| `max_kernel_size` | int | Maximum smoothing kernel size | No | `null` (auto) |
| `progression_rate` | float | Rate of kernel size increase (for exponential) | No | `1.5` |
| `boundary_flatness_mode` | str | Flattening method: `heavy_smooth` or `blend_target` | No | `heavy_smooth` |
| `uniform_elevation` | float | Override boundary height (m) | No | `null` (auto-calculate) |

**Boundary modes**:
- `uniform`: Apply flattening uniformly around entire perimeter
- `directional`: Apply flattening only to specified boundaries (e.g., inlet/outlet only)

#### **Visualization Section**
| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `create_plots` | bool | Generate visualization plots | No | `true` |
| `show_grid_lines` | bool | Display grid lines on plots | No | `true` |
| `save_high_res` | bool | Save high-resolution plots | No | `true` |
| `plot_format` | str | Output format: `png`, `pdf`, `svg` | No | `png` |
| `dpi` | int | Plot resolution | No | `150` |

---


### Demo: Simple Case

1. Save this as `config_demo.yaml`:

```yaml
terrain:
  center_lat: 39.71121111
  center_lon: -7.73483333
  crop_size_km: 30
  rotation_deg: 225

grid:
  nx: 200
  ny: 200

mesh:
  domain_height: 3000.0
  total_z_cells: 50
```

2. Run the pipeline:

```bash
python run.py --config config_demo.yaml --dem path/to/terrain.tif --output ./output
```

### Demo: Ouput

Paste screenshots directly under each heading below.

#### 1) Output figures
![alt text](examples/sample_Plots/boundary_treatment.png)
![alt text](examples/sample_Plots/roughness_analysis.png)

#### 2) Sample slice of the mesh generated from the blockMeshDict
![alt text](examples/sample_Plots/volMeshSlice.png)
