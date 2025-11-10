# Terrain-Following Mesh Generator for Atmospheric Simulations (OpenFOAM)

This tool generates a **structured, terrain-following orthogonal mesh** for atmospheric simulations in [OpenFOAM](https://openfoam.org/).  
It downloads terrain elevation data (DEM) and optional surface roughness maps, then produces a `blockMeshDict` and aerodynamic roughness field (`z0`) for high-fidelity wind flow simulations.

### PS: The circular O-grid mesh implementation is incomplete
---

## Features
- **Automated DEM download** from global elevation datasets (GLO-30, SRTM, etc.)
- **Automatic surface roughness mapping** from ESA WorldCover land classification
- **UTM reprojection** at download time for optimal processing performance
- **Structured, graded surface mesh** generation with user-defined resolution and grading
- **Terrain-following vertical extrusion** to specified ceiling height
- **Aerodynamic roughness field (`z0`) generation** for OpenFOAM ABL simulations
- **Boundary treatment** to minimize artificial terrain edge effects
- **Multi-block grading** support for refined regions of interest
- **Batch processing** for multiple locations via CSV input
- Outputs ready-to-use `system/blockMeshDict` and `0/include/z0Values` files

---

## Workflow

### Phase 1: Download Terrain & Roughness Data
1. Specify location(s) via **lat/lon coordinates** (single location or CSV batch)
2. Tool **downloads DEM tiles** and **stitches** them into domain-sized GeoTIFF
3. (Optional) Downloads **ESA WorldCover** land classification and converts to **z0 roughness values**
4. **Reprojects** both DEM and roughness map to **UTM** (meters) for processing efficiency
5. Saves **GeoTIFF files** and **metadata JSON** to location-specific folders

### Phase 2: Generate Mesh
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

## Usage

Add the path to the DEM file in the run.py (dem_path) and the output directory-output_dir (preferebly the openFOAM case directory so that it will create the Dictionary file in the correct location). Setup all the required settings in the config.yaml file and then run the python code.

If you have your own DEM files:
- **GeoTIFF files**: Supported directly (will be reprojected to UTM if needed)
- **DAT files**: Custom format with UTM coordinates (debug/testing)
- **NetCDF files**: With 2D coordinate arrays (debug/testing)

```bash
python run.py
```

## Configuration

### Configuration Parameters

#### **Terrain Section**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `center_lat` | float | Center latitude of terrain region | *required* |
| `center_lon` | float | Center longitude of terrain region | *required* |
| `center_coordinates` | tuple/null | Override with UTM coordinates `[x, y]` (for advanced use) | `null` |
| `crop_size_km` | float | Size of terrain region to extract (km) | *required* |
| `rotation_deg` | float | Rotation angle of domain (degrees, clockwise from North) | `0.0` |
| `smoothing_sigma` | float | Gaussian smoothing applied to terrain (0=no smoothing) | `2.0` |

**Note**: If using downloaded GeoTIFFs, `center_coordinates` will be auto-loaded from metadata JSON.

#### **Grid Section**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `nx` | int | Number of cells in x-direction | *required* |
| `ny` | int | Number of cells in y-direction | *required* |
| `x_grading` | list | Multi-block grading in x-direction (see format below) | `null` (uniform) |
| `y_grading` | list | Multi-block grading in y-direction (see format below) | `null` (uniform) |

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
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `domain_height` | float | Height of computational domain (m) | `1000.0` |
| `total_z_cells` | int | Total cells in vertical direction | `10` |
| `z_grading` | list | Vertical grading specification (same format as x/y grading) | `null` (uniform) |
| `patch_types` | dict | OpenFOAM patch type for each boundary | See below |
| `extract_inlet_face_info` | bool | Extract inlet face information | `true` |

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

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `aoi_fraction` | float | Fraction of domain considered as area-of-interest (center region) | `0.4` |
| `boundary_mode` | str | Treatment mode: `uniform` or `directional` | `uniform` |
| `flat_boundary_thickness_fraction` | float | Thickness of flattened boundary region (fraction of domain) | `0.1` |
| `enabled_boundaries` | list | Boundaries to flatten: `[east, west, north, south]` | `[east, west]` |
| `smoothing_method` | str | Smoothing kernel: `mean`, `gaussian`, or `median` | `mean` |
| `kernel_progression` | str | How smoothing increases: `exponential` or `linear` | `exponential` |
| `base_kernel_size` | int | Initial smoothing kernel size | `null` (auto) |
| `max_kernel_size` | int | Maximum smoothing kernel size | `null` (auto) |
| `progression_rate` | float | Rate of kernel size increase (for exponential) | `1.5` |
| `boundary_flatness_mode` | str | Flattening method: `heavy_smooth` or `blend_target` | `heavy_smooth` |
| `uniform_elevation` | float | Override boundary height (m) | `null` (auto-calculate) |

**Boundary modes**:
- `uniform`: Apply flattening uniformly around entire perimeter
- `directional`: Apply flattening only to specified boundaries (e.g., inlet/outlet only)

#### **Visualization Section**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `create_plots` | bool | Generate visualization plots | `true` |
| `show_grid_lines` | bool | Display grid lines on plots | `true` |
| `save_high_res` | bool | Save high-resolution plots | `true` |
| `plot_format` | str | Output format: `png`, `pdf`, `svg` | `png` |
| `dpi` | int | Plot resolution | `150` |

---

### Typical Use Cases

Configure the mesh generation using a YAML file. Below are two example configurations.

### Simple Configuration (Uniform Mesh)
```yaml
# config_simple.yaml
terrain:
  center_lat: 39.71121111
  center_lon: -7.73483333
  crop_size_km: 25
  rotation_deg: 0
  smoothing_sigma: 0
  
grid:
  nx: 200
  ny: 200
  
mesh:
  domain_height: 3000.0
  total_z_cells: 50
```

### Advanced Configuration (Multi-Grading)
```yaml
# config_advanced.yaml
terrain:
  center_lat: 39.71121111
  center_lon: -7.73483333
  crop_size_km: 25
  rotation_deg: 45
  smoothing_sigma: 0
  center_coordinates: false
  
grid:
  nx: 384
  ny: 384
  x_grading:
    - [0.35, 0.12, 0.05]   # [length_fraction, cell_fraction, expansion_ratio]
    - [0.30, 0.76, 1.0]
    - [0.35, 0.12, 20.0]
  y_grading:
    - [0.35, 0.12, 0.05]
    - [0.30, 0.76, 1.0]
    - [0.35, 0.12, 20.0]
    
mesh:
  domain_height: 3000.0
  total_z_cells: 60
  z_grading:
    - [0.033, 0.50, 1.0]   # Near ground refinement
    - [0.967, 0.50, 100.0] # Expansion to domain top
  patch_types:
    ground: wall
    sky: patch
    inlet: patch
    outlet: patch
    sides: patch
    
boundary:
  aoi_fraction: 0.3
  boundary_mode: directional
  flat_boundary_thickness_fraction: 0.08
  enabled_boundaries: [east, west]
  smoothing_method: mean
  kernel_progression: exponential
  base_kernel_size: 5
  progression_rate: 10
  boundary_flatness_mode: blend_target
  
visualization:
  create_plots: true
  show_grid_lines: true
  save_high_res: true
  plot_format: png
  dpi: 150
```