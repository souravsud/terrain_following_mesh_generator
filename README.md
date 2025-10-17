# Terrain-Following Mesh Generator for Atmospheric Simulations (OpenFOAM)

This tool generates a **structured, terrain-following orthogonal mesh** for atmospheric simulations in [OpenFOAM](https://openfoam.org/).  
It takes an input terrain surface and automatically produces a `blockMeshDict` that can be used directly with OpenFOAM’s `blockMesh` utility.

---

## Features
- Creates a **graded, structured surface mesh** from input terrain, based on **user-defined resolution and grading scheme**.  
- Extrudes the surface mesh up to a **user-specified ceiling height**.  
- Supports **vertical grading** in the z-direction, controlled by user input.  
- Outputs a ready-to-use `system/blockMeshDict` file.  
- Designed for **atmospheric boundary layer and wind flow simulations**.  

---

## Workflow
1. **Input terrain** (e.g., DEM or structured point set).  
2. Tool generates a **structured terrain mesh** (surface aligned with terrain, following user-specified grading and resolution).  
3. Mesh is **extruded vertically** to ceiling height.  
4. Vertical layers are **graded** based on user-specified z-grading.  
5. Final mesh specification is exported as a `blockMeshDict`.  
6. Run OpenFOAM’s `blockMesh` to generate the mesh.  

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
| `crop_size_km` | float | Size of terrain region to extract (km) | *required* |
| `rotation_deg` | float | Rotation angle of domain (degrees, clockwise) | `0.0` |
| `smoothing_sigma` | float | Gaussian smoothing applied to terrain (0=no smoothing) | `2.0` |
| `center_coordinates` | bool | Center coordinates at origin | `false` |

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