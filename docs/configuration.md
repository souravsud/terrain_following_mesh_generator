# Configuration Guide

All configuration is supplied via a YAML file and mapped to typed dataclasses.
Load it with:

```python
from terrain_mesh.config import load_config
configs = load_config("terrain_config.yaml")
```

---

## Terrain

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `crop_size_km` | float | Yes | — | Domain side length (km). The DEM is cropped to this size centred on the site. |
| `rotation_deg` | float | Yes | — | Domain rotation clockwise from North (degrees). Aligns the domain x-axis with the prevailing wind direction. |
| `center_lat` | float | No | auto | Site latitude (decimal degrees). Auto-detected from the DEM if omitted (GeoTIFF only). |
| `center_lon` | float | No | auto | Site longitude (decimal degrees). Auto-detected from the DEM if omitted (GeoTIFF only). |
| `smoothing_sigma` | float | No | 0.0 | Gaussian pre-smoothing of the raw DEM (sigma in pixels). 0 = disabled. Useful for noisy or coarse DEMs. |
| `roughness_smoothing_sigma` | float | No | 0.0 | Same smoothing applied to the roughness map. |
| `center_coordinates` | bool | No | false | If true, the output coordinate system is shifted so the domain centre is (0, 0). |

---

## Grid

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `nx` | int | Yes | — | Number of grid vertices in x (flow direction). Cells = `nx − 1`. |
| `ny` | int | Yes | — | Number of grid vertices in y (cross-flow direction). Cells = `ny − 1`. |
| `x_grading` | list | No | uniform | Multi-block grading in x. See format below. |
| `y_grading` | list | No | uniform | Multi-block grading in y. See format below. |

### Grading format

Each entry is `[length_fraction, cell_fraction, expansion_ratio]`:

- **length_fraction** — fraction of domain length occupied by this block.
- **cell_fraction** — fraction of total cells allocated to this block.
- **expansion_ratio** — `last_cell_size / first_cell_size` within the block. Values < 1 refine toward the start, > 1 refine toward the end.

All `length_fraction` values must sum to 1.0; same for `cell_fraction`.

### Example — refined at boundaries, uniform in centre

```yaml
x_grading:
  - [0.35, 0.12, 0.05]   # 35 % of domain, 12 % of cells, coarsening inward
  - [0.30, 0.76, 1.0]    # central 30 %, uniform
  - [0.35, 0.12, 20.0]   # 35 % of domain, 12 % of cells, coarsening outward
```

---

## Mesh

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `domain_height` | float | No | 4000 | Vertical domain extent (m). |
| `total_z_cells` | int | Yes | — | Number of cells in the vertical direction. |
| `z_grading` | list | No | uniform | Vertical grading (same format as `x_grading`). |
| `terrain_normal_first_layer` | bool | No | false | If true, the first cell layer is extruded along the terrain surface normal rather than purely vertically. |
| `adjust_ceiling_for_terrain` | bool | No | false | If true, raises the domain ceiling by the minimum terrain elevation so the effective air-column height above ground is constant regardless of site altitude. |
| `patch_types` | dict | No | see below | OpenFOAM patch type for each boundary face. |
| `default_z0` | float | No | 0.1 | Constant aerodynamic roughness length (m) used when no roughness map is provided. |

### Default patch types

```yaml
patch_types:
  ground: wall
  sky: patch
  inlet: patch
  outlet: patch
  sides: patch
```

---

## Boundary

Controls the 3-zone smooth-step boundary treatment applied at the inlet/outlet faces to enforce numerical stability.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `aoi_fraction` | float | No | 0.4 | Fraction of the domain size reserved as the Area of Interest (AOI). Original terrain is fully preserved inside the AOI. |
| `boundary_mode` | str | No | `uniform` | `directional` — treat selected faces independently (recommended for wind-aligned domains). `uniform` — radial treatment applied symmetrically to all sides. |
| `flat_boundary_thickness_fraction` | float | No | 0.1 | Fraction of the domain width used as the flat zone on each enabled face. The outer half is set to the target elevation; the inner half is used for the clamp calculation. |
| `enabled_boundaries` | list | No | `[east, west]` (directional) | Which faces to treat. Use `east` (outlet/downwind) and `west` (inlet/upwind) in the flow-aligned frame. |
| `clamp_target` | bool | No | true | If true, the extrapolated target elevation is capped at the median of the real terrain in the flat zone, preventing the flat face from being set above the actual terrain on upward-sloping domains. Recommended for large diverse datasets. |

### How the target elevation is computed

The flat zone target is determined independently for each enabled face:

1. The transition zone pixels for that face are collapsed to a 1-D median profile along the flow direction.
2. A robust line (Theil-Sen estimator) is fitted to this profile.
3. The line is extrapolated to the flat-zone boundary x-coordinate.
4. If `clamp_target: true`, the result is capped at `median(real terrain in flat zone)`.

This approach is insensitive to edge anomalies (cliffs, rivers at the crop boundary) and handles all terrain shapes — isolated peaks, valleys, coastal terrain, and sloped domains — correctly.

### Zone geometry (directional mode)

```
|── flat ──|──── smooth-step transition ────|──── AOI ────|──── smooth-step ────|── flat ──|
  ~5–8 %            ~20–35 %                  ~30–50 %          ~20–35 %          ~5–8 %
```

- **AOI**: `|flow_x − centre| ≤ aoi_fraction × domain_size / 2`
- **Flat zone**: outermost `flat_boundary_thickness_fraction / 2` of domain width
- **Transition**: everything in between; blend weight `w = 3t²−2t³` where `t = 0` at the AOI edge and `t = 1` at the flat-zone boundary
- **Corners** (side-boundary bands): left as original terrain

### Example

```yaml
boundary:
  aoi_fraction: 0.50
  boundary_mode: directional
  flat_boundary_thickness_fraction: 0.08
  enabled_boundaries: [west, east]
  clamp_target: true
```

---

## Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `create_plots` | bool | true | Generate PNG plots alongside the mesh outputs. |
| `show_grid_lines` | bool | true | Overlay the mesh grid on terrain plots. |
| `save_high_res` | bool | true | Also save 300 dpi versions of all plots. |
| `plot_format` | str | `png` | Output format: `png`, `pdf`, or `svg`. |
| `dpi` | int | 150 | Resolution for raster formats. |

---

## Full example

```yaml
terrain:
  crop_size_km: 30
  rotation_deg: 225
  smoothing_sigma: 0

grid:
  nx: 361
  ny: 361
  x_grading:
    - [0.35, 0.0835, 0.025]
    - [0.30, 0.833,  1.0]
    - [0.35, 0.0835, 40.0]
  y_grading:
    - [0.35, 0.0835, 0.025]
    - [0.30, 0.833,  1.0]
    - [0.35, 0.0835, 40.0]

mesh:
  domain_height: 5000.0
  total_z_cells: 76
  terrain_normal_first_layer: true
  adjust_ceiling_for_terrain: true
  z_grading:
    - [0.04, 0.658, 1.0]
    - [0.96, 0.342, 180.0]

boundary:
  aoi_fraction: 0.30
  boundary_mode: directional
  flat_boundary_thickness_fraction: 0.08
  enabled_boundaries: [west, east]
  clamp_target: true

visualization:
  create_plots: true
  dpi: 150
```

Run:

```bash
python run.py --config terrain_config.yaml --dem path/to/terrain.tif --output ./output
```
