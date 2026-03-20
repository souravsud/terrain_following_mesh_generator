# Configuration Guide

This document describes all configuration options.

---

## Terrain

| Parameter | Type | Description | Required | Default |
|----------|------|------------|----------|---------|
| center_lat | float | Center latitude | No | auto |
| center_lon | float | Center longitude | No | auto |
| center_coordinates | bool | Center coordinate system | No | false |
| crop_size_km | float | Domain size (km) | Yes | — |
| rotation_deg | float | Domain rotation (deg) | Yes | — |
| smoothing_sigma | float | Terrain smoothing | No | 0.0 |

---

## Grid

| Parameter | Type | Description | Required | Default |
|----------|------|------------|----------|---------|
| nx | int | Cells in x-direction | Yes | — |
| ny | int | Cells in y-direction | Yes | — |
| x_grading | list | Multi-block grading (x) | No | uniform |
| y_grading | list | Multi-block grading (y) | No | uniform |

### Grading format

```
[length_fraction, cell_fraction, expansion_ratio]
```

- Fractions must sum to 1.0
- Expansion ratio = last cell / first cell

### Example

```yaml
x_grading:
  - [0.35, 0.12, 0.05]
  - [0.30, 0.76, 1.0]
  - [0.35, 0.12, 20.0]
```

---

## Mesh

| Parameter | Type | Description | Required | Default |
|----------|------|------------|----------|---------|
| domain_height | float | Domain height (m) | No | 4000 |
| total_z_cells | int | Vertical cells | Yes | — |
| z_grading | list | Vertical grading | No | uniform |
| patch_types | dict | OpenFOAM patch types | No | default |
| extract_inlet_face_info | bool | Extract inlet faces | No | true |

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

## Boundary (Advanced)

| Parameter | Description |
|----------|------------|
| aoi_fraction | Area of interest fraction |
| boundary_mode | uniform / directional |
| flat_boundary_thickness_fraction | Boundary thickness |
| enabled_boundaries | Boundaries to apply |
| smoothing_method | mean / gaussian / median |
| kernel_progression | exponential / linear |
| progression_rate | Growth rate |
| boundary_flatness_mode | heavy_smooth / blend_target |

---

## Visualization

| Parameter | Description | Default |
|----------|------------|---------|
| create_plots | Generate plots | true |
| show_grid_lines | Show grid | true |
| save_high_res | Save high resolution | true |
| plot_format | png/pdf/svg | png |
| dpi | Resolution | 150 |

---

## Demo

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

Run:

```bash
python run.py --config config_demo.yaml --dem path/to/terrain.tif --output ./output
```