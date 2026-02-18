# Examples

This directory contains example scripts demonstrating different use cases for the terrain mesh generator.

## Files

### `simple_example.py`
Basic example showing how to:
- Create a simple configuration programmatically
- Generate a uniform mesh without grading
- Run the pipeline with minimal settings

**Usage:**
```bash
python simple_example.py
```

### `advanced_example.py`
Advanced example demonstrating:
- Multi-block grading for refined center regions
- Vertical grading for near-ground refinement
- Directional boundary treatment (inlet/outlet flattening)
- Roughness map integration for z0 field

**Usage:**
```bash
python advanced_example.py
```

## Configuration via YAML

For more complex setups, it's recommended to use YAML configuration files.
See the sample configurations in the `../sampleConfigs/` directory.

**Example using YAML:**
```bash
cd ..
python run.py --config sampleConfigs/terrain_config_4M_40m.yaml --dem path/to/dem.tif --output ./output
```

## Typical Workflow

1. **Start with simple example** - Get familiar with the basic workflow
2. **Customize parameters** - Adjust grid size, domain height, etc.
3. **Add grading** - Use multi-block grading for refined regions
4. **Add boundary treatment** - Flatten inlet/outlet for better flow
5. **Use YAML config** - Move to YAML for production runs

## Common Parameters

### Grid Resolution
- **Coarse**: 100-200 cells per direction (~250m resolution for 25km domain)
- **Medium**: 200-400 cells per direction (~100m resolution)
- **Fine**: 400-800 cells per direction (~50m resolution)

### Domain Height
- **Flat terrain**: 1000-2000m
- **Hilly terrain**: 2000-3000m
- **Mountainous**: 3000-5000m

### Vertical Cells
- **Coarse**: 20-30 cells
- **Medium**: 40-60 cells
- **Fine**: 80-120 cells

## Tips

1. **Start coarse** - Test with coarse mesh first to verify setup
2. **Match resolution** - Horizontal and vertical cell sizes should be similar near ground
3. **Use grading** - Refine where flow features are important (center, ground level)
4. **Boundary treatment** - Flatten inlet/outlet to reduce artificial terrain effects
5. **Check outputs** - Always review visualization plots before running simulations
