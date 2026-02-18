"""Main entry point for terrain mesh generation.

This script provides a simple interface to run the complete mesh generation
pipeline using configuration from a YAML file.

Usage:
    python run.py --config terrain_config.yaml --dem path/to/dem.tif --output ./output
    
For help:
    python run.py --help
"""
import terrain_mesh as tm
import argparse
import sys
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Configure logging for the application.
    
    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
    """
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate terrain-following mesh for OpenFOAM simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python run.py
  
  # Specify custom paths
  python run.py --config my_config.yaml --dem terrain.tif --output ./mesh_output
  
  # Include roughness map for z0 field generation
  python run.py --dem terrain.tif --rmap roughness.tif --output ./output
  
  # Enable verbose logging
  python run.py --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='terrain_config.yaml',
        help='Path to YAML configuration file (default: terrain_config.yaml)'
    )
    
    parser.add_argument(
        '--dem', '-d',
        type=str,
        help='Path to Digital Elevation Model file (GeoTIFF, DAT, or NetCDF). '
             'If not provided, must be set in the script.'
    )
    
    parser.add_argument(
        '--rmap', '-r',
        type=str,
        help='Optional path to roughness map for z0 field generation'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for generated files. '
             'If not provided, must be set in the script.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for mesh generation pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print(f"Please create a configuration file or specify one with --config")
        sys.exit(1)
    
    # Load all configurations from YAML file
    try:
        configs = tm.load_config(str(config_path))
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Determine paths (command line overrides hardcoded defaults)
    # TODO: Update these default paths for your system
    default_dem_path = "/Users/ssudhakaran/Documents/Simulations/2025/generateInputs/Data_test/downloads/terrain_0001_N39_711_W007_735/terrain_0001_glo_30_N39_711_W007_735_50km.tif"
    default_rmap_path = "/Users/ssudhakaran/Documents/Simulations/2025/generateInputs/Data_test/downloads/terrain_0001_N39_711_W007_735/roughness_0001_worldcover_N39_711_W007_735_50km.tif"
    default_output_dir = "/Users/ssudhakaran/Documents/Simulations/API/openFoam/meshRefine"
    
    dem_path = args.dem if args.dem else default_dem_path
    rmap_path = args.rmap if args.rmap else default_rmap_path
    output_dir = args.output if args.output else default_output_dir
    
    # Check if DEM file exists
    if not Path(dem_path).exists():
        print(f"Error: DEM file not found: {dem_path}")
        print(f"Please specify a valid DEM file with --dem")
        sys.exit(1)
    
    # Check if roughness map exists (optional)
    if rmap_path and not Path(rmap_path).exists():
        print(f"Warning: Roughness map not found: {rmap_path}")
        print(f"Continuing without roughness map...")
        rmap_path = None
    
    print(f"Configuration file: {config_path}")
    print(f"DEM file: {dem_path}")
    print(f"Roughness map: {rmap_path if rmap_path else 'None'}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Run pipeline with loaded configs
    try:
        pipeline = tm.TerrainMeshPipeline()
        results = pipeline.run(
            dem_path=dem_path,
            rmap_path=rmap_path,
            output_dir=output_dir,
            **configs  # Unpacks all config objects
        )
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {results['output_dir']}")
        
        return 0
    
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())