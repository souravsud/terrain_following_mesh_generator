"""Main entry point for terrain mesh generation.

This script provides a simple interface to run the complete mesh generation
pipeline using configuration from a YAML file.

Usage:
    python run.py --config terrain_config.yaml --dem terrain_data.tif --output ./output
    
For help:
    python run.py --help
"""
import terrain_mesh as tm
import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for the application.
    
    Args:
        verbose: If True, set log level to DEBUG with full format, otherwise INFO with minimal format
    """
    import logging
    
    if verbose:
        level = logging.DEBUG
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    else:
        level = logging.INFO
        fmt = '%(message)s'
    logging.basicConfig(level=level, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')


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
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please create a configuration file or specify one with --config")
        sys.exit(1)
    
    # Load all configurations from YAML file
    try:
        configs = tm.load_config(str(config_path))
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Determine paths
    if not args.dem:
        logger.error("No DEM file specified. Provide a path with --dem.")
        sys.exit(1)

    dem_path = args.dem
    rmap_path = args.rmap  # None unless explicitly provided via --rmap
    output_dir = args.output if args.output else "./output"

    # Check if DEM file exists
    if not Path(dem_path).exists():
        logger.error(f"DEM file not found: {dem_path}")
        sys.exit(1)
    
    # Check if roughness map exists (optional)
    if rmap_path and not Path(rmap_path).exists():
        logger.warning(f"Roughness map not found: {rmap_path} — continuing without it")
        rmap_path = None
    
    logger.info(f"Config: {config_path} | DEM: {dem_path} | Output: {output_dir}")
    
    # Run pipeline with loaded configs
    try:
        pipeline = tm.TerrainMeshPipeline()
        results = pipeline.run(
            dem_path=dem_path,
            rmap_path=rmap_path,
            output_dir=output_dir,
            **configs  # Unpacks all config objects
        )
        
        logger.info(f"Results saved to: {results['output_dir']}")
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())