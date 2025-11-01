import terrain_mesh as tm

def main():
    # Load all configurations from YAML file
    configs = tm.load_config("terrain_config.yaml")
    
    # Run pipeline with loaded configs
    pipeline = tm.TerrainMeshPipeline()
    results = pipeline.run(
        dem_path="/Users/ssudhakaran/Documents/Simulations/2025/perdigao_validation_ventos_openFoam/ventosData/topo_square.dat",
        rmap_path = "/Users/ssudhakaran/Documents/Simulations/2025/perdigao_validation_ventos_openFoam/ventosData/newa_perdigao_map_roug_summer.nc",
        output_dir="/Users/ssudhakaran/Documents/Simulations/API/openFoam/meshRefine",
        **configs  # Unpacks all config objects
    )
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()