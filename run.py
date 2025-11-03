import terrain_mesh as tm

def main():
    # Load all configurations from YAML file
    configs = tm.load_config("terrain_config.yaml")
    
    # Run pipeline with loaded configs
    pipeline = tm.TerrainMeshPipeline()
    results = pipeline.run(
        dem_path="/Users/ssudhakaran/Documents/CFD_Dataset/GenerateInputs/Data/terrain_0001_glo_30_N39_711_W007_735_50km.tif",
        rmap_path = "/Users/ssudhakaran/Documents/CFD_Dataset/GenerateInputs/Data/roughness_0001_worldcover_N39_711_W007_735_50km.tif",
        output_dir="/Users/ssudhakaran/Documents/Simulations/API/openFoam/meshRefine",
        **configs  # Unpacks all config objects
    )
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()