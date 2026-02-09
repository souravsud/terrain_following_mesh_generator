import terrain_mesh as tm

def main():
    # Load all configurations from YAML file
    configs = tm.load_config("terrain_config.yaml")
    
    # Run pipeline with loaded configs
    pipeline = tm.TerrainMeshPipeline()
    results = pipeline.run(
        dem_path="/Users/ssudhakaran/Documents/Simulations/generateInputs/Data_test/downloads/terrain_0001_N39_711_W007_735/terrain_0001_glo_30_N39_711_W007_735_50km.tif",
        rmap_path = "/Users/ssudhakaran/Documents/Simulations/generateInputs/Data_test/downloads/terrain_0001_N39_711_W007_735/roughness_0001_worldcover_N39_711_W007_735_50km.tif",
        output_dir="/Users/ssudhakaran/Documents/validation/validationMeshCases/zAGL",
        **configs  # Unpacks all config objects
    )
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()