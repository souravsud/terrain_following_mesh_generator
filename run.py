import terrain_mesh as tm

def main():
    # Load all configurations from YAML file
    configs = tm.load_config("terrain_config.yaml")
    
    # Run pipeline with loaded configs
    pipeline = tm.TerrainMeshPipeline()
    results = pipeline.run(
        dem_path="/home/ssudhakaran/sourav_files/6_OpenFOAM/backup/terrain_0001_glo_30_N39_711_W007_735_50km.tif",
        output_dir="/home/ssudhakaran/sourav_files/6_OpenFOAM/meshStructured",
        **configs  # Unpacks all config objects
    )
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()