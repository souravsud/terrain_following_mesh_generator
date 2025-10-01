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
