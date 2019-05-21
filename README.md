# Sequential Bayesian Experimental Design code

*TODO: dissertation reference*

This repository holds all the code necessary to run a grid search (i.e. discrete) experimental design for informing heterogeneous surface catalycity model parameters using simulated laser absorption spectroscopy.

Required packages:
1. GCC
2. Autotools
3. MPICH
4. OpenBLAS
5. PETSc
6. HDF5
7. VTK
8. Boost
9. libMesh
10. GRINS
11. GSL
12. Antioch
13. GLPK
14. QUESO

With the packages loaded into `PATH` and `LD_LIBRARY_PATH` as needed by the `Makefile`, one can type `make` in the `src/` directory to build all executables.

## General Outline ##
1. Generate surrogate using `generate_surrogate`
2. Create/modify input file for experimental design run
3. Run `grid_exp_design`

## Executables ##

### generate_surrogate ###
  - surrogate parameters: thermodynamic pressure, inlet oxygen mole fraction, furnace temperature, catalycity parameter(s)
  - arguments: dimension catalycity_model
  - example: `mpiexec -np 8 ./generate_surrogate 2 constant`
     
    Generates a surrogate for a 2D mesh with the Constant model

  - places data in `"./"+dim+"d/"+cat_model+"/surrogate_data/surr_pTXo.dat"`

### grid_exp_design ###
  - arguments: input_file metric output_prefix
  - example: `mpiexec -np 8 ./ggrid_exp_design ./expdesign_constant.in min_var test`
     
    Does a full experimental design step for the given problem. Input file specifies initial scenarios and any scenario parameters to hold fixed.

  - outputs chosen experimental scenario to the screen
