# Radiotherapy scheduling under patient arrival uncertainty

## Informations

In this repository can be found the code used for experiments relative to the paper #4388 submitted to ECAI 2025, named "Radiotherapy scheduling under patient arrival uncertainty".  

Our CP model is in `models`  under the name `cp_sat_2_stages_extensive.py`.

## How to run 
 
You first need to setup a virtual environment or install the packages in requirements.txt.

Simualtion files with parameters are pre-loaded, all you need to do is to run :
```
python simulations/simulate_over_horizon.py path_to_config_file/config_file.toml
```

By default, the solver time limit is set to 3600s for each scheduling problem. Given that one simulation is around 30 days of scheduling, running one simulation can take up to 30h. Obtaining all the results took hundreds of computational hours over multiple CPUs.  
The overall results table can be found in `simulations\simulations_res.csv`.  All the plots in the article were made from this file.  
The detailed results are in `results` folder.



