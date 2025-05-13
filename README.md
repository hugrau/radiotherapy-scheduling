# Radiotherapy scheduling problem research

## Informations
---

In this repo can be found the code used for experiments relative to the paper #4388 submitted to ECAI 2025.

## How to run 
 ---

Simualtion files with parameters are pre-loaded, all you need to do is to run :
```
python simulations/simulate_over_horizon.py
```

By default, the solver time limit is set to 3600s for each scheduling problem. Given that one simulation is around 30 days of scheduling, running one simulation can take up to 30h.
The results we obtained can be found in results folder, and took multiple days of runtime over multiple CPUs.



