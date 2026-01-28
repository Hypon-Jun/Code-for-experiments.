## RoLL: Robust Low-Rank Learning under General Losses via Nesterov Momentum

This repository contains the code used in our experiments. It includes utility functions, dataset scripts, and the main scripts for running the proposed RoLL method on both simulated and real-world datasets.

+ \- **function/**: General utility functions supporting the algorithm implementation.  

  \- **dataset/**: Real datasets used in our experiments.  

  Main scripts:  

  \- **simulation_data_generation.py**: Generates synthetic datasets for simulation studies.  

  \- **split_noisy_data.py**: Splits real multi-label datasets and introduces label noise into the training set.  

  \- **split_real_data.py**: Splits clean multi-label datasets and flips a proportion of training labels.  

  \- **main_RoLL_simulation.py**: Runs experiments on simulated datasets and reports evaluation metrics.  

  \- **main_RoLL_emotions.py**: Runs experiments on the `emotions` dataset and reports evaluation metrics.
