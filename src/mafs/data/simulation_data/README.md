## Dataset Overview
This collection contains one simulated datasets with varying sample sizes and feature dimensions, designed to evaluate feature selection methods under different scenarios.

## Available Datasets

### Small Sample Size (n=500)

#### 1. data_500_25k_combine_numerical.npz
- **Samples**: 500
- **Features**: 25,963
- **Informative Features**: 48 


## Data Structure

Each `.npz` file contains the following arrays:

### Core Data
- **`X`**: Feature matrix
- **`Y`**: Target variable Continuous response variable (regression)
- **`features`**: Column indices of informative features in X 
- **`variables`**: Groups of features forming casual variables
- **`coefficients`**: Coefficient for each casual variable
- **`types`**: Functional form of each variable's relationship with 

