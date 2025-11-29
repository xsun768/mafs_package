## Dataset Overview
This collection contains 6 simulated datasets with varying sample sizes and feature dimensions, designed to evaluate feature selection methods under different scenarios.

## Available Datasets

### Small Sample Size (n=500)

#### 1. data_500_25k_combine_numerical.npz
- **Samples**: 500
- **Features**: 25,963
- **Informative Features**: 48 

#### 2. data_500_50k_combine_numerical.npz
- **Samples**: 500
- **Features**: ~51,926
- **Informative Features**: 48

#### 3. data_500_100k_combine_numerical.npz
- **Samples**: 500
- **Features**: ~103,852
- **Informative Features**: 48

### Large Sample Size (n=2000)

#### 4. data_2k_25k_combine_numerical.npz
- **Samples**: 2,000
- **Features**: 25,963
- **Informative Features**: 48 

#### 5. data_2k_50k_combine_numerical.npz
- **Samples**: 2,000
- **Features**: ~51,926
- **Informative Features**: 48

#### 6. data_2k_100k_combine_numerical.npz
- **Samples**: 2,000
- **Features**: ~103,852
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

