# MAFS: Multi-head Attention Feature Selection for High-Dimensional Data via Deep Fusion of Filter Methods

<div align="center">
  <img src="https://github.com/user-attachments/assets/8f04b682-960e-494c-9baa-05b7efba1225" alt="MAFS Framework" width="800"/>
</div>

---

## Abstract

Feature selection in biomedical data analysis confronts significant challenges: existing approaches typically employ singular strategies and exhibit inherent limitations. Filter methods demonstrate computational efficiency but fail to identify complex feature interaction patterns, while embedded methods can capture nonlinear relationships yet lack statistical prior guidance and incur high computational costs. Most existing methods cannot simultaneously leverage the interpretability of statistical approaches and the sophisticated modeling advantages of deep learning, a problem particularly pronounced in high-dimensional biomedical data.

We propose a Multi-Head Attention-based Feature Selection Framework (MAFS), a unified model that integrates statistical priors and deep learning capabilities. This framework initially employs filter methods to compute initial correlation weights, then utilizes multi-head attention mechanisms with Lasso adaptive regularization to learn complex feature interactions from multiple perspectives, ensuring statistically guided deep modeling results. Through a multi-stage dimensionality reduction strategy, the most discriminative features are selected from each attention head to construct optimal subsets.

Comprehensive experiments on multiple high-dimensional biomedical datasets, including cancer gene expression and Alzheimer's disease data, demonstrate the performance of this approach. Downstream analyses validate the potential of our method in terms of feature selection effectiveness and classification performance.

---

## Installation
```bash
git clone https://github.com/yourname/mafs.git
cd mafs
pip install -r requirements.txt
```
```bash
Bcor relies on R packages, including ball. If automatic installation fails, you can manually install them in R by running the following codes:
Rscript -e "install.packages('Ball')"
library(Ball)
```

---

## Required Argument Details
- `--data_path`: Path to .npz file
- `--y_type`:  Task type- `categorical` (classification) or `numerical` (regression)
- `--hidden_scale`: Hidden layer scale, (hidden_size = input_features / hidden_scale) [default: `200`]
- `--methods`: Filter methods for multi-head initialization [default: `sis bcor kendall`]
  - `sis`: Sure Independence Screening 
  - `bcor`: Ball-correlation Sure Independence Screening 
  - `kendall`: Kendall's Tau
- `--gamma`:  adaptive weight parameter for regularization strength adjustment [default: `0.5`]
- `--reg_lambda`:  regularization coefficient [default: `1e-5`]
---

## Examples

### Example 1: Classification (Multi-class)
```bash
python test_multi_head.py \
  --data_path data/simulation_data/data_25k_combine_categorical.npz \
  --y_type categorical \
  --hidden_scale 200 \
  --methods sis bcor kendall
  --gamma 0.5
  --reg_lambda 1e-5
```

### Example 2: Regression
```bash
python test_multi_head.py \
  --data_path data/simulation_data/data_25k_combine_numerical.npz \
  --y_type numerical \
  --hidden_scale 200 \
  --methods sis bcor kendall
  --gamma 0.5
  --reg_lambda 1e-5
```

### Example 3: Fast Mode (2 heads)
```bash
python test_multi_head.py \
  --data_path data/simulation_data/data_25k_combine_numerical.npz \
  --y_type numerical \
  --hidden_scale 200 \
  --methods sis kendall
  --gamma 0.5
  --reg_lambda 1e-5
```
