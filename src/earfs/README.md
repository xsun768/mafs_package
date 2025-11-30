## Required Argument Details
- `--data_path`: Path to .npz file
- `--y_type`: Task type - `categorical` (classification) or `numerical` (regression)
- `--hidden_scale`: Hidden layer scale, (hidden_size = input_features / hidden_scale) [default: `200`]
- `--methods`: Filter methods for multi-head initialization [default: `sis bcor kendall`]
  - `sis`: Sure Independence Screening 
  - `bcor`: Ball-correlation Sure Independence Screening 
  - `kendall`: Kendall's Tau
- `--reg_lambda`: Regularization coefficient [default: `1e-5`]

---

## Examples

### Example 1: Classification (Multi-class)
```bash
python test_multi_head_earfs.py \
  --data_path data/simulation_data/data_25k_combine_categorical.npz \
  --y_type categorical \
  --hidden_scale 200 \
  --methods sis bcor kendall \
  --reg_lambda 1e-5
```

### Example 2: Regression
```bash
python test_multi_head_earfs.py \
  --data_path data/simulation_data/data_25k_combine_numerical.npz \
  --y_type numerical \
  --hidden_scale 200 \
  --methods sis bcor kendall \
  --reg_lambda 1e-5
```

### Example 3: Fast Mode (2 heads)
```bash
python test_multi_head_earfs.py \
  --data_path data/simulation_data/data_25k_combine_numerical.npz \
  --y_type numerical \
  --hidden_scale 200 \
  --methods sis kendall \
  --reg_lambda 1e-5
```
