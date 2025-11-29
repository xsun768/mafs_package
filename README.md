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
pip install git+https://github.com/xsun768/mafs_package.git
```
