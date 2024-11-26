# Calibrated Mixup: A Solution for Imbalanced Regression

This repository provides the implementation of **Calibrated Mixup**, an approach designed to tackle the challenges of imbalanced regression by combining data augmentation, calibration techniques, and loss function optimization.

---

## Overview

Imbalanced regression datasets often lead to biased predictions and reduced model performance. **Calibrated Mixup** addresses these challenges through:
- **Nearest Cell (NC)**: A binning-based approach to generate synthetic data.
- **Feature Similarity (FS)**: A feature-space similarity method using a Siamese Network.

Key benefits:
- Handles heterogeneous tabular datasets.
- Preserves statistical properties (mean and variance) of the data.
- Computationally efficient for real-world applications.

---

## Repository Structure

- **`Calibrated Mixup_final.ipynb`**: Complete implementation with multiple datasets and results.
- **`Calibrated Mixup_Demo_Abalone.ipynb`**: Simplified demo using the Abalone dataset.
- **`helper_functions.py`**: Utility functions for preprocessing, metrics, and visualization.

---

## Getting Started

### Prerequisites

Install the required Python packages:
```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn seaborn scipy
```

## Parameter Settings

### Key Parameters
Below are the primary parameters used in the Calibrated Mixup approach. Adjustments to these parameters can be made to fine-tune the model for specific datasets or requirements.

- **`tau`**: Temperature parameter for the SNN loss. Default: `2.45`.
- **`lr`**: Learning rate for the optimization algorithm (Adam optimizer). Default: `0.001`.
- **`num_epochs`**: The number of training epochs for the model. Default: `100`.
- **`batch_size`**: Size of the mini-batch for training. Default: `64`.
- **`k`**: Number of bins for the Nearest Cell (NC) approach. Calculated as \( K = \sqrt{n} \), where \( n \) is the number of samples.
- **`gamma`**: Multiplicative factor to control the number of synthetic samples generated in the NC and FS approaches. Default: `1.5`.

### Additional Notes:
1. **SNN Loss Parameter**:
   - The SNN loss requires careful tuning of \( \tau \) to ensure effective separation between good-quality and poor-quality synthetic samples.
   
2. **Binning Strategy**:
   - Binning employs the square root choice \( K = \sqrt{n} \) for computational efficiency and to preserve the underlying data distribution.
   
3. **Synthetic Sample Calibration**:
   - Synthetic data weights are iteratively optimized using calibration constraints to align with original data distributions (see the calibration section in the paper).

Refer to the accompanying notebooks for detailed parameter definitions and implementation.


