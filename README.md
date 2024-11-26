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

