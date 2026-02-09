# Airfoil XAI Analysis - Hybrid Regression Model

## Overview
This repository contains a complete Explainable AI (XAI) analysis for a hybrid regression model predicting airfoil-related phenomena. The analysis uses SHAP (SHapley Additive exPlanations) to provide model-agnostic explanations. The project compares multiple regression models and selects the best-performing model for detailed XAI analysis.

## Model Comparison Results
The project evaluates multiple regression models with the following performance metrics:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Neural Network (MLP) | 3.0150 | 2.3104 | **0.7626** |
| Linear Regression | 4.3512 | 3.4314 | 0.5056 |
| Ridge | 4.4062 | 3.5078 | 0.4931 |
| Elastic Net | 5.2637 | 4.3421 | 0.2765 |
| Lasso | 5.2675 | 4.3376 | 0.2755 |

Based on these results, the hybrid regression model (Linear + MLP residual learning) was selected for XAI analysis, combining the interpretability of linear models with the predictive power of neural networks.

## Files Included

### Main Analysis Scripts
- `xai_hybrid_regression.py` - Main XAI analysis script
- `hybrid_model_module.py` - Hybrid regression model implementation
- `model_comparison_metrics.py` - Script for comparing multiple regression models
- `create_mock_model.py` - Script to create mock trained model
- `check_xai_setup.py` - Environment verification script

### Output Files
- `xai_outputs/airfoil_global_importance.png` - Global feature importance plot
- `xai_outputs/airfoil_shap_beeswarm.png` - SHAP summary/beeswarm plot  
- `xai_outputs/airfoil_local_force_plot.png` - Local explanation plot
- `xai_analysis_report.md` - Detailed analysis report
- `metrics_summary.md` - Model comparison metrics summary
- `model_comparison_results.csv` - Raw comparison metrics data

### Configuration
- `requirements_xai.txt` - Python dependencies
- `hybrid_model.pkl` - Trained hybrid regression model (generated)

## Requirements
```
numpy>=1.21.0
pandas>=1.3.0  
scikit-learn>=1.0.0
shap>=0.40.0
matplotlib>=3.5.0
openpyxl>=3.0.0
xlrd>=2.0.0
```

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements_xai.txt
```

2. **Verify setup:**
```bash
python check_xai_setup.py
```

3. **Create mock model (if needed):**
```bash
python create_mock_model.py
```

4. **Run XAI analysis:**
```bash
python xai_hybrid_regression.py
```

## Data Description
The analysis uses preprocessed tabular data with:
- **Target variable**: `Scaled_Sound_Pressure` 
- **Features**: 5 airfoil-related variables
  - `Frequency`
  - `Attack_Angle`
  - `Chord_Length` 
  - `Free_Stream_Velocity`
  - `Displacement_Thickness`

## Model Architecture
The hybrid regression model combines:
- **Linear Regression**: For interpretable main effects
- **MLP (Multi-Layer Perceptron)**: For learning residual non-linear patterns
- **Feature scaling**: Standardization for consistent training

## XAI Methodology
- **SHAP KernelExplainer**: Model-agnostic explanation method
- **Background dataset**: 50 centroids from k-means clustering
- **Explanation set**: 100 test samples for efficient computation
- **Visualizations**: Global importance, summary plots, and local explanations

## Output Interpretation

### Global Feature Importance
Shows which features have the largest overall impact on predictions.

### SHAP Summary Plot
Reveals how feature values relate to their impact on predictions across all samples.

### Local Explanations
Explains individual predictions by showing how each feature contributes to moving from the base prediction to the final output.

## Customization
You can modify the following parameters in `xai_hybrid_regression.py`:

```python
# Configuration section
DATA_PATH = 'Preprocessed_Data.xls'     # Input data file
MODEL_PATH = 'hybrid_model.pkl'         # Trained model file
TARGET_COLUMN = 'airfall_deposition_rate'  # Target variable name
OUTPUT_DIR = 'xai_outputs'              # Output directory
```

## Troubleshooting

**Common issues:**
- Missing dependencies → Run `pip install -r requirements_xai.txt`
- File not found errors → Check file paths in configuration
- Memory issues → Reduce `n_explain` parameter in main function
- Visualization errors → Ensure matplotlib backend is properly configured

## License
This analysis framework is provided for educational and research purposes.