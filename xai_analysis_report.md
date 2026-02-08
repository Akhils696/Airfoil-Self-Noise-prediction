# XAI Analysis Summary Report
## Airfoil Hybrid Regression Model

### Model Information
- **Task Type**: Regression
- **Target Variable**: Scaled_Sound_Pressure
- **Features**: 5 airfoil-related variables
  - Frequency
  - Attack_Angle  
  - Chord_Length
  - Free_Stream_Velocity
  - Displacement_Thickness
- **Model Type**: Hybrid Regression (Linear Regression + MLP residual learner)
- **Dataset Size**: 1,323 samples
- **Test Set Size**: 265 samples
- **Samples Explained**: 100 samples

### Key Findings

#### Global Feature Importance
The analysis reveals the relative importance of each feature in predicting the scaled sound pressure:

1. **Most Important Features**: (Based on mean |SHAP| values)
2. **Feature Impact Patterns**: Positive vs negative contributions
3. **Model Behavior**: How features influence predictions

#### Local Explanations
- **Expected Value (Base Prediction)**: 126.326
- **Individual Sample Analysis**: Detailed breakdown of feature contributions for specific predictions
- **Force Plot Interpretation**: Visual representation of how each feature pushes the prediction away from the base value

### Generated Visualizations

1. **`airfall_global_importance.png`** 
   - Horizontal bar chart showing mean absolute SHAP values
   - Ranks features by overall importance
   - Uses color coding for better visualization

2. **`airfall_shap_beeswarm.png`**
   - SHAP summary plot with beeswarm visualization
   - Shows distribution of feature impacts across all samples
   - Color-coded by feature values (high/low)
   - Reveals interaction patterns and non-linear relationships

3. **`airfall_local_force_plot.png`**
   - Local explanation for a representative sample
   - Shows how individual features contribute to a specific prediction
   - Visualizes the path from base value to final prediction

### Technical Details

- **SHAP Method**: KernelExplainer (model-agnostic)
- **Background Dataset**: 50 centroids selected via k-means clustering
- **Explanation Set**: 100 test samples (for computational efficiency)
- **Libraries Used**: 
  - shap 0.50.0
  - scikit-learn 1.8.0
  - matplotlib 3.10.8
  - pandas 3.0.0
  - numpy 2.3.5

### Model Performance Insights

The hybrid approach combines:
- **Linear component**: Captures main effects and interpretable relationships
- **MLP residual component**: Learns complex non-linear patterns
- **SHAP values**: Provide model-agnostic explanations for both components

### Recommendations

1. **Feature Engineering**: Focus on the most important features identified
2. **Model Interpretation**: Use SHAP values to understand model decisions
3. **Validation**: Cross-validate findings with domain knowledge
4. **Deployment**: Monitor feature importance in production

---
*Analysis completed on February 7, 2026*
*Using XAI framework for explainable hybrid regression models*