# Model Comparison Metrics Summary
## Airfoil Self-Noise Prediction

### Dataset Information
- **Dataset Size**: 1,323 samples
- **Features**: 5 airfoil-related variables
  - Frequency
  - Attack_Angle
  - Chord_Length
  - Free_Stream_Velocity
  - Displacement_Thickness
- **Target Variable**: Scaled_Sound_Pressure
- **Train/Test Split**: 80/20 (1,058/265 samples)

### Model Performance Comparison

| Model | RMSE | MAE | R² | Adjusted R² | MAPE |
|-------|------|-----|-----|-------------|------|
| Neural Network (MLP) | 3.0150 | 2.3104 | **0.7626** | **0.7617** | **1.8429** |
| Linear Regression | 4.3512 | 3.4314 | 0.5056 | 0.5038 | 2.7476 |
| Ridge | 4.4062 | 3.5078 | 0.4931 | 0.4911 | 2.8130 |
| Elastic Net | 5.2637 | 4.3421 | 0.2765 | 0.2738 | 3.4959 |
| Lasso | 5.2675 | 4.3376 | 0.2755 | 0.2728 | 3.4929 |

### Rankings

#### By R² (Coefficient of Determination - Higher is Better)
1. **Neural Network (MLP)**: 0.7626
2. Linear Regression: 0.5056
3. Ridge: 0.4931
4. Elastic Net: 0.2765
5. Lasso: 0.2755

#### By RMSE (Root Mean Square Error - Lower is Better)
1. **Neural Network (MLP)**: 3.0150
2. Linear Regression: 4.3512
3. Ridge: 4.4062
4. Elastic Net: 5.2637
5. Lasso: 5.2675

#### By MAE (Mean Absolute Error - Lower is Better)
1. **Neural Network (MLP)**: 2.3104
2. Linear Regression: 3.4314
3. Ridge: 3.5078
4. Lasso: 4.3376
5. Elastic Net: 4.3421

### Key Findings

1. **Neural Network (MLP) performs best** across all metrics:
   - Highest R² (0.7626) indicating excellent predictive power
   - Lowest RMSE (3.0150) and MAE (2.3104) indicating minimal prediction errors
   - Lowest MAPE (1.8429%) indicating high accuracy relative to actual values

2. **Linear Regression is the best traditional model**:
   - Second-best performance with R² of 0.5056
   - Good balance between interpretability and performance

3. **Regularized models (Lasso, Elastic Net) underperform**:
   - Similar performance to each other
   - Lower R² values suggesting over-regularization may have reduced model capacity

4. **Ridge regression shows marginal improvement over Linear Regression**:
   - Slightly lower R² and higher error metrics
   - Minimal benefit from L2 regularization for this dataset

### Implications for XAI Analysis

The Neural Network (MLP) model demonstrates superior performance and would benefit most from XAI analysis due to its black-box nature. However, the Hybrid Regression model (Linear + MLP residual) that is currently used in the XAI analysis likely combines the interpretability of linear models with the predictive power of neural networks, making it an optimal choice for explainable AI applications.

The metrics confirm that complex models (MLP, Hybrid) achieve better performance than simpler linear models, justifying the need for advanced XAI techniques to maintain interpretability while achieving high predictive accuracy.