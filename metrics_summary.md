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
| Hybrid (Lin+MLP) | 1.5826 | 1.1525 | **0.9346** | **0.9344** | **0.9199** |
| Neural Network (MLP) | 3.0150 | 2.3104 | 0.7626 | 0.7617 | 1.8429 |
| Linear Regression | 4.3512 | 3.4314 | 0.5056 | 0.5038 | 2.7476 |
| Ridge | 4.4062 | 3.5078 | 0.4931 | 0.4911 | 2.8130 |
| Elastic Net | 5.2637 | 4.3421 | 0.2765 | 0.2738 | 3.4959 |
| Lasso | 5.2675 | 4.3376 | 0.2755 | 0.2728 | 3.4929 |

### Rankings

#### By R² (Coefficient of Determination - Higher is Better)
1. **Hybrid (Lin+MLP)**: 0.9346
2. Neural Network (MLP): 0.7626
3. Linear Regression: 0.5056
4. Ridge: 0.4931
5. Elastic Net: 0.2765
6. Lasso: 0.2755

#### By RMSE (Root Mean Square Error - Lower is Better)
1. **Hybrid (Lin+MLP)**: 1.5826
2. Neural Network (MLP): 3.0150
3. Linear Regression: 4.3512
4. Ridge: 4.4062
5. Elastic Net: 5.2637
6. Lasso: 5.2675

#### By MAE (Mean Absolute Error - Lower is Better)
1. **Hybrid (Lin+MLP)**: 1.1525
2. Neural Network (MLP): 2.3104
3. Linear Regression: 3.4314
4. Ridge: 3.5078
5. Lasso: 4.3376
6. Elastic Net: 4.3421

### Key Findings

1. **Hybrid Model performs best** across all metrics:
   - Highest R² (0.9346) indicating exceptional predictive power
   - Lowest RMSE (1.5826) and MAE (1.1525) indicating minimal prediction errors
   - Lowest MAPE (0.9199%) indicating high accuracy relative to actual values

2. **Neural Network (MLP) is second best**:
   - Strong performance with R² of 0.7626
   - Good balance of accuracy and generalization

3. **Linear Regression is the best traditional model**:
   - Third-best performance with R² of 0.5056
   - Good balance between interpretability and performance

4. **Regularized models (Lasso, Elastic Net) underperform**:
   - Similar performance to each other
   - Lower R² values suggesting over-regularization may have reduced model capacity

5. **Ridge regression shows marginal improvement over Linear Regression**:
   - Slightly lower R² and higher error metrics
   - Minimal benefit from L2 regularization for this dataset

### Implications for XAI Analysis

The Hybrid model (Linear + MLP residual) demonstrates exceptional performance, confirming its selection for XAI analysis. This model combines the interpretability of linear models with the predictive power of neural networks, making it ideal for explainable AI applications.

The significant performance gap between the hybrid model (R²=0.9346) and the next best model (Neural Network at R²=0.7626) validates the choice to focus XAI analysis on the hybrid model, as it provides both superior performance and the need for explainability due to its complex architecture.