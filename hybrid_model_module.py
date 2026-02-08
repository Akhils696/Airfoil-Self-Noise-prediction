"""
Hybrid Regression Model Module for Airfoil Prediction
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class HybridRegressionModel:
    """
    Hybrid Regression Model combining Linear Regression + MLP residual learner for airfoil prediction.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=(100, 50), 
            max_iter=1000, 
            random_state=42,
            early_stopping=True
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        """Train the hybrid model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train linear model
        self.linear_model.fit(X_scaled, y)
        
        # Get linear predictions
        y_linear_pred = self.linear_model.predict(X_scaled)
        
        # Compute residuals
        residuals = y - y_linear_pred
        
        # Train MLP on residuals
        self.mlp_model.fit(X_scaled, residuals)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using the hybrid model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get linear prediction
        y_linear_pred = self.linear_model.predict(X_scaled)
        
        # Get MLP residual prediction
        y_mlp_pred = self.mlp_model.predict(X_scaled)
        
        # Combine predictions
        y_final = y_linear_pred + y_mlp_pred
        
        return y_final
    
    def get_feature_importance(self):
        """Get feature importance from linear model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return np.abs(self.linear_model.coef_)