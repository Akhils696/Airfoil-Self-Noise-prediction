"""
Create a mock hybrid regression model for demonstration purposes.
This creates a hybrid model combining Linear Regression and MLP residual learner for airfoil prediction.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

from hybrid_model_module import HybridRegressionModel

def create_mock_hybrid_model():
    """
    Create and save a mock hybrid regression model using the actual data.
    """
    print("Creating mock hybrid regression model...")
    
    # Load the actual data
    try:
        df = pd.read_csv('Preprocessed_Data.xls')
        df = df.dropna()
        
        # Identify target column (last numeric column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = numeric_cols[-1]
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        X = X.select_dtypes(include=[np.number])  # Keep only numeric features
        
        print(f"Data shape: {X.shape}")
        print(f"Target variable: {target_col}")
        print(f"Feature columns: {list(X.columns)}")
        
        # Create and train hybrid model
        model = HybridRegressionModel()
        model.fit(X, y)
        
        # Save model
        with open('hybrid_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print("✓ Hybrid model created and saved as 'hybrid_model.pkl'")
        
        # Test prediction
        sample_pred = model.predict(X.iloc[:5])
        print(f"Sample predictions: {sample_pred}")
        
        return model
        
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

if __name__ == '__main__':
    create_mock_hybrid_model()