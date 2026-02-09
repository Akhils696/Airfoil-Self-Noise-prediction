"""
Model Comparison Metrics for Airfoil Prediction Project
Generates comprehensive metrics for comparing multiple regression models
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path='Preprocessed_Data.xls'):
    """Load and prepare the airfoil dataset"""
    try:
        # Try to read as CSV first, then as Excel
        try:
            df = pd.read_csv(data_path)
        except:
            df = pd.read_excel(data_path)
        
        # Remove rows with missing values
        df = df.dropna(how='any').reset_index(drop=True)
        
        # Identify target column (last numeric column if specific one not found)
        target_col = 'airfoil_scaled_sound_pressure'  # Try specific name first
        if target_col not in df.columns:
            # Fall back to last numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_col = numeric_cols[-1]
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        X = X.select_dtypes(include=[np.number])  # Keep only numeric features
        
        return X, y, target_col
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    n = len(y_true)
    p = 1  # Number of features (will be updated in model evaluation)
    
    # Adjusted R²
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Adjusted R²': adj_r2,
        'MAPE': mape
    }

def evaluate_models(X_train, X_test, y_train, y_test):
    """Evaluate multiple regression models"""
    
    # Scale features for models that require it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(random_state=42, max_iter=2000),
        'Ridge': Ridge(random_state=42),
        'Elastic Net': ElasticNet(random_state=42, max_iter=2000),
        'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(100, 50), 
                                            max_iter=1000, 
                                            random_state=42, 
                                            early_stopping=True)
    }
    
    results = {}
    
    print("Evaluating models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        try:
            # Fit model
            if name in ['Linear Regression', 'Lasso', 'Ridge', 'Elastic Net']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:  # Neural Network
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)
            
            # Store results
            results[name] = metrics
            
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE:  {metrics['MAE']:.4f}")
            print(f"  R²:   {metrics['R²']:.4f}")
            print("-" * 40)
            
        except Exception as e:
            print(f"  Error evaluating {name}: {e}")
            results[name] = {'Error': str(e)}
    
    return results

def create_comparison_dataframe(results):
    """Convert results to a comparison dataframe"""
    comparison_data = []
    
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Reorder columns to put Model first
    cols = ['Model'] + [col for col in df.columns if col != 'Model']
    df = df[cols]
    
    return df

def print_ranking(df):
    """Print ranking based on key metrics"""
    print("\n" + "="*60)
    print("MODEL RANKINGS")
    print("="*60)
    
    # Rank by R² (higher is better)
    df_sorted_r2 = df.sort_values(by='R²', ascending=False)
    print("\nRanking by R² (Higher is Better):")
    for i, (_, row) in enumerate(df_sorted_r2.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['R²']:.4f}")
    
    # Rank by RMSE (lower is better)
    df_sorted_rmse = df.sort_values(by='RMSE', ascending=True)
    print("\nRanking by RMSE (Lower is Better):")
    for i, (_, row) in enumerate(df_sorted_rmse.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['RMSE']:.4f}")
    
    # Rank by MAE (lower is better)
    df_sorted_mae = df.sort_values(by='MAE', ascending=True)
    print("\nRanking by MAE (Lower is Better):")
    for i, (_, row) in enumerate(df_sorted_mae.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['MAE']:.4f}")

def main():
    """Main function to run model comparison"""
    print("Airfoil Prediction Model Comparison")
    print("="*60)
    
    try:
        # Load and prepare data
        print("Loading data...")
        X, y, target_col = load_and_prepare_data()
        print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target variable: {target_col}")
        print(f"Feature columns: {list(X.columns)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Evaluate models
        results = evaluate_models(X_train, X_test, y_train, y_test)
        
        # Create comparison dataframe
        comparison_df = create_comparison_dataframe(results)
        
        # Print comparison table
        print("\n" + "="*60)
        print("MODEL COMPARISON TABLE")
        print("="*60)
        print(comparison_df.round(4).to_string(index=False))
        
        # Print rankings
        print_ranking(comparison_df)
        
        # Save results
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        print(f"\nResults saved to 'model_comparison_results.csv'")
        
        # Identify best model
        best_r2_idx = comparison_df['R²'].idxmax()
        best_model_r2 = comparison_df.loc[best_r2_idx, 'Model']
        print(f"\nBest model based on R²: {best_model_r2}")
        
        return comparison_df
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == '__main__':
    main()