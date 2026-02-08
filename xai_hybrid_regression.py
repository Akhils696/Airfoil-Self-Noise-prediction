import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
import pickle
from hybrid_model_module import HybridRegressionModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = 'Preprocessed_Data.xls'           # Input preprocessed tabular data
MODEL_PATH = 'hybrid_model.pkl'               # Trained hybrid regression model
TARGET_COLUMN = 'airfoil_scaled_sound_pressure'     # Target variable name
OUTPUT_DIR = 'xai_outputs'                    # Output directory for visualizations

# Output file paths
GLOBAL_IMPORTANCE_PNG = os.path.join(OUTPUT_DIR, 'airfoil_global_importance.png')
BEESWARM_PNG = os.path.join(OUTPUT_DIR, 'airfoil_shap_beeswarm.png')
FORCE_PLOT_HTML = os.path.join(OUTPUT_DIR, 'airfoil_local_force_plot.html')

# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load the preprocessed dataset from Excel file.
    Handles both .xls and .xlsx formats.
    """
    try:
        # Try loading as Excel file
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        raise
    
    # Remove rows with any missing values
    initial_rows = len(df)
    df = df.dropna(how='any').reset_index(drop=True)
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing values")
    
    return df

def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Separate input features (X) and target variable (y).
    Automatically detects target column if not found.
    """
    # Check if target column exists
    if target_col in df.columns:
        resolved_target = target_col
    else:
        # Fallback: use last numeric column as target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in dataset")
        resolved_target = numeric_cols[-1]
        print(f"Target column '{target_col}' not found. Using '{resolved_target}' as target.")
    
    # Separate features and target
    y = df[resolved_target]
    X = df.drop(columns=[resolved_target])
    
    # Ensure all features are numeric
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric feature columns found")
    
    # Align indices
    common_indices = X.index.intersection(y.index)
    X = X.loc[common_indices].reset_index(drop=True)
    y = y.loc[common_indices].reset_index(drop=True)
    
    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target variable: {resolved_target}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y, resolved_target

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model(model_path: str):
    """
    Load the trained hybrid regression model from pickle file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Verify model has predict method
    if not hasattr(model, 'predict'):
        raise TypeError("Loaded model does not have predict() method")
    
    print(f"Model loaded successfully: {type(model).__name__}")
    return model

# -----------------------------------------------------------------------------
# Background dataset selection
# -----------------------------------------------------------------------------
def select_background(X_train: pd.DataFrame, method: str = 'kmeans', k: int = 50) -> np.ndarray:
    """
    Select representative background dataset for SHAP KernelExplainer.
    
    Args:
        X_train: Training features
        method: 'kmeans' or 'random'
        k: Number of background samples
    
    Returns:
        Background dataset as numpy array
    """
    if method == 'kmeans':
        # Use k-means clustering to get representative samples
        background = shap.kmeans(X_train, k)
        print(f"Selected {k} centroids using k-means clustering")
    elif method == 'random':
        # Random sampling
        k_eff = min(k, len(X_train))
        background = X_train.sample(n=k_eff, random_state=42).to_numpy()
        print(f"Selected {k_eff} random samples")
    else:
        raise ValueError("Method must be 'kmeans' or 'random'")
    
    return background

# -----------------------------------------------------------------------------
# SHAP value computation
# -----------------------------------------------------------------------------
def compute_shap_values(model, background: np.ndarray, X_explain: pd.DataFrame) -> tuple:
    """
    Compute SHAP values using KernelExplainer.
    
    Returns:
        (explainer, shap_values, expected_value)
    """
    feature_names = X_explain.columns.tolist()
    
    # Define prediction function that maintains feature order
    def predict_fn(X: np.ndarray) -> np.ndarray:
        # Convert numpy array to DataFrame with proper column names
        X_df = pd.DataFrame(X, columns=feature_names)
        return model.predict(X_df)
    
    # Initialize KernelExplainer
    print("Initializing SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(predict_fn, background)
    
    # Compute SHAP values
    print(f"Computing SHAP values for {len(X_explain)} samples...")
    shap_values = explainer.shap_values(X_explain.to_numpy())
    
    # Get expected value (base prediction)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.array(expected_value).ravel()[0])
    
    print("SHAP values computed successfully")
    return explainer, shap_values, expected_value

# -----------------------------------------------------------------------------
# Visualization functions
# -----------------------------------------------------------------------------
def plot_global_feature_importance(shap_values: np.ndarray, feature_names: list, output_path: str = None):
    """
    Create global feature importance bar plot using mean absolute SHAP values.
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Sort features by importance
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_values = mean_abs_shap[sorted_indices]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(sorted_features)), sorted_values[::-1], 
                    color=plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))[::-1])
    plt.yticks(range(len(sorted_features)), sorted_features[::-1])
    plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    plt.title('Global Feature Importance\n(Airfoil Hybrid Regression Model)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_values[::-1])):
        plt.text(value + max(sorted_values) * 0.01, i, f'{value:.4f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Global importance plot saved to: {output_path}")
    
    plt.show()

def plot_shap_summary(shap_values: np.ndarray, X_explain: pd.DataFrame, output_path: str = None):
    """
    Create SHAP summary (beeswarm) plot.
    """
    plt.figure(figsize=(12, 8))
    
    # Create SHAP summary plot
    shap.summary_plot(
        shap_values, 
        X_explain, 
        show=False,
        plot_type='dot',
        color=plt.cm.coolwarm,
        title='SHAP Summary Plot\n(Airfoil Hybrid Regression Model)'
    )
    
    plt.title('SHAP Summary Plot\n(Airfoil Hybrid Regression Model)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to: {output_path}")
    
    plt.show()

def create_force_plot(explainer, shap_values: np.ndarray, X_explain: pd.DataFrame, 
                     sample_index: int, expected_value: float, output_path: str = None):
    """
    Create local explanation force plot for a specific sample.
    """
    # Get sample data
    sample_shap = shap_values[sample_index]
    sample_features = X_explain.iloc[sample_index]
    feature_names = X_explain.columns.tolist()
    
    # Create force plot (matplotlib version to avoid IPython dependency)
    try:
        # Try to create matplotlib force plot
        fig = plt.figure(figsize=(12, 6))
        shap.force_plot(
            expected_value,
            sample_shap,
            sample_features,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            figsize=(12, 6)
        )
        plt.title(f'Local Explanation - Sample #{sample_index}', fontsize=14, fontweight='bold')
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
            print(f"Force plot saved to: {output_path.replace('.html', '.png')}")
        
        plt.show()
        
    except Exception as e:
        print(f"Warning: Could not create force plot: {e}")
        print("Creating alternative local explanation...")
        
        # Fallback: create a simple bar plot of SHAP values for this sample
        create_local_bar_plot(sample_shap, feature_names, sample_index, expected_value, output_path)
    
    return None

def create_local_bar_plot(shap_values_sample: np.ndarray, feature_names: list, 
                    sample_index: int, expected_value: float, output_path: str = None):
    """
    Create alternative local explanation bar plot when force plot fails.
    """
    # Sort features by absolute SHAP values
    abs_shap = np.abs(shap_values_sample)
    sorted_indices = np.argsort(abs_shap)[::-1]
    
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_shap = [shap_values_sample[i] for i in sorted_indices]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color bars based on positive/negative SHAP values
    colors = ['#2E8B57' if val >= 0 else '#8B0000' for val in sorted_shap]
    bars = ax.barh(range(len(sorted_features)), sorted_shap[::-1], color=colors[::-1])
    
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features[::-1])
    ax.set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Local Feature Impact - Sample #{sample_index}\n(Base value: {expected_value:.3f})', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_shap[::-1])):
        if abs(value) > 0.01:  # Only show significant values
            ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                   va='center', fontsize=9, 
                   ha='left' if value >= 0 else 'right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path.replace('.html', '_bar.png'), dpi=300, bbox_inches='tight')
        print(f"Local bar plot saved to: {output_path.replace('.html', '_bar.png')}")
    
    plt.show()

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    """
    Main function to execute the complete XAI analysis pipeline.
    """
    print("=" * 60)
    print("EXPLAINABLE AI ANALYSIS - AIRFOIL HYBRID REGRESSION MODEL")
    print("=" * 60)
    
    try:
        # 1. Load preprocessed dataset
        print("\n1. Loading dataset...")
        df = load_data(DATA_PATH)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # 2. Separate features and target
        print("\n2. Separating features and target...")
        X, y, target_name = split_features_target(df, TARGET_COLUMN)
        
        # 3. Split into train/test sets
        print("\n3. Creating train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # 4. Load trained model
        print("\n4. Loading trained hybrid regression model...")
        model = load_model(MODEL_PATH)
        
        # 5. Select background dataset
        print("\n5. Selecting background dataset for SHAP...")
        background = select_background(X_train, method='kmeans', k=50)
        
        # 6. Compute SHAP values (limit explanation set for efficiency)
        print("\n6. Computing SHAP values...")
        n_explain = min(100, len(X_test))  # Limit for faster computation
        X_explain = X_test.iloc[:n_explain].copy()
        explainer, shap_values, expected_value = compute_shap_values(
            model, background, X_explain
        )
        
        # 7. Generate visualizations
        print("\n7. Generating visualizations...")
        
        # Global feature importance
        print("   - Creating global feature importance plot...")
        plot_global_feature_importance(
            shap_values, 
            X_explain.columns.tolist(), 
            GLOBAL_IMPORTANCE_PNG
        )
        
        # SHAP summary plot
        print("   - Creating SHAP summary plot...")
        plot_shap_summary(shap_values, X_explain, BEESWARM_PNG)
        
        # Local force plot (for first sample)
        print("   - Creating local force plot...")
        force_plot = create_force_plot(
            explainer, shap_values, X_explain, 
            sample_index=0, expected_value=expected_value,
            output_path=FORCE_PLOT_HTML
        )
        
        # 8. Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Target variable: {target_name}")
        print(f"Features analyzed: {X.shape[1]}")
        print(f"Samples explained: {n_explain}")
        print(f"Expected value (base prediction): {expected_value:.4f}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == '__main__':
    main()