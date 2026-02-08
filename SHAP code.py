import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = 'Preprocessed_Data.xls'           # Input preprocessed tabular data
MODEL_PATH = 'hybrid_model.pkl'               # Trained hybrid regression model (Linear Regression + MLP residual)
TARGET_COLUMN = 'airfoil_scaled_sound_pressure'     # Placeholder target name; will fall back if not found

# Output paths for generated visualizations
OUTPUT_DIR = 'xai_outputs'
GLOBAL_IMPORTANCE_PNG = os.path.join(OUTPUT_DIR, 'airfoil_global_importance.png')
BEESWARM_PNG = os.path.join(OUTPUT_DIR, 'airfoil_shap_beeswarm.png')
FORCE_PLOT_HTML = os.path.join(OUTPUT_DIR, 'airfoil_local_force_plot.html')


# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load the preprocessed dataset from Excel (.xls or .xlsx).
    """
    df = pd.read_excel(path)
    # Ensure we only keep rows with fully available features and target
    df = df.dropna(how='any').reset_index(drop=True)
    return df


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Separate input features (X) and target (y).
    If the specified target column is not present, fall back to the last numeric column.
    Returns X, y, and the resolved target column name.
    """
    resolved_target = target_col
    if target_col not in df.columns:
        # Fallback: choose last numeric column as target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns available to use as target.")
        resolved_target = numeric_cols[-1]

    y = df[resolved_target]
    X = df.drop(columns=[resolved_target])

    # Optional: ensure all remaining features are numeric for SHAP KernelExplainer
    X = X.apply(pd.to_numeric, errors='coerce').dropna(how='any').reset_index(drop=True)
    y = y.loc[X.index].reset_index(drop=True)
    return X, y, resolved_target


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model(model_path: str):
    """
    Load the trained hybrid regression model using pandas read_pickle
    (compatible with sklearn models serialized via pickle).
    """
    model = pd.read_pickle(model_path)
    # Sanity check: ensure model follows sklearn-like API
    if not hasattr(model, 'predict'):
        raise TypeError("Loaded object does not implement .predict().")
    return model


# -----------------------------------------------------------------------------
# Background selection for KernelExplainer
# -----------------------------------------------------------------------------
def select_background(X_train: pd.DataFrame, method: str = 'kmeans', k: int = 50, random_state: int = 42) -> np.ndarray:
    """
    Select representative background dataset for SHAP KernelExplainer.
    - method='kmeans' uses shap.kmeans to summarize background as k centroids.
    - method='random' samples k random rows from X_train.
    Returns a numpy array background matrix of shape (k, n_features).
    """
    if method == 'kmeans':
        # shap.kmeans summarizes the dataset into k centroids for KernelExplainer background
        background = shap.kmeans(X_train, k)
    elif method == 'random':
        k_eff = min(k, len(X_train))
        background = X_train.sample(n=k_eff, random_state=random_state).to_numpy()
    else:
        raise ValueError("Unsupported background selection method. Use 'kmeans' or 'random'.")
    return background


# -----------------------------------------------------------------------------
# SHAP computation
# -----------------------------------------------------------------------------
def compute_kernel_shap(model, X_background: np.ndarray, X_explain: pd.DataFrame, nsamples: str | int = 'auto') -> tuple[shap.KernelExplainer, np.ndarray, float]:
    """
    Compute SHAP values using KernelExplainer for a regression model.
    Returns (explainer, shap_values, expected_value).
    """
    feature_names = X_explain.columns.tolist()

    # Prediction function that preserves feature order using a DataFrame
    def f_predict(x: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame(x, columns=feature_names)
        return model.predict(x_df)

    # Initialize KernelExplainer with the background data
    explainer = shap.KernelExplainer(f_predict, X_background)

    # Compute SHAP values for the subset to explain
    shap_values = explainer.shap_values(X_explain.to_numpy(), nsamples=nsamples)

    # Expected value (base value) for regression is scalar; handle list/array case safely
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.array(expected_value).ravel()[0])

    return explainer, shap_values, expected_value


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def plot_global_importance(shap_values: np.ndarray, feature_names: list[str], output_path: str | None = None):
    """
    Global feature importance using mean absolute SHAP values.
    Saves a horizontal bar plot and shows it.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    ordered_names = [feature_names[i] for i in order]
    ordered_values = mean_abs_shap[order]

    plt.figure(figsize=(10, 6))
    plt.barh(ordered_names[::-1], ordered_values[::-1], color='#2E86C1')
    plt.xlabel('Mean |SHAP| (impact on model output)')
    plt.title('Global Feature Importance (Airfoil)')
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_beeswarm(shap_values: np.ndarray, X_explain: pd.DataFrame, output_path: str | None = None):
    """
    SHAP summary (beeswarm) plot for global distribution of feature impacts.
    Saves the plot and shows it.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain, show=False, plot_type='dot')
    plt.title('SHAP Summary (Beeswarm) - Airfoil')
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_local_force(explainer: shap.KernelExplainer,
                     shap_values: np.ndarray,
                     X_explain: pd.DataFrame,
                     sample_index: int,
                     expected_value: float,
                     output_html_path: str | None = None):
    """
    Local explanation force plot for a single representative sample.
    Saves as an interactive HTML using shap.save_html.
    """
    shap.initjs()
    # Extract SHAP values for the chosen sample
    sv_row = shap_values[sample_index]
    x_row = X_explain.iloc[sample_index]
    feature_names = X_explain.columns.tolist()

    # Generate JS force plot
    force_plot = shap.force_plot(expected_value, sv_row, x_row, matplotlib=False, feature_names=feature_names)

    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        shap.save_html(output_html_path, force_plot)


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    # 1. Load preprocessed dataset
    df = load_data(DATA_PATH)

    # 2. Separate input features (X) and target (y)
    X, y, resolved_target = split_features_target(df, TARGET_COLUMN)

    # 3. Train/test split (no retraining; only to select test subset to explain)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Load trained hybrid regression model (Linear Regression + MLP residual learner)
    model = load_model(MODEL_PATH)

    # 5. Select representative background dataset (k-means summarization)
    background = select_background(X_train, method='kmeans', k=50, random_state=42)

    # 6. Compute SHAP values for a test subset (limit to manageable size)
    n_explain = min(200, len(X_test))
    X_explain = X_test.iloc[:n_explain].copy()
    explainer, shap_values, expected_value = compute_kernel_shap(
        model, background, X_explain, nsamples='auto'
    )

    # 7a. Global feature importance bar plot
    plot_global_importance(shap_values, X_explain.columns.tolist(), GLOBAL_IMPORTANCE_PNG)

    # 7b. SHAP summary (beeswarm) plot
    plot_beeswarm(shap_values, X_explain, BEESWARM_PNG)

    # 7c. Local explanation force plot for one representative sample (first sample)
    plot_local_force(
        explainer=explainer,
        shap_values=shap_values,
        X_explain=X_explain,
        sample_index=0,
        expected_value=expected_value,
        output_html_path=FORCE_PLOT_HTML
    )

    # Print brief run info to aid reproducibility
    print(f"Resolved target column: {resolved_target}")
    print(f"Explained {n_explain} samples from test set.")
    print(f"Saved plots to: {OUTPUT_DIR}")
    print(f"Force plot HTML: {FORCE_PLOT_HTML}")


if __name__ == '__main__':
    main()