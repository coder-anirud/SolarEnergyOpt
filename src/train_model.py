# train_vmpp_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import argparse
import os

# ----------------------
# Config
# ----------------------
CSV_PATH = "solar_log_labeled.csv"   # Path to your log + labels CSV
SAMPLE_DT_SEC = 0.02                 # e.g., 50 Hz logging -> 0.02 sec
WINDOW_SEC = 1.0                     # 1-second rolling window for stats
W = int(WINDOW_SEC / SAMPLE_DT_SEC)

# ----------------------
# Helper: build features from raw data
# ----------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # assumes df has V, I, duty columns
    df = df.copy()
    df["P"] = df["V"] * df["I"]

    # Derivatives (delta from previous sample)
    df["dV"] = df["V"].diff().fillna(0.0)
    df["dI"] = df["I"].diff().fillna(0.0)
    df["dP"] = df["P"].diff().fillna(0.0)

    # Rolling stats over last W samples
    # Use min_periods=W so we only get stats when there is enough data
    df["P_mean"] = df["P"].rolling(W, min_periods=W).mean()
    df["P_std"]  = df["P"].rolling(W, min_periods=W).std()

    df["I_mean"] = df["I"].rolling(W, min_periods=W).mean()
    df["I_std"]  = df["I"].rolling(W, min_periods=W).std()

    df["dP_mean"] = df["dP"].rolling(W, min_periods=W).mean()
    df["dP_std"]  = df["dP"].rolling(W, min_periods=W).std()

    # Drop rows without full window
    df = df.dropna().reset_index(drop=True)
    return df

# ----------------------
# Main
# ----------------------
def main(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load data
    df_raw = pd.read_csv(csv_path)
    df = build_features(df_raw)

    # Only keep rows where we have a true label
    # If you forward-fill labels, you might choose only rows at label times.
    # Here we assume V_mpp_label exists on rows where we've done sweep
    if "V_mpp_label" not in df.columns:
        raise ValueError("CSV must contain 'V_mpp_label' column for supervised training.")

    df = df.dropna(subset=["V_mpp_label"]).reset_index(drop=True)

    # Feature columns to use
    feature_cols = [
        "V", "I", "P", "dV", "dI", "dP", "duty",
        "P_mean", "P_std", "I_mean", "I_std",
        "dP_mean", "dP_std"
    ]

    X = df[feature_cols].astype(np.float32).values
    y = df["V_mpp_label"].astype(np.float32).values

    # ----------------------
    # Train/test split (time-based, NOT random shuffle)
    # ----------------------
    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ----------------------
    # Scaling
    # ----------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ----------------------
    # Baseline model: Ridge regression
    # ----------------------
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    pred_ridge = ridge.predict(X_test_s)
    mae_ridge = mean_absolute_error(y_test, pred_ridge)
    r2_ridge = r2_score(y_test, pred_ridge)

    print("Baseline Ridge regression")
    print(f"MAE: {mae_ridge:.4f} V")
    print(f"R^2: {r2_ridge:.4f}")

    # ----------------------
    # Tiny MLP model
    # ----------------------
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20
    )
    mlp.fit(X_train_s, y_train)
    pred_mlp = mlp.predict(X_test_s)
    mae_mlp = mean_absolute_error(y_test, pred_mlp)
    r2_mlp = r2_score(y_test, pred_mlp)

    print("\nTiny MLP regression")
    print(f"MAE: {mae_mlp:.4f} V")
    print(f"R^2: {r2_mlp:.4f}")

    # ----------------------
    # Save scaler + best model (choose MLP if better, else Ridge)
    # ----------------------
    best_model = mlp if mae_mlp < mae_ridge else ridge
    model_name = "vmpp_model_mlp" if mae_mlp < mae_ridge else "vmpp_model_ridge"
    artifact = {
        "scaler": scaler,
        "model": best_model,
        "feature_cols": feature_cols,
        "model_type": "mlp" if mae_mlp < mae_ridge else "ridge"
    }
    joblib.dump(artifact, f"{model_name}.joblib")
    print(f"\nSaved model to {model_name}.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model for predicting Vmpp")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to labeled CSV")
    args = parser.parse_args()
    main(args.csv)