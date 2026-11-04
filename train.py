"""House Price Prediction - training script.

Trains several regression models on the California Housing dataset,
compares them with cross-validation, and saves the best one.

Usage:
    python train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")


def load_data() -> pd.DataFrame:
    """Load California Housing data. Target `MedHouseVal` is in units of $100,000."""
    data = fetch_california_housing(as_frame=True)
    return data.frame


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature engineering."""
    df = df.copy()
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"].clip(lower=1e-3)
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"].clip(lower=1e-3)
    return df


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    df = add_features(load_data())
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "LinearRegression": Pipeline([("scale", StandardScaler()), ("m", LinearRegression())]),
        "Ridge": Pipeline([("scale", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE
        ),
    }

    results = {}
    best_name, best_cv = None, -np.inf
    for name, model in candidates.items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1).mean()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics["CV_R2"] = float(cv)
        results[name] = metrics
        print(f"{name:18s} CV R2={cv:.3f} | test R2={metrics['R2']:.3f} "
              f"RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f}")
        if cv > best_cv:
            best_name, best_cv = name, cv

    best_model = candidates[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump({"model": best_model, "features": list(X.columns)}, MODELS_DIR / "best_model.joblib")
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    # Plot: actual vs predicted
    pred = best_model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.3, s=8)
    lims = [0, 5.2]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual price ($100k)")
    plt.ylabel("Predicted price ($100k)")
    plt.title(f"{best_name}: actual vs predicted")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", dpi=120)
    plt.close()

    # Plot: feature importance (tree models only)
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
        plt.figure(figsize=(7, 5))
        imp.plot.barh()
        plt.title("Feature importance")
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=120)
        plt.close()

    print(f"Saved model to {MODELS_DIR/'best_model.joblib'} and plots/metrics to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
