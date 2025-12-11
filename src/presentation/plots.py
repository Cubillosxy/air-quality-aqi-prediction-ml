import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PLOTS_DIR
from src.domain.entities import ModelRunResult


def _ensure_plots_dir() -> None:
    """Make sure the plots directory exists."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_predictions(
    ts_test: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """
    Plot actual vs predicted AQI over time on the test set.
    """
    _ensure_plots_dir()

    plt.figure(figsize=(12, 5))
    plt.plot(ts_test, y_true, label="Actual AQI", linewidth=1.5)
    plt.plot(ts_test, y_pred, label="Predicted AQI", linewidth=1.5)
    plt.title(f"AQI 24h Ahead Prediction - {model_name}")
    plt.xlabel("Time")
    plt.ylabel("US AQI")
    plt.legend()
    plt.tight_layout()

    out_path = PLOTS_DIR / f"aqi_pred_vs_actual_{model_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved prediction plot to {out_path}")


def plot_feature_importance(result: ModelRunResult, top_n: int = 15) -> None:
    """
    Plot top-N feature importances for a model that exposes them.
    """
    _ensure_plots_dir()

    if not result.feature_importances:
        print(f"No feature importances available for {result.name}.")
        return

    items = sorted(
        result.feature_importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    labels: List[str]
    values: List[float]
    labels, values = zip(*items)

    indices = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.barh(indices, values)
    plt.yticks(indices, labels)
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances - {result.name}")
    plt.xlabel("Importance")
    plt.tight_layout()

    out_path = PLOTS_DIR / f"feature_importance_{result.name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {out_path}")

def _bucket_weather_for_plot(row: pd.Series) -> str:
    """
    Rebuild a coarse weather category for plotting purposes.
    Uses temperature, humidity and wind to approximate conditions.
    """
    temp = row["temperature_2m"]
    humidity = row["relative_humidity_2m"]
    wind = row["wind_speed_10m"]

    if humidity >= 85 and temp < 20:
        return "foggy_or_misty"

    if wind >= 7:
        return "windy"

    if temp >= 28 and humidity <= 40:
        return "hot_dry"

    if temp <= 10:
        return "cold"

    return "mild"

def plot_aqi_by_weather(df: pd.DataFrame) -> None:
    """
    Plot average AQI per weather category using processed features.
    """
    _ensure_plots_dir()

    df = df.copy()

    # Si no existe la columna, la reconstruimos para el plot
    if "weather_category" not in df.columns:
        required_cols = {"temperature_2m", "relative_humidity_2m", "wind_speed_10m"}
        if not required_cols.issubset(df.columns):
            print(
                "Cannot compute 'weather_category' for plot. "
                "Missing temperature_2m / relative_humidity_2m / wind_speed_10m."
            )
            return

        df["weather_category"] = df.apply(_bucket_weather_for_plot, axis=1)

    if "us_aqi" not in df.columns:
        print("No 'us_aqi' column found. Skipping AQI by weather plot.")
        return

    mean_by_weather = (
        df.groupby("weather_category")["us_aqi"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(8, 5))
    mean_by_weather.plot(kind="bar")
    plt.title("Average AQI by Weather Category")
    plt.xlabel("Weather Category")
    plt.ylabel("Average AQI")
    plt.tight_layout()

    out_path = PLOTS_DIR / "aqi_by_weather_category.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved AQI by weather category plot to {out_path}")