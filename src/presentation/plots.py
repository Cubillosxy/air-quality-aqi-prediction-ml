import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import PLOTS_DIR
from ..domain.entities import ModelRunResult


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
