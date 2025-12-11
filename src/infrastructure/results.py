import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from ..config import BASE_DIR

RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "best_model_results.json"


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results(
    model_name: str,
    mae: float,
    rmse: float,
    feature_importances: Optional[Dict[str, float]],
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> None:
    """Persist model evaluation results + predictions into JSON."""
    _ensure_results_dir()

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "feature_importances": feature_importances,
        "timestamps": timestamps.astype(str).tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved evaluation results to: {RESULTS_FILE}")

