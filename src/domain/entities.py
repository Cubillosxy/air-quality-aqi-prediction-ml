from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class ModelRunResult:
    """Stores the outcome of a single model run."""
    name: str
    mae: float
    rmse: float
    y_true: np.ndarray
    y_pred: np.ndarray
    feature_importances: Optional[Dict[str, float]] = None
