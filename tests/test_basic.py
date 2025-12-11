import sys
from pathlib import Path

# Ensure src is on path for imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd
import numpy as np

from application.feature_service import bucket_weather
from infrastructure.storage import time_split
from domain.entities import ModelRunResult


def test_bucket_weather_cases():
    # foggy_or_misty: humidity >=85 and temp < 20
    s = pd.Series({"temperature_2m": 15, "relative_humidity_2m": 90, "wind_speed_10m": 1})
    assert bucket_weather(s) == "foggy_or_misty"

    # windy: wind >= 7
    s = pd.Series({"temperature_2m": 25, "relative_humidity_2m": 50, "wind_speed_10m": 8})
    assert bucket_weather(s) == "windy"

    # hot_dry: temp >=28 and humidity <=40
    s = pd.Series({"temperature_2m": 29, "relative_humidity_2m": 40, "wind_speed_10m": 2})
    assert bucket_weather(s) == "hot_dry"

    # cold: temp <=10
    s = pd.Series({"temperature_2m": 5, "relative_humidity_2m": 50, "wind_speed_10m": 1})
    assert bucket_weather(s) == "cold"

    # mild: none of the above
    s = pd.Series({"temperature_2m": 20, "relative_humidity_2m": 50, "wind_speed_10m": 1})
    assert bucket_weather(s) == "mild"

    print("All bucket_weather test cases passed.")


def test_time_split_basic():
    from datetime import timedelta

    n = 10
    timestamps = [pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i) for i in range(n)]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "city": ["X"] * n,
        "target": list(range(n)),
        "feat1": np.arange(n),
    })

    X_train, X_test, y_train, y_test, ts_test = time_split(df, "target")

    # TRAIN_FRACTION is 0.8 in config.py -> 8 training rows
    assert len(X_train) == int(n * 0.8)
    assert len(X_test) == n - int(n * 0.8)

    assert list(y_train) == list(range(0, 8))
    assert list(y_test) == list(range(8, 10))
    assert len(ts_test) == len(X_test)

    print("All time_split basic test cases passed.")


def test_model_run_result_dataclass():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    m = ModelRunResult(name="test", mae=0.15, rmse=0.2, y_true=y_true, y_pred=y_pred)

    assert m.name == "test"
    assert abs(m.mae - 0.15) < 1e-9
    assert (m.y_true == y_true).all()
    assert (m.y_pred == y_pred).all()
    assert m.feature_importances is None
    print("All ModelRunResult dataclass test cases passed.")


if __name__ == "__main__":
    # Basic runner for environments without pytest
    print("Running basic tests...")
    test_bucket_weather_cases()
    test_time_split_basic()
    test_model_run_result_dataclass()
    print("All basic tests passed")
