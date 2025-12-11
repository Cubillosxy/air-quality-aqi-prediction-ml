from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import RANDOM_SEED
from src.domain.entities import ModelRunResult
from src.infrastructure.storage import load_processed_data, time_split
from src.infrastructure.results import save_results
from src.presentation.plots import plot_predictions, plot_feature_importance, plot_aqi_by_weather

class RandomForestTrainer:
    """
    Wraps a RandomForestRegressor for AQI prediction.
    """

    def __init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def name(self) -> str:
        return "RandomForest"

    def feature_importances(self, feature_names: List[str]):
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(feature_names, self.model.feature_importances_))
        return None


class GradientBoostingTrainer:
    """
    Wraps a GradientBoostingRegressor for AQI prediction.
    """

    def __init__(self) -> None:
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_SEED,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def name(self) -> str:
        return "GradientBoosting"

    def feature_importances(self, feature_names: List[str]):
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(feature_names, self.model.feature_importances_))
        return None


def run_training_and_evaluation() -> ModelRunResult:
    """
    Train both models, compare metrics and generate plots for the best one.
    """
    df = load_processed_data()
    X_train, X_test, y_train, y_test, ts_test = time_split(df, target_col="target_aqi_24h")

    feature_names = list(X_train.columns)

    trainers = [RandomForestTrainer(), GradientBoostingTrainer()]
    results: List[ModelRunResult] = []

    for trainer in trainers:
        trainer.fit(X_train, y_train)
        y_pred = trainer.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)

        importances = trainer.feature_importances(feature_names)
        result = ModelRunResult(
            name=trainer.name(),
            mae=mae,
            rmse=rmse,
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            feature_importances=importances,
        )
        results.append(result)

        print(f"[{result.name}] MAE={result.mae:.2f}, RMSE={result.rmse:.2f}")

    # pick best by RMSE
    best = min(results, key=lambda r: r.rmse)
    print(f"Best model: {best.name}")

    save_results(
        model_name=best.name,
        mae=best.mae,
        rmse=best.rmse,
        feature_importances=best.feature_importances,
        timestamps=ts_test,
        y_true=best.y_true,
        y_pred=best.y_pred,
    )

    plot_predictions(ts_test, best.y_true, best.y_pred, best.name)
    plot_feature_importance(best)
    plot_aqi_by_weather(df)

    return best
