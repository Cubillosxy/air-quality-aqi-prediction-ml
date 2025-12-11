from src.infrastructure.air_quality_api import OpenMeteoClient
from src.application.feature_service import build_features
from src.application.training_service import run_training_and_evaluation


def run_pipeline() -> None:
    """
    End-to-end pipeline: download, build features, train models, evaluate and plot.
    """
    print("Step 1: Downloading raw data from Open-Meteo...")
    client = OpenMeteoClient()
    df_raw = client.download_and_merge()
    print(f"Raw data shape: {df_raw.shape}")

    print("Step 2: Building features...")
    df_features = build_features()
    print(f"Features shape after cleaning: {df_features.shape}")

    print("Step 3: Training and evaluation...")
    best = run_training_and_evaluation()
    print(
        f"Finished. Best model: {best.name}, "
        f"MAE={best.mae:.2f}, RMSE={best.rmse:.2f}"
    )


if __name__ == "__main__":
    run_pipeline()
