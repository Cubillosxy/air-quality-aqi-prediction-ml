# src/config.py
from pathlib import Path

# Basic project paths
# __file__ = .../src/config.py
# parent    = .../src
# BASE_DIR  = project root (parent of src)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "air_quality_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "aqi_features.csv"
PLOTS_DIR = BASE_DIR / "plots"

# City and data config
CITY_NAME = "Los Angeles"
LATITUDE = 34.05
LONGITUDE = -118.25
PAST_DAYS = 60

# Train/test split
TRAIN_FRACTION = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42
