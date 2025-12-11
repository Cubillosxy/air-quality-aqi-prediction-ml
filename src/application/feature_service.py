import numpy as np
import pandas as pd

from src.infrastructure.storage import load_raw_data, save_processed_data




def bucket_weather(row: pd.Series) -> str:
    """
    Bucket raw weather variables into coarse weather categories.
    This replaces the need for an external 'weathercode' column.
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


def build_features() -> pd.DataFrame:
    """
    Build time-based, rolling and weather features and define target AQI 24h ahead.
    """
    df = load_raw_data()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Rolling features
    rolling_windows = [3, 6, 24]
    for w in rolling_windows:
        df[f"us_aqi_roll_{w}h"] = df["us_aqi"].rolling(window=w).mean()
        df[f"pm2_5_roll_{w}h"] = df["pm2_5"].rolling(window=w).mean()
        df[f"pm10_roll_{w}h"] = df["pm10"].rolling(window=w).mean()

    # Weather-based features
    df["temp_change_3h"] = df["temperature_2m"].diff(3)
    df["humidity_roll_6h"] = df["relative_humidity_2m"].rolling(window=6).mean()
    df["wind_roll_6h"] = df["wind_speed_10m"].rolling(window=6).mean()

    # Weathercode to categorical
    df["weather_category"] = df.apply(bucket_weather, axis=1)
    weather_dummies = pd.get_dummies(df["weather_category"], prefix="weather", drop_first=True)
    df = pd.concat([df, weather_dummies], axis=1)
    df = df.drop(columns=["weather_category"])

    # Target: AQI 24h ahead
    df["target_aqi_24h"] = df["us_aqi"].shift(-24)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    save_processed_data(df)
    return df
