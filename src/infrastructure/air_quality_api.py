import os
from typing import Optional

import pandas as pd
import requests

from src.config import (
    LATITUDE,
    LONGITUDE,
    PAST_DAYS,
    CITY_NAME,
    RAW_DATA_PATH,
)


class OpenMeteoClient:
    """
    Simple HTTP client for Open-Meteo air quality + weather.
    """

    AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        latitude: float = LATITUDE,
        longitude: float = LONGITUDE,
        city: str = CITY_NAME,
        past_days: int = PAST_DAYS,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.city = city
        self.past_days = past_days

    def fetch_air_quality(self) -> pd.DataFrame:
        """Fetch hourly air quality data."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "auto",
            "past_days": self.past_days,
            "hourly": [
                "us_aqi",
                "pm10",
                "pm2_5",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
            ],
        }
        resp = requests.get(self.AIR_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        df = pd.DataFrame(data)
        df.rename(columns={"time": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["city"] = self.city
        return df

    def fetch_weather(self) -> pd.DataFrame:
        """Fetch hourly weather data."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "auto",
            "past_days": self.past_days,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
            ],
        }
        resp = requests.get(self.WEATHER_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        df = pd.DataFrame(data)
        df.rename(columns={"time": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def download_and_merge(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """Download AQI and weather, merge and save as raw CSV."""
        aq_df = self.fetch_air_quality()
        wx_df = self.fetch_weather()

        merged = pd.merge(aq_df, wx_df, on="timestamp", how="inner")
        path = output_path or RAW_DATA_PATH
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        merged.to_csv(path, index=False)
        return merged
