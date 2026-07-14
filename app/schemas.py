"""Pydantic schemas - one-to-one with the 8 model features from model_metadata.json."""
from pydantic import BaseModel, Field, field_validator


class EnergyFeatures(BaseModel):
    """8 input features for the RDN energy price forecasting model."""

    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    is_holiday_or_weekend: int = Field(..., ge=0, le=1, description="1 = weekend or holiday")
    temp_pl: float = Field(..., ge=-40.0, le=50.0, description="National temperature [°C]")
    wind_pl: float = Field(..., ge=0.0, le=150.0, description="National wind speed [km/h]")
    radiation_pl: float = Field(..., ge=0.0, le=1500.0, description="National solar radiation [W/m²]")
    price_lag_24h: float = Field(..., ge=-500.0, le=20000.0, description="RDN price from 24h ago [PLN/MWh]")
    price_lag_168h: float = Field(..., ge=-500.0, le=20000.0, description="RDN price from 168h ago [PLN/MWh]")

    model_config = {"json_schema_extra": {
        "example": {
            "hour": 14,
            "day_of_week": 2,
            "is_holiday_or_weekend": 0,
            "temp_pl": 15.3,
            "wind_pl": 12.5,
            "radiation_pl": 320.0,
            "price_lag_24h": 285.50,
            "price_lag_168h": 271.30,
        }
    }}


class PredictionResponse(BaseModel):
    prediction: float
    model_file: str
    status: str
    request_id: str | None = None


class NextDayForecastItem(BaseModel):
    hour: int
    datetime: str
    predicted_price_pln: float
    temp_pl: float
    wind_pl: float
    radiation_pl: float
    is_holiday_or_weekend: int


class NextDayForecastResponse(BaseModel):
    date: str
    forecasts: list[NextDayForecastItem]
    model_file: str
    status: str
    generated_at: str
