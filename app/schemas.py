"""Schematy Pydantic — 1:1 z 8 cechami modelu z model_metadata.json."""
from pydantic import BaseModel, Field, field_validator


class EnergyFeatures(BaseModel):
    """8 cech wejściowych modelu prognozowania cen energii RDN."""

    hour: int = Field(..., ge=0, le=23, description="Godzina doby (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Dzień tygodnia (0=pon, 6=niedz)")
    is_holiday_or_weekend: int = Field(..., ge=0, le=1, description="1 = weekend lub święto")
    temp_pl: float = Field(..., ge=-40.0, le=50.0, description="Temperatura krajowa [°C]")
    wind_pl: float = Field(..., ge=0.0, le=150.0, description="Prędkość wiatru krajowa [km/h]")
    radiation_pl: float = Field(..., ge=0.0, le=1500.0, description="Nasłonecznienie krajowe [W/m²]")
    price_lag_24h: float = Field(..., ge=-500.0, le=20000.0, description="Cena RDN sprzed 24h [PLN/MWh]")
    price_lag_168h: float = Field(..., ge=-500.0, le=20000.0, description="Cena RDN sprzed 168h [PLN/MWh]")

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
