from datetime import date
from app.features import fetch_pse_prices
df = fetch_pse_prices(date(2026, 6, 1), date(2026, 6, 5))
print(df.dtypes)
