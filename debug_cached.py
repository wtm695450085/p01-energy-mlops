from app.features import load_cached_prices
df = load_cached_prices()
print(df.dtypes)
print(df["ts"].head())
