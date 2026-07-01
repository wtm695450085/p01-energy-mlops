import sqlite3
import pandas as pd
from pathlib import Path

p = Path("data/timeseries.db")
try:
    with sqlite3.connect(p) as con:
        df = pd.read_sql("SELECT ts, price_pln FROM prices LIMIT 5", con, parse_dates=["ts"])
        print(df)
except Exception as e:
    print(f"Exception: {e}")
