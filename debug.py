from app.features import build_tomorrow_features
try:
    build_tomorrow_features()
except Exception as e:
    print(f"Exception in build_tomorrow_features: {e}")
    import traceback
    traceback.print_exc()
