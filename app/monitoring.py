"""Moduł monitoringu: Metryki Prometheus dla aplikacji."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Definicje metryk
mlops_predictions_total = Counter(
    "mlops_predictions_total",
    "Total number of predictions requested"
)

mlops_prediction_errors_total = Counter(
    "mlops_prediction_errors_total",
    "Total number of prediction errors"
)

mlops_prediction_latency_seconds = Histogram(
    "mlops_prediction_latency_seconds",
    "Latency of prediction requests in seconds"
)

mlops_last_prediction_value = Gauge(
    "mlops_last_prediction_value",
    "Value of the last successful prediction"
)

mlops_data_drift_detected = Gauge(
    "mlops_data_drift_detected",
    "1 if data drift was detected in the last report, 0 otherwise"
)

def get_metrics_response() -> Response:
    """Zwraca metryki w formacie Prometheus."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
