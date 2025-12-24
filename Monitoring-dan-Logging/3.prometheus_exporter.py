from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server
)
import time
import random

# =========================
# DEFINE METRICS
# =========================

REQUEST_COUNT = Counter(
    "request_count",
    "Total number of prediction requests"
)

ERROR_COUNT = Counter(
    "error_count",
    "Total number of prediction errors"
)

PREDICTION_COUNT = Counter(
    "prediction_count",
    "Total number of predictions made"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

UPTIME_SECONDS = Gauge(
    "uptime_seconds",
    "Service uptime in seconds"
)

CPU_USAGE = Gauge(
    "cpu_usage",
    "CPU usage percentage"
)

MEMORY_USAGE = Gauge(
    "memory_usage",
    "Memory usage percentage"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Model accuracy"
)

MODEL_PRECISION = Gauge(
    "model_precision",
    "Model precision"
)

MODEL_RECALL = Gauge(
    "model_recall",
    "Model recall"
)

# =========================
# START EXPORTER
# =========================

if __name__ == "__main__":
    start_time = time.time()

    # Start Prometheus exporter on port 8001
    start_http_server(8001)
    print("Prometheus exporter running on http://localhost:8001/metrics")

    while True:
        # Simulate metrics update
        REQUEST_COUNT.inc(random.randint(0, 2))
        PREDICTION_COUNT.inc(random.randint(0, 2))

        if random.random() < 0.1:
            ERROR_COUNT.inc()

        PREDICTION_LATENCY.observe(random.uniform(0.05, 0.8))

        UPTIME_SECONDS.set(time.time() - start_time)
        CPU_USAGE.set(random.uniform(10, 60))
        MEMORY_USAGE.set(random.uniform(20, 70))

        MODEL_ACCURACY.set(0.82)
        MODEL_PRECISION.set(0.78)
        MODEL_RECALL.set(0.74)

        time.sleep(5)