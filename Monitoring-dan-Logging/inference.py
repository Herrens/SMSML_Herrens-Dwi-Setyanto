from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)
import mlflow.pyfunc
import pandas as pd
import threading
import time
import random

MODEL_PATH = "../mlruns/612928296766799338/3d92cf5917f74cd6b7f0ab0adc029e25/artifacts/model"
NUM_FEATURES = 20
AUTO_INTERVAL = 3

model = mlflow.pyfunc.load_model(MODEL_PATH)
app = FastAPI()

LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
UPTIME = Gauge("uptime_seconds", "Service uptime")

start_time = time.time()

def auto_predict_loop():
    while True:
        start = time.time()
        try:
            dummy = pd.DataFrame(
                [[random.random() for _ in range(NUM_FEATURES)]]
            )
            model.predict(dummy)
        finally:
            LATENCY.observe(time.time() - start)
        time.sleep(AUTO_INTERVAL)

threading.Thread(target=auto_predict_loop, daemon=True).start()

@app.get("/metrics")
def metrics():
    UPTIME.set(time.time() - start_time)
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
