import time
import csv
import os
from datetime import datetime

LOG_FILE = "logs/inference_monitoring.csv"
os.makedirs("logs", exist_ok=True)

# Create CSV header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "model_version",
            "num_images",
            "latency_ms",
            "status"
        ])


def log_inference(model_version, num_images, latency_ms, status="success"):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            model_version,
            num_images,
            latency_ms,
            status
        ])
