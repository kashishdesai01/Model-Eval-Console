from prometheus_client import Counter, Histogram

RUNS = Counter("mec_runs_total", "Completed evaluation attempts", ["status"])
RECLAIMS = Counter("mec_lease_reclaims_total", "Expired run leases reclaimed")
RUN_DURATION = Histogram("mec_run_duration_seconds", "Evaluation attempt duration")
INFERENCE_LATENCY = Histogram(
    "mec_inference_latency_seconds",
    "Warm inference batch latency",
    ["quantization"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
PREDICT_REQUESTS = Counter("mec_predict_requests_total", "Online prediction requests", ["status"])
