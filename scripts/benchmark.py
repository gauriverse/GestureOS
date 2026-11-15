import glob
import time

import numpy as np
import pandas as pd

from gestureos.recognition.classifier import GestureClassifier

N_SAMPLES = 500
WARMUP = 20

clf = GestureClassifier("models/gesture_model_v2.joblib")

df = pd.concat([pd.read_csv(f) for f in glob.glob("data/raw/*.csv")], ignore_index=True)
feature_cols = [c for c in df.columns if c.startswith("f")]
X = df[feature_cols].to_numpy(dtype=np.float32)

rng = np.random.default_rng(0)
samples = X[rng.choice(len(X), size=min(N_SAMPLES, len(X)), replace=False)]


for s in samples[:WARMUP]:
    clf.predict(s)

times_ms = []
for s in samples:
    t0 = time.perf_counter()
    clf.predict(s)
    times_ms.append((time.perf_counter() - t0) * 1000)

times = np.array(times_ms)
print(f"samples timed: {len(times)}")
print(f"median:  {np.median(times):.3f} ms")
print(f"p95:     {np.percentile(times, 95):.3f} ms")
print(f"mean:    {times.mean():.3f} ms")
print(f"max:     {times.max():.3f} ms")