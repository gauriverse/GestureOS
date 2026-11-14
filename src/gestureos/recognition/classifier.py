import joblib
import numpy as np

from gestureos.features.normalize import distance_features


class GestureClassifier:

    def __init__(self, model_path: str = "models/gesture_model_v2.joblib"):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.labels = bundle["labels"]  # same order as the model's class probabilities

    def predict(self, features63: np.ndarray) -> tuple[str, float]:
        row = features63.reshape(1, -1).astype(np.float32)
        # Must match training: 63 normalized values plus 7 distance features
        x = np.hstack([row, distance_features(row)])
        probs = self.model.predict_proba(x)[0]
        i = int(np.argmax(probs))
        return self.labels[i], float(probs[i])