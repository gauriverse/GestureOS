import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gestureos.features.normalize import distance_features


DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
USE_DISTANCES = True
MODEL_NAME = "gesture_model_v2" if USE_DISTANCES else "gesture_model_v1"
TRAIN_SESSIONS = [1, 2, 4]
TEST_SESSION = 3


def load_data():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA_DIR / "*.csv"))],
                   ignore_index=True)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    if USE_DISTANCES:
        X = np.hstack([X, distance_features(X)])
    y = df["label"].to_numpy()
    sessions = df["session"].to_numpy()
    return X, y, sessions


def build_candidates():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42),
        "svm_rbf": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                  probability=True, random_state=42)),
        "mlp": make_pipeline(
            StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, random_state=42)),
    }


def main():
    X, y, sessions = load_data()
    train_mask = np.isin(sessions, TRAIN_SESSIONS)
    test_mask = sessions == TEST_SESSION
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"train: {len(X_train)} samples | test: {len(X_test)} samples")

    best_name, best_model, best_acc = None, None, -1.0
    for name, model in build_candidates().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = (pred == y_test).mean()
        print(f"\n=== {name}  accuracy: {acc:.4f} ===")
        print(classification_report(y_test, pred, digits=3))
        labels = sorted(set(y_test))
        print("confusion matrix (rows = true, cols = predicted):")
        print(pd.DataFrame(confusion_matrix(y_test, pred, labels=labels),
                           index=labels, columns=labels))
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / f"{MODEL_NAME}.joblib"
    joblib.dump({"model": best_model, "classifier": best_name,
                 "labels": sorted(set(y))}, out_path)
    print(f"\nBest: {best_name} ({best_acc:.4f}). Saved to {out_path}")


if __name__ == "__main__":
    main()