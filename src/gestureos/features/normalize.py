import numpy as np

WRIST = 0
MIDDLE_MCP = 9
FEATURE_DIM = 63

def normalize_hand(landmarks: np.ndarray, aspect: float = 1.0) -> np.ndarray:
    pts = landmarks.astype(np.float32).copy()
    pts[:,0] *= aspect

    pts -= pts[WRIST]

    scale = float(np.linalg.norm(pts[MIDDLE_MCP, :2]))

    if scale < 1e-6:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    pts /= scale
    return pts.flatten()

# Thumb tip (4) to index MCP (5), middle MCP (9), pinky MCP (17);
# each fingertip (8, 12, 16, 20) to the wrist (0).
DISTANCE_PAIRS = [(4, 5), (4, 9), (4, 17), (8, 0), (12, 0), (16, 0), (20, 0)]
N_DISTANCE_FEATURES = len(DISTANCE_PAIRS)


def distance_features(X: np.ndarray) -> np.ndarray:
    """
    X: (n, 63) normalized feature rows from normalize_hand().
    Returns (n, 7) in-plane distances between landmark pairs.
    Works on the stored CSV features, so no re-recording is needed.
    """
    pts = X.reshape(-1, 21, 3)[:, :, :2]  # keep x and y only
    cols = [np.linalg.norm(pts[:, a] - pts[:, b], axis=1) for a, b in DISTANCE_PAIRS]
    return np.stack(cols, axis=1).astype(np.float32)