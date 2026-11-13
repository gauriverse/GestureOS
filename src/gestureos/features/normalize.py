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