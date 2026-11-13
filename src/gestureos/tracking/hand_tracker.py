from dataclasses import dataclass
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

@dataclass
class HandResult:
    landmarks: np.ndarray
    handedness: str
    score: float

class HandTracker:
    def __init__(self, model_path: str, max_hands: int =2, min_detection: float = 0.6, min_tracking: float = 0.5):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_tracking,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start = time.monotonic()

    def process(self, frame_bgr: np.ndarray) -> list[HandResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        timestamp_ms = int((time.monotonic() - self._start) * 1000)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        hands = []
        for lm_list, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
            hands.append(HandResult(pts, handed[0].category_name, handed[0].score))
        return hands

    def close(self):
        self._landmarker.close()