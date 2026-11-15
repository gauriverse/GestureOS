import time
from dataclasses import dataclass

import cv2

from gestureos.tracking.hand_tracker import HandTracker
from gestureos.features.normalize import normalize_hand
from gestureos.recognition.classifier import GestureClassifier
from gestureos.recognition.smoothing import VoteSmoother
from gestureos.actions import media, presentation
from gestureos.actions.mouse import MouseController
from gestureos.actions.cooldown import Cooldown
from gestureos.modes import ModeManager

INDEX_TIP = 8

@dataclass
class Status:
    mode: str
    raw: str | None
    stable: str | None
    confidence: float
    fps: float
    latency_ms: float

class GestureController:
    def __init__(self, model_path="models/gesture_model_v2.joblib",
                 hand_model="models/hand_landmarker.task", camera_index=0):
        self.tracker = HandTracker(hand_model, max_hands=1)
        self.clf = GestureClassifier(model_path)
        self.smoother = VoteSmoother()
        self.modes = ModeManager(hold_seconds=1.0)
        self.mouse = MouseController()
        self.click_cd = Cooldown(0.5)
        self.repeat_cd = Cooldown(0.4)
        self.slide_cd = Cooldown(0.5)
        self.slide_armed = True
        self.prev_stable = None
        self.status = Status(self.modes.mode, None, None, 0.0, 0.0, 0.0)

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam not found")

    def process_frame(self, frame):
        """frame: already mirrored BGR image. Returns (raw, conf, stable, lm, latency_ms)."""
        t0 = time.perf_counter()
        aspect = frame.shape[1] / frame.shape[0]

        hands = self.tracker.process(frame)
        if hands:
            lm = hands[0].landmarks
            raw, conf = self.clf.predict(normalize_hand(lm, aspect))
        else:
            lm, raw, conf = None, None, 0.0

        stable = self.smoother.update(raw, conf)
        self.modes.update(stable)
        self._dispatch(stable, lm)
        self.prev_stable = stable

        latency_ms = (time.perf_counter() - t0) * 1000
        return raw, conf, stable, lm, latency_ms

    def _dispatch(self, stable, lm):
        mode = self.modes.mode
        prev = self.prev_stable

        if stable not in ("thumb_up", "thumb_down"):
            self.slide_armed = True

        if mode == "desktop":
            if stable == "index_up" and lm is not None:
                self.mouse.move_to(lm[INDEX_TIP, 0], lm[INDEX_TIP, 1])
            elif stable == "fist" and prev != "fist" and self.click_cd.ready("click"):
                self.mouse.click()
            elif stable == "peace" and prev != "peace" and self.click_cd.ready("right"):
                self.mouse.right_click()
            elif stable == "thumb_up" and self.repeat_cd.ready("scroll_up"):
                self.mouse.scroll_up()
            elif stable == "thumb_down" and self.repeat_cd.ready("scroll_down"):
                self.mouse.scroll_down()

        elif mode == "media":
            if stable == "thumb_up" and self.repeat_cd.ready("vol_up"):
                media.volume_up()
            elif stable == "thumb_down" and self.repeat_cd.ready("vol_down"):
                media.volume_down()
            elif stable == "peace" and prev != "peace" and self.click_cd.ready("playpause"):
                media.play_pause()

        elif mode == "presentation":
            if stable == "index_up" and lm is not None:
                self.mouse.move_to(lm[INDEX_TIP, 0], lm[INDEX_TIP, 1])
            elif stable == "fist" and prev != "fist" and self.click_cd.ready("click"):
                self.mouse.click()
            elif stable == "thumb_up" and self.slide_armed and self.slide_cd.ready("next"):
                presentation.next_slide()
                self.slide_armed = False
            elif stable == "thumb_down" and self.slide_armed and self.slide_cd.ready("prev"):
                presentation.previous_slide()
                self.slide_armed = False

    def run(self, show_window=True):
        fps = 0.0
        last = time.perf_counter()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            raw, conf, stable, lm, latency_ms = self.process_frame(frame)

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last, 1e-6)
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps
            last = now

            self.status = Status(self.modes.mode, raw, stable, conf, fps, latency_ms)

            if show_window:
                self._draw(frame)
                cv2.imshow("GestureOS", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.close()

    def _draw(self, frame):
        s = self.status
        lines = [
            f"mode: {s.mode}",
            f"action: {s.stable or '...'}  (raw: {s.raw or '-'} {s.confidence:.2f})",
            f"fps: {s.fps:.1f}   latency: {s.latency_ms:.1f} ms",
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 35 + 35 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def close(self):
        self.tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()