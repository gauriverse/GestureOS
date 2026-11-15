import cv2

from gestureos.tracking.hand_tracker import HandTracker
from gestureos.features.normalize import normalize_hand
from gestureos.recognition.classifier import GestureClassifier
from gestureos.recognition.smoothing import VoteSmoother
from gestureos.actions import media
from gestureos.actions.mouse import MouseController
from gestureos.actions.cooldown import Cooldown
from gestureos.modes import ModeManager
from gestureos.actions import presentation

INDEX_TIP = 8

tracker = HandTracker("models/hand_landmarker.task", max_hands=1)
clf = GestureClassifier("models/gesture_model_v2.joblib")
smoother = VoteSmoother()
modes = ModeManager(hold_seconds=1.0)
mouse = MouseController()
cooldown = Cooldown(0.5)
volume_repeat = Cooldown(0.4)
slide_cooldown = Cooldown(0.5)
slide_armed = True

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Webcam not found")

prev_stable = None
print("Hold open palm for 1 second to switch modes. Q in the window to stop.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    aspect = frame.shape[1] / frame.shape[0]

    hands = tracker.process(frame)
    if hands:
        lm = hands[0].landmarks
        label, conf = clf.predict(normalize_hand(lm, aspect))
    else:
        lm, label, conf = None, None, 0.0

    stable = smoother.update(label, conf)
    if stable != prev_stable:
        print("stable:", stable, "| raw:", label, f"{conf:.2f}", flush=True)
    if modes.update(stable):
        print("mode:", modes.mode)

    if modes.mode == "desktop":
        if stable == "index_up" and lm is not None:
            mouse.move_to(lm[INDEX_TIP, 0], lm[INDEX_TIP, 1])
        elif stable == "fist" and prev_stable != "fist" and cooldown.ready("click"):
            mouse.click()
        elif stable == "peace" and prev_stable != "peace" and cooldown.ready("right"):
            mouse.right_click()
        elif stable == "thumb_up" and volume_repeat.ready("scroll_up"):
            mouse.scroll_up()
        elif stable == "thumb_down" and volume_repeat.ready("scroll_down"):
            mouse.scroll_down()

    elif modes.mode == "media":
        if stable == "thumb_up" and volume_repeat.ready("vol_up"):
            media.volume_up()
        elif stable == "thumb_down" and volume_repeat.ready("vol_down"):
            media.volume_down()
        elif stable == "peace" and prev_stable != "peace" and cooldown.ready("playpause"):
            media.play_pause()

    elif modes.mode == "presentation":
        if stable not in ("thumb_up", "thumb_down"):
            slide_armed = True

        if stable == "index_up" and lm is not None:
            mouse.move_to(lm[INDEX_TIP, 0], lm[INDEX_TIP, 1])
        elif stable == "fist" and prev_stable != "fist" and cooldown.ready("click"):
            mouse.click()
        elif stable == "thumb_up" and slide_armed and slide_cooldown.ready("next"):
            presentation.next_slide()
            slide_armed = False
        elif stable == "thumb_down" and slide_armed and slide_cooldown.ready("prev"):
            presentation.previous_slide()
            slide_armed = False

    prev_stable = stable

    cv2.putText(frame, f"mode: {modes.mode}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"action: {stable or '...'}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"stable: {stable or '-'} armed: {slide_armed}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("GestureOS - modes test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
tracker.close()
cv2.destroyAllWindows()