import cv2

from gestureos.tracking.hand_tracker import HandTracker
from gestureos.features.normalize import normalize_hand
from gestureos.recognition.classifier import GestureClassifier
from gestureos.recognition.smoothing import VoteSmoother
from gestureos.actions import media
from gestureos.actions.cooldown import Cooldown

REPEAT_SECONDS = 0.4

tracker = HandTracker("models/hand_landmarker.task", max_hands=1)
clf = GestureClassifier("models/gesture_model_v2.joblib")
smoother = VoteSmoother()
cooldown = Cooldown(REPEAT_SECONDS)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Webcam not found")

print("Media test ON. thumb_up = volume up, thumb_down = volume down. Press Q in the window to stop.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    aspect = frame.shape[1] / frame.shape[0]

    hands = tracker.process(frame)
    if hands:
        label, conf = clf.predict(normalize_hand(hands[0].landmarks, aspect))
    else:
        label, conf = None, 0.0

    stable = smoother.update(label, conf)

    if stable == "thumb_up" and cooldown.ready("vol_up"):
        media.volume_up()
    elif stable == "thumb_down" and cooldown.ready("vol_down"):
        media.volume_down()

    cv2.putText(frame, f"action: {stable or '...'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("GestureOS - media test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
tracker.close()
cv2.destroyAllWindows()