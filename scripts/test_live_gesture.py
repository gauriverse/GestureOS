import cv2

from gestureos.tracking.hand_tracker import HandTracker
from gestureos.features.normalize import normalize_hand
from gestureos.recognition.classifier import GestureClassifier
from gestureos.recognition.smoothing import VoteSmoother

tracker = HandTracker("models/hand_landmarker.task", max_hands=1)
clf = GestureClassifier("models/gesture_model_v2.joblib")
smoother = VoteSmoother()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Webcam not found")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    aspect = frame.shape[1] / frame.shape[0]

    hands = tracker.process(frame)
    if hands:
        feats = normalize_hand(hands[0].landmarks, aspect)
        label, conf = clf.predict(feats)
    else:
        label, conf = None, 0.0

    stable = smoother.update(label, conf)
    text = f"{stable or '...'}  (raw: {label or '-'} {conf:.2f})"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("GestureOS - live recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
tracker.close()
cv2.destroyAllWindows()