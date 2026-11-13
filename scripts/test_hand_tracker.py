import cv2
from gestureos.tracking.hand_tracker import HandTracker

tracker = HandTracker("models/hand_landmarker.task")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Webcam not found")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  
    h, w = frame.shape[:2]

    for hand in tracker.process(frame):
        for x, y, _ in hand.landmarks:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"{hand.handedness} {hand.score:.2f}", (10, 30 + 30 * 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("GestureOS - hand tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
tracker.close()
cv2.destroyAllWindows()