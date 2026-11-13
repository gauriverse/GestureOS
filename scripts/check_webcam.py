import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Webcam not found at index 0")

print("Webcam opened. Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("GestureOS - webcam check", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()