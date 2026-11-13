import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from gestureos.tracking.hand_tracker import HandTracker
from gestureos.features.normalize import normalize_hand

SAVE_EVERY_N_FRAMES = 3
OUT_DIR = Path("data/raw")


def main():
    parser = argparse.ArgumentParser(description="Record normalized hand landmarks")
    parser.add_argument("--gesture", required=True, help="gesture label, e.g. fist")
    parser.add_argument("--session", required=True, type=int, help="session number, e.g. 1")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.gesture}_session{args.session}.csv"
    is_new = not out_path.exists()

    tracker = HandTracker("models/hand_landmarker.task", max_hands=1)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Webcam not found")

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect = width / height

    recording = False
    frame_count = 0
    saved = 0

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            header = ["label", "session"] + [f"f{i}" for i in range(63)]
            writer.writerow(header)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)

            if hands:
                lm = hands[0].landmarks
                h, w = frame.shape[:2]
                for x, y, _ in lm:
                    cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)

                if recording:
                    frame_count += 1
                    if frame_count % SAVE_EVERY_N_FRAMES == 0:
                        features = normalize_hand(lm, aspect)
                        writer.writerow([args.gesture, args.session, *features.tolist()])
                        saved += 1

            status = "REC" if recording else "PAUSED"
            color = (0, 0, 255) if recording else (255, 255, 255)
            cv2.putText(frame, f"{args.gesture} | session {args.session} | {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"saved: {saved}  hand: {'yes' if hands else 'no'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "R: record/pause   Q: quit",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("GestureOS - collect samples", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                recording = not recording
                frame_count = 0
            elif key == ord("q"):
                break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print(f"Saved {saved} samples to {out_path}")


if __name__ == "__main__":
    main()