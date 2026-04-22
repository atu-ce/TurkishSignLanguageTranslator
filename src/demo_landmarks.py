"""Canlı kamera + el landmark görselleştirme demosu.

Kullanım:
    python src/demo_landmarks.py
    python src/demo_landmarks.py --camera 1 --hands 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks.hand_detector import HAND_CONNECTIONS, HandLandmarkExtractor  # noqa: E402


def draw_landmarks(frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
    h, w = frame_bgr.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame_bgr, p, 4, (0, 0, 255), -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TID landmark canlı demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hands", type=int, default=1, help="Eş zamanlı el sayısı")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--model",
        type=str,
        default="models/hand_landmarker.task",
        help="MediaPipe .task dosyası",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[x] Kamera {args.camera} açılamadı", file=sys.stderr)
        sys.exit(1)

    print("[i] Çıkış için 'q' tuşuna basın.")

    fps_alpha = 0.9
    fps = 0.0
    prev = time.monotonic()
    start = prev

    with HandLandmarkExtractor(model_path=args.model, num_hands=args.hands) as extractor:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[x] Kare okunamadı", file=sys.stderr)
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            ts_ms = int((time.monotonic() - start) * 1000)
            hands = extractor.extract(frame_rgb, ts_ms)

            for i, hand in enumerate(hands):
                draw_landmarks(frame_bgr, hand.landmarks)
                cv2.putText(
                    frame_bgr,
                    f"{hand.handedness} ({hand.score:.2f})",
                    (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            now = time.monotonic()
            inst_fps = 1.0 / max(now - prev, 1e-6)
            fps = fps_alpha * fps + (1 - fps_alpha) * inst_fps if fps else inst_fps
            prev = now
            cv2.putText(
                frame_bgr,
                f"FPS: {fps:.1f}",
                (args.width - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("TID Landmark Demo (q=cikis)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
