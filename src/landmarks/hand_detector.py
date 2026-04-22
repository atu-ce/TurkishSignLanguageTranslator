"""MediaPipe Tasks API ile el landmark çıkarımı."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class HandResult:
    landmarks: np.ndarray
    world_landmarks: np.ndarray
    handedness: str
    score: float


class HandLandmarkExtractor:
    """MediaPipe Tasks API sarıcısı. Video akışından el landmark'larını çıkarır."""

    def __init__(
        self,
        model_path: str | Path = "models/hand_landmarker.task",
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {model_path}\n"
                "İndirmek için: python scripts/download_models.py"
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def extract(self, rgb_frame: np.ndarray, timestamp_ms: int) -> list[HandResult]:
        """Tek karede tespit edilen elleri döndürür.

        Args:
            rgb_frame: (H, W, 3) uint8 RGB dizisi.
            timestamp_ms: Karenin monoton olarak artan zaman damgası (ms).
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        hands: list[HandResult] = []
        for idx, landmarks in enumerate(result.hand_landmarks):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
            )
            world = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[idx]],
                dtype=np.float32,
            )
            cat = result.handedness[idx][0]
            hands.append(
                HandResult(
                    landmarks=coords,
                    world_landmarks=world,
                    handedness=cat.category_name,
                    score=cat.score,
                )
            )
        return hands

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
