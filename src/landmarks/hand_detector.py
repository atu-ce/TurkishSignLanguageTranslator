"""MediaPipe ile el tespiti ve 21 landmark çıkarımı (iskelet)."""

from __future__ import annotations


class HandLandmarkExtractor:
    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.5) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def extract(self, frame):
        """Tek karede 21 landmark noktası döndürür."""
        raise NotImplementedError
