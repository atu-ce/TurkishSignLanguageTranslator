"""Temel görüntü önişleme yardımcıları (iskelet)."""

from __future__ import annotations


def resize(frame, width: int, height: int):
    raise NotImplementedError


def rgb_to_bgr(frame):
    raise NotImplementedError


def denoise(frame):
    raise NotImplementedError
