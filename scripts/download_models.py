"""MediaPipe önceden eğitilmiş model indir.

Kullanım:
    python scripts/download_models.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def download(name: str, url: str) -> None:
    target = MODELS_DIR / name
    if target.exists():
        size_kb = target.stat().st_size / 1024
        print(f"[=] {name} zaten var ({size_kb:.1f} KB)")
        return

    print(f"[>] {name} indiriliyor...")
    urllib.request.urlretrieve(url, target)
    size_kb = target.stat().st_size / 1024
    print(f"[OK] {target} ({size_kb:.1f} KB)")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        try:
            download(name, url)
        except Exception as exc:
            print(f"[x] {name} indirilemedi: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
