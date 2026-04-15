"""Gerçek zamanlı Türk İşaret Dili çevirmeni giriş noktası."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TID Gerçek Zamanlı Çevirmen")
    parser.add_argument("--camera", type=int, default=0, help="Kamera indeksi")
    parser.add_argument("--model", type=str, default=None, help="Model dosya yolu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[TID] Kamera {args.camera} başlatılıyor... (iskelet)")
    # TODO: kamera akışı, önişleme, landmark, özellik, sınıflandırma döngüsü


if __name__ == "__main__":
    main()
