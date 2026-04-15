"""Etiketli el landmark verisi toplama betiği (iskelet)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TID veri toplama")
    parser.add_argument("--label", type=str, required=True, help="Harf etiketi")
    parser.add_argument("--samples", type=int, default=200, help="Örnek sayısı")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[TID] '{args.label}' için {args.samples} örnek toplanacak. (iskelet)")
    # TODO: kamera + MediaPipe ile landmark kaydet


if __name__ == "__main__":
    main()
