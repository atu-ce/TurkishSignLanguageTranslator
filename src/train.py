"""Model eğitim betiği (iskelet)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TID model eğitimi")
    parser.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[TID] {args.model.upper()} modeli {args.epochs} epoch ile eğitilecek. (iskelet)")
    # TODO: veri yükleme, model oluşturma, eğitim, kaydetme


if __name__ == "__main__":
    main()
