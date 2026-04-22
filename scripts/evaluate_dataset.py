"""Bir görüntü veri setinde MediaPipe landmark çıkarım başarısını ölçer.

Dataset yapısı iki şekilde olabilir:
  (a) Flat:           <root>/*.{png,jpg,jpeg}   (etiket dosya adından çıkarılır)
  (b) Class-folders:  <root>/<etiket>/*.{png,jpg,jpeg}

Kullanım:
    python scripts/evaluate_dataset.py data/Images
    python scripts/evaluate_dataset.py data/train --sample 500 --num-hands 2
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parents[1]


def find_images(root: Path) -> tuple[list[tuple[Path, str]], str]:
    """Görüntüleri ve etiketlerini bulur. Döner: (list[(path, label)], layout)."""
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    has_class_folders = bool(subdirs) and any(
        any(f.suffix.lower() in IMG_EXT for f in d.iterdir() if f.is_file())
        for d in subdirs
    )

    items: list[tuple[Path, str]] = []
    if has_class_folders:
        for d in subdirs:
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    items.append((f, d.name))
        return items, "class-folders"

    for f in root.iterdir():
        if f.is_file() and f.suffix.lower() in IMG_EXT:
            # Etiket = dosya adı boşluğa kadarki ilk kısım (ör. "A (1).png" -> "A")
            label = f.stem.split(" ")[0].split("_")[0]
            items.append((f, label))
    return items, "flat"


def build_detector(model_path: Path, num_hands: int, confidence: float) -> vision.HandLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def evaluate(
    items: list[tuple[Path, str]],
    detector: vision.HandLandmarker,
) -> dict:
    per_class_total: Counter = Counter()
    per_class_success: Counter = Counter()
    hand_counts: Counter = Counter()

    for path, label in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
        n_hands = len(result.hand_landmarks)

        per_class_total[label] += 1
        hand_counts[n_hands] += 1
        if n_hands > 0:
            per_class_success[label] += 1

    return {
        "total": sum(per_class_total.values()),
        "success": sum(per_class_success.values()),
        "per_class_total": per_class_total,
        "per_class_success": per_class_success,
        "hand_counts": hand_counts,
    }


def print_report(root: Path, layout: str, stats: dict) -> None:
    total, success = stats["total"], stats["success"]
    rate = 100.0 * success / total if total else 0.0

    print(f"\n{'='*60}")
    print(f"Dataset: {root}")
    print(f"Yapi:    {layout}")
    print(f"Toplam:  {total} gorunt")
    print(f"Basari:  {success} / {total} ({rate:.1f}%)")
    print(f"Tespit dagilimi: {dict(sorted(stats['hand_counts'].items()))}")
    print(f"\n{'Sinif':<15} {'Toplam':>8} {'Basari':>8} {'Oran':>8}")
    print("-" * 45)
    for label in sorted(stats["per_class_total"]):
        t = stats["per_class_total"][label]
        s = stats["per_class_success"][label]
        r = 100.0 * s / t if t else 0.0
        print(f"{label:<15} {t:>8} {s:>8} {r:>7.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset uzerinde MediaPipe basari olcumu")
    parser.add_argument("dataset", type=Path, help="Dataset kok klasoru")
    parser.add_argument("--sample", type=int, default=0, help="Rastgele orneklem (0 = tumu)")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "hand_landmarker.task",
        help="MediaPipe .task yolu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.is_dir():
        print(f"[x] Dataset klasoru yok: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    if not args.model.exists():
        print(f"[x] Model bulunamadi: {args.model}", file=sys.stderr)
        print("    python scripts/download_models.py calistirin.", file=sys.stderr)
        sys.exit(1)

    items, layout = find_images(args.dataset)
    if not items:
        print(f"[x] {args.dataset} icinde gorunt bulunamadi", file=sys.stderr)
        sys.exit(1)

    if args.sample > 0 and args.sample < len(items):
        random.seed(args.seed)
        items = random.sample(items, args.sample)

    detector = build_detector(args.model, args.num_hands, args.confidence)
    try:
        stats = evaluate(items, detector)
    finally:
        detector.close()

    print_report(args.dataset, layout, stats)


if __name__ == "__main__":
    main()
