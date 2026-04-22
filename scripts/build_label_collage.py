"""Dataset'teki her sınıf için örnek görüntüleri toplayıp tek kolaj yapar.

Amaç: TİD alfabesi görsellerini yan yana dizerek ASL alfabesi ile
manuel karşılaştırma yapılabilmesini sağlamak.

Kullanım:
    python scripts/build_label_collage.py data/Images
    python scripts/build_label_collage.py data/Images --per-class 3 --output tid.png
    python scripts/build_label_collage.py data/train --output asl.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# data/Images dosya adlarındaki ASCII simgeleri → muhtemel Türkçe harfler
# (Dosya sistemi Türkçe karakterlerle uyumsuz olduğu için değiştirilmiş)
SPECIAL_CHAR_MAP = {
    "!": "Ç (?)",
    "+": "Ğ (?)",
    ",": "Ö (?)",
    ";": "Ş (?)",
    "=": "Ü (?)",
    "_": "İ (?)",
}


def find_images_by_class(root: Path) -> dict[str, list[Path]]:
    """Hem flat (label-in-filename) hem class-folders yapısını destekler."""
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    has_class_folders = bool(subdirs) and any(
        any(f.suffix.lower() in IMG_EXT for f in d.iterdir() if f.is_file())
        for d in subdirs
    )

    groups: dict[str, list[Path]] = {}
    if has_class_folders:
        for d in subdirs:
            files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]
            if files:
                groups[d.name] = sorted(files)
    else:
        for f in root.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                label = f.stem.split(" ")[0].split("_")[0]
                groups.setdefault(label, []).append(f)
        for label in groups:
            groups[label].sort()
    return groups


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_collage(
    groups: dict[str, list[Path]],
    per_class: int,
    cols: int,
    output: Path,
) -> None:
    labels = sorted(groups.keys())
    cells = len(labels) * per_class
    rows = math.ceil(cells / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_2d(axes).flatten()

    idx = 0
    for label in labels:
        display_label = SPECIAL_CHAR_MAP.get(label, label)
        samples = groups[label][:per_class]
        for path in samples:
            ax = axes[idx]
            ax.imshow(load_rgb(path))
            ax.set_title(f"{display_label}", fontsize=11, fontweight="bold")
            ax.axis("off")
            idx += 1

    for i in range(idx, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{output.stem}  —  {len(labels)} sınıf × {per_class} örnek", fontsize=13)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[OK] {output}  ({idx} hücre, {len(labels)} sınıf)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sınıf bazlı örnek kolajı oluştur")
    p.add_argument("dataset", type=Path, help="Dataset kök klasörü")
    p.add_argument("--per-class", type=int, default=1, help="Sınıf başına örnek sayısı")
    p.add_argument("--cols", type=int, default=6, help="Kolaj sütun sayısı")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/collage.png"),
        help="Çıktı PNG dosyası",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_dir():
        raise SystemExit(f"[x] Dataset bulunamadı: {args.dataset}")

    groups = find_images_by_class(args.dataset)
    if not groups:
        raise SystemExit(f"[x] {args.dataset} içinde görüntü bulunamadı")

    make_collage(groups, args.per_class, args.cols, args.output)


if __name__ == "__main__":
    main()
