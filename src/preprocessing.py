"""Tryb preprocessing: kontrola jakości danych, statystyki i wykresy."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import CLASS_NAMES, SPLITS, labels_to_text, prepare_images
from src.utils import get_logger, save_json


def run_preprocessing(splits: dict, run_dir) -> None:
    """Sprawdza dane (kształty, typy, braki, rozkład klas) i zapisuje wykresy oraz podsumowanie."""
    log = get_logger()
    log.info("preprocessing: start")
    summary = {}

    for split in SPLITS:
        images, labels = splits[split]
        assert len(images) == len(labels), f"{split}: różna liczba obrazów i etykiet"

        labels_per_image = labels.sum(axis=1)
        without_label = int((labels_per_image == 0).sum())
        blank_images = int((images.reshape(len(images), -1).max(axis=1) == 0).sum())
        label_values = [int(v) for v in np.unique(labels)]

        summary[split] = {
            "images": int(len(images)),
            "image_shape": list(images.shape[1:]),
            "image_dtype": str(images.dtype),
            "pixel_min": int(images.min()),
            "pixel_max": int(images.max()),
            "label_values": label_values,
            "blank_images": blank_images,
            "images_without_label": without_label,
            "avg_labels_per_image": round(float(labels_per_image.mean()), 3),
            "max_labels_per_image": int(labels_per_image.max()),
        }
        log.info(f"preprocessing: {split}: {len(images)} obrazów {images.shape[1:]}, "
                 f"piksele {images.min()}-{images.max()}, całkiem czarne obrazy: {blank_images}")
        log.info(f"preprocessing: {split}: wartości etykiet {label_values} (poprawnie tylko 0 i 1), "
                 f"bez żadnej choroby: {without_label} ({without_label / len(images):.1%}), "
                 f"średnio chorób na obraz: {labels_per_image.mean():.2f}, max: {labels_per_image.max()}")

    # Liczebność każdej choroby w każdym podziale
    counts = pd.DataFrame({split: splits[split][1].sum(axis=0) for split in SPLITS}, index=CLASS_NAMES)
    counts["train_percent"] = (counts["train"] / len(splits["train"][1]) * 100).round(2)
    counts.to_csv(run_dir / "metrics" / "class_counts.csv", index_label="class")
    rarest, most_common = counts["train"].idxmin(), counts["train"].idxmax()
    log.info(f"preprocessing: najrzadsza choroba (train): {rarest} ({counts.loc[rarest, 'train']} zdjęć), "
             f"najczęstsza: {most_common} ({counts.loc[most_common, 'train']} zdjęć)")
    log.info(f"preprocessing: niezbalansowanie klas: najczęstsza jest "
             f"{counts.loc[most_common, 'train'] / counts.loc[rarest, 'train']:.0f}x częstsza niż najrzadsza")

    _plot_class_distribution(splits, counts, run_dir / "figures" / "class_distribution.png")
    _plot_pixel_histogram(splits["train"][0], run_dir / "figures" / "pixel_histogram.png")
    _plot_sample_images(splits["train"], run_dir / "figures" / "sample_images.png")

    # Kontrola skalowania: sprawdzamy na próbce, że po przeliczeniu zakres to 0-1
    scaled = prepare_images(splits["train"][0][:1000])
    summary["after_scaling_sample"] = {
        "dtype": str(scaled.dtype), "shape": list(scaled.shape),
        "min": float(scaled.min()), "max": float(scaled.max()),
    }
    log.info(f"preprocessing: po skalowaniu (próbka 1000 obrazów): {scaled.dtype}, kształt {scaled.shape}, "
             f"zakres {scaled.min():.3f}-{scaled.max():.3f}")

    save_json(summary, run_dir / "metrics" / "preprocessing_summary.json")
    log.info("preprocessing: koniec (liczby zdjęć nie zmieniły się, bo żadnych nie usuwamy)")


def _plot_class_distribution(splits: dict, counts: pd.DataFrame, path) -> None:
    """Rysuje, jaki procent zdjęć ma daną chorobę (osobno dla train/val/test)."""
    percent = pd.DataFrame({s: counts[s] / len(splits[s][1]) * 100 for s in SPLITS}).sort_values("train")
    ax = percent.plot.barh(figsize=(9, 7))
    ax.set_xlabel("% zdjęć z daną chorobą")
    ax.set_title("Rozkład klas (niezbalansowanie)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_pixel_histogram(images: np.ndarray, path) -> None:
    """Rysuje histogram jasności pikseli w zbiorze treningowym."""
    histogram = np.bincount(images.ravel(), minlength=256)
    plt.figure(figsize=(8, 4))
    plt.bar(range(256), histogram, width=1.0)
    plt.xlabel("wartość piksela (0 = czarny, 255 = biały)")
    plt.ylabel("liczba pikseli")
    plt.title("Histogram pikseli (train)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_sample_images(split_data: tuple, path) -> None:
    """Rysuje 16 losowych zdjęć treningowych z ich etykietami."""
    images, labels = split_data
    rng = np.random.default_rng(42)  # stały seed, żeby zawsze losowało te same
    indexes = rng.choice(len(images), size=16, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, i in zip(axes.ravel(), indexes):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(textwrap.fill(labels_to_text(labels[i]), 22), fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
