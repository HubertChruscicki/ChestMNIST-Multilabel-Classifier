"""Pobieranie i wczytywanie ChestMNIST z biblioteki medmnist."""
from pathlib import Path

import numpy as np
from medmnist import INFO, ChestMNIST

from src.utils import DATA_DIR, get_logger

SPLITS = ("train", "val", "test")
NUM_CLASSES = 14
# Nazwy chorób bierzemy z biblioteki, żeby nie przepisywać ich ręcznie
CLASS_NAMES = [INFO["chestmnist"]["label"][str(i)] for i in range(NUM_CLASSES)]
NPZ_NAME = "chestmnist.npz"


def download_dataset() -> Path:
    """Pobiera plik ChestMNIST do folderu data/ (tylko jeśli jeszcze go tam nie ma)."""
    log = get_logger()
    npz_path = DATA_DIR / NPZ_NAME
    if npz_path.exists():
        log.info(f"dataset: plik {NPZ_NAME} już jest w data/, pomijam pobieranie")
        return npz_path

    log.info("dataset: pobieram ChestMNIST przez bibliotekę medmnist...")
    DATA_DIR.mkdir(exist_ok=True)  # medmnist wymaga, żeby folder już istniał
    # Samo utworzenie obiektu z download=True pobiera plik .npz ze wszystkimi podziałami
    ChestMNIST(split="train", root=str(DATA_DIR), download=True)
    log.info(f"dataset: pobrano do {npz_path}")
    return npz_path


def load_splits() -> dict:
    """Wczytuje oficjalne podziały train/val/test.

    Zwraca słownik {"train": (obrazy, etykiety), "val": ..., "test": ...}.
    Obrazy mają kształt (N, 28, 28) i typ uint8, etykiety (N, 14) z wartościami 0/1.
    """
    log = get_logger()
    data = np.load(download_dataset())
    splits = {}
    for split in SPLITS:
        images = data[f"{split}_images"]
        labels = data[f"{split}_labels"]
        splits[split] = (images, labels)
        log.info(f"dataset: {split}: obrazy {images.shape} ({images.dtype}), etykiety {labels.shape} ({labels.dtype})")
    return splits


def prepare_images(images: np.ndarray) -> np.ndarray:
    """Zamienia obrazy na wejście sieci.

    Piksele 0-255 (uint8) dzielimy przez 255, żeby były w zakresie 0-1,
    a na końcu dodajemy kanał: (N, 28, 28) -> (N, 28, 28, 1), bo tego oczekuje Conv2D.
    """
    scaled = images.astype("float32") / 255.0
    return scaled[..., np.newaxis]


def labels_to_text(label_row: np.ndarray) -> str:
    """Zamienia wektor 0/1 na nazwy chorób, np. 'effusion, mass' (albo 'brak etykiet')."""
    names = [CLASS_NAMES[i] for i in np.where(label_row == 1)[0]]
    return ", ".join(names) if names else "brak etykiet"
