"""Trening modelu: historia, krzywe uczenia i zapis modelu."""
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from src.dataset import prepare_images
from src.model import build_model
from src.utils import get_logger

# Domyślne hiperparametry (można je nadpisać flagami w main.py albo plikiem --config)
DEFAULT_PARAMS = {
    "filters": 32,          # filtry w pierwszym bloku konwolucyjnym (drugi ma 2x więcej)
    "dropout": 0.5,         # część neuronów wyłączana w warstwie Dense
    "learning_rate": 1e-3,  # wielkość kroku Adama: mniejsza = wolniej, ale spokojniej
    "batch_size": 128,      # ile obrazów sieć przetwarza naraz przed korektą wag
    "epochs": 15,           # maksymalna liczba przejść przez cały zbiór treningowy
    "seed": 42,             # żeby wyniki dało się powtórzyć
}
PATIENCE = 3  # ile epok bez poprawy AUC na walidacji, zanim trening zostanie przerwany


class EpochLogger(keras.callbacks.Callback):
    """Zapisuje wynik każdej epoki do logu (konsola + log.txt)."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        get_logger().info(
            f"train: epoka {epoch + 1}: loss={logs['loss']:.4f} auc={logs['auc']:.4f} "
            f"val_loss={logs['val_loss']:.4f} val_auc={logs['val_auc']:.4f}"
        )


def train_model(splits: dict, params: dict, run_dir=None):
    """Trenuje model na zbiorze train, walidacja na val.

    Jeśli podano run_dir, zapisuje model, historię (CSV) i krzywe uczenia.
    Bez run_dir (np. w tuningu) tylko trenuje. Zwraca (model, historia).
    """
    log = get_logger()
    keras.utils.set_random_seed(params["seed"])

    x_train, y_train = prepare_images(splits["train"][0]), splits["train"][1].astype("float32")
    x_val, y_val = prepare_images(splits["val"][0]), splits["val"][1].astype("float32")
    log.info(f"train: dane treningowe {x_train.shape}, walidacyjne {x_val.shape}")
    log.info(f"train: parametry: {params}")

    model = build_model(params["filters"], params["dropout"], params["learning_rate"])
    if run_dir is not None:
        model.summary(print_fn=log.info)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=PATIENCE,
        restore_best_weights=True,  # na koniec wracamy do najlepszej epoki, nie ostatniej
    )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[early_stopping, EpochLogger()],
        verbose=0,  # postęp epok zapisuje EpochLogger
    ).history

    best_epoch = int(max(range(len(history["val_auc"])), key=lambda i: history["val_auc"][i]))
    log.info(f"train: koniec po {len(history['val_auc'])} epokach, najlepsza epoka {best_epoch + 1} "
             f"(val_auc={history['val_auc'][best_epoch]:.4f})")

    if run_dir is not None:
        model.save(run_dir / "model.keras")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index_label="epoch")
        _plot_learning_curves(history, run_dir / "figures" / "learning_curves.png")
        log.info("train: zapisano model.keras, history.csv i figures/learning_curves.png")
    return model, history


def _plot_learning_curves(history: dict, path) -> None:
    """Rysuje loss i AUC w kolejnych epokach (train kontra walidacja)."""
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(history["loss"], label="train")
    ax_loss.plot(history["val_loss"], label="val")
    ax_loss.set_title("Loss (błąd)")
    ax_auc.plot(history["auc"], label="train")
    ax_auc.plot(history["val_auc"], label="val")
    ax_auc.set_title("Macro AUC")
    for ax in (ax_loss, ax_auc):
        ax.set_xlabel("epoka")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
