"""Ewaluacja: progi dla każdej choroby, metryki na teście, tabela klas i analiza błędów."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)

from src.dataset import CLASS_NAMES, NUM_CLASSES, labels_to_text, prepare_images
from src.utils import get_logger, save_json

DEFAULT_THRESHOLD = 0.5  # standardowy próg: powyżej tego uznajemy, że choroba występuje


def find_thresholds(y_val: np.ndarray, probs_val: np.ndarray) -> np.ndarray:
    """Dla każdej choroby szuka progu, przy którym F1 na zbiorze walidacyjnym jest najwyższe.

    Rzadkie choroby dostają zwykle niższy próg niż 0.5, bo model rzadko jest ich pewny.
    """
    thresholds = []
    for i in range(NUM_CLASSES):
        if y_val[:, i].sum() == 0:  # brak przypadków w walidacji: zostajemy przy 0.5
            thresholds.append(DEFAULT_THRESHOLD)
            continue
        precision, recall, cuts = precision_recall_curve(y_val[:, i], probs_val[:, i])
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        # precision i recall mają o jeden element więcej niż progi, stąd f1[:-1]
        thresholds.append(float(cuts[np.argmax(f1[:-1])]))
    return np.array(thresholds)


def compute_class_auc(y: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Liczy AUC osobno dla każdej choroby (NaN, jeśli w danych jest tylko jedna wartość)."""
    return np.array([
        roc_auc_score(y[:, i], probs[:, i]) if len(np.unique(y[:, i])) == 2 else np.nan
        for i in range(NUM_CLASSES)
    ])


def summarize(y: np.ndarray, preds: np.ndarray) -> dict:
    """Zbiorcze metryki dla całego zbioru przy podanych decyzjach (0/1)."""
    precision, recall, _, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
    return {
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(y, preds, average="micro", zero_division=0)), 4),
        # Accuracy podajemy tylko informacyjnie: przy rzadkich chorobach jest zawyżona
        "binary_accuracy": round(float((preds == y).mean()), 4),
    }


def evaluate_model(model, splits: dict, run_dir) -> None:
    """Dobiera progi na walidacji i liczy metryki na teście (próg 0.5 kontra progi dla klas)."""
    log = get_logger()
    log.info("evaluate: start")
    x_val, y_val = prepare_images(splits["val"][0]), splits["val"][1].astype(int)
    x_test, y_test = prepare_images(splits["test"][0]), splits["test"][1].astype(int)

    # Progi dobieramy wyłącznie na walidacji, test zostaje nietknięty
    probs_val = model.predict(x_val, batch_size=256, verbose=0)
    thresholds = find_thresholds(y_val, probs_val)
    save_json({name: round(float(t), 4) for name, t in zip(CLASS_NAMES, thresholds)},
              run_dir / "thresholds.json")
    log.info("evaluate: progi dobrane na walidacji: "
             + ", ".join(f"{name}={t:.2f}" for name, t in zip(CLASS_NAMES, thresholds)))

    probs_test = model.predict(x_test, batch_size=256, verbose=0)
    preds_default = (probs_test >= DEFAULT_THRESHOLD).astype(int)
    preds_tuned = (probs_test >= thresholds).astype(int)  # każda kolumna porównana ze swoim progiem
    auc_values = compute_class_auc(y_test, probs_test)

    # Tabela klas: jedna choroba = jeden wiersz, od najgorszej (wg F1)
    p_tuned, r_tuned, f_tuned, support = precision_recall_fscore_support(y_test, preds_tuned, zero_division=0)
    p_default, r_default, f_default, _ = precision_recall_fscore_support(y_test, preds_default, zero_division=0)
    table = pd.DataFrame({
        "class": CLASS_NAMES, "support": support, "auc": auc_values.round(4),
        "threshold": thresholds.round(3),
        "precision": p_tuned.round(4), "recall": r_tuned.round(4), "f1": f_tuned.round(4),
        "precision_05": p_default.round(4), "recall_05": r_default.round(4), "f1_05": f_default.round(4),
    }).sort_values("f1")
    table.to_csv(run_dir / "metrics" / "per_class.csv", index=False)

    metrics = {
        "macro_auc": round(float(np.nanmean(auc_values)), 4),  # AUC nie zależy od progu
        "threshold_0.5": summarize(y_test, preds_default),
        "threshold_per_class": summarize(y_test, preds_tuned),
    }
    save_json(metrics, run_dir / "metrics" / "test_metrics.json")
    log.info(f"evaluate: test: macro AUC = {metrics['macro_auc']}")
    log.info(f"evaluate: test: próg 0.5: {metrics['threshold_0.5']}")
    log.info(f"evaluate: test: progi dla klas: {metrics['threshold_per_class']}")
    worst = table.head(3)
    log.info("evaluate: najsłabsze klasy (F1 z progami dla klas): "
             + ", ".join(f"{row['class']}={row['f1']:.2f}" for _, row in worst.iterrows()))

    _plot_per_class_f1(table, run_dir / "figures" / "per_class_f1.png")
    _save_errors(splits["test"][0], y_test, preds_tuned, run_dir)
    log.info("evaluate: koniec")


def _plot_per_class_f1(table: pd.DataFrame, path) -> None:
    """Rysuje F1 dla każdej choroby: próg 0.5 kontra progi dobrane na walidacji."""
    ax = table.set_index("class")[["f1_05", "f1"]].plot.barh(figsize=(9, 7))
    ax.legend(["próg 0.5", "progi dla klas"])
    ax.set_xlabel("F1 na zbiorze testowym")
    ax.set_title("F1 dla każdej choroby")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _save_errors(test_images: np.ndarray, y_test: np.ndarray, preds: np.ndarray, run_dir) -> None:
    """Zapisuje listę błędnie sklasyfikowanych zdjęć (CSV) i galerię 16 losowych przykładów."""
    log = get_logger()
    errors_per_image = (preds != y_test).sum(axis=1)
    wrong = np.where(errors_per_image > 0)[0]
    log.info(f"evaluate: zdjęcia z przynajmniej jednym błędem: {len(wrong)} z {len(y_test)} "
             f"({len(wrong) / len(y_test):.1%})")

    # Do pliku trafiają tylko indeksy i opisy, bez kopiowania zdjęć na dysk
    pd.DataFrame({
        "test_index": wrong,
        "true_labels": [labels_to_text(y_test[i]) for i in wrong],
        "predicted_labels": [labels_to_text(preds[i]) for i in wrong],
        "n_errors": errors_per_image[wrong],
    }).sort_values("n_errors", ascending=False).to_csv(run_dir / "metrics" / "misclassified.csv", index=False)

    if len(wrong) == 0:
        return
    rng = np.random.default_rng(42)
    sample = rng.choice(wrong, size=min(16, len(wrong)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), sample):
        ax.imshow(test_images[i], cmap="gray")
        title = f"prawda: {labels_to_text(y_test[i])}\nmodel: {labels_to_text(preds[i])}"
        ax.set_title("\n".join(textwrap.fill(line, 28) for line in title.split("\n")), fontsize=6)
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "error_gallery.png", dpi=120)
    plt.close()
