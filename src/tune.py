"""Tuning hiperparametrów: sprawdzamy wszystkie kombinacje z małej siatki (grid search)."""
import itertools

import pandas as pd
from tensorflow import keras

from src.train import DEFAULT_PARAMS, train_model
from src.utils import get_logger, save_json

# Wartości do sprawdzenia: 3 x 3 x 3 x 3 = 81 kombinacji (siatka rośnie jak iloczyn wartości)
TUNE_GRID = {
    "learning_rate": [5e-3, 2e-3, 1e-3],  # model uczy się wolno, więc sprawdzamy większe kroki
    "dropout": [0.3, 0.4, 0.5],
    "filters": [32, 48, 64],
    "batch_size": [64, 128, 256],  # mniejsza paczka = więcej korekt wag na epokę, ale wolniej
}
TUNE_EPOCHS = 8  # krótki trening na kombinację (za mało epok faworyzuje tylko szybki start)


def run_tuning(splits: dict, run_dir) -> dict:
    """Trenuje model dla każdej kombinacji, ocenia na walidacji i zwraca najlepszą.

    Wyniki trafiają do metrics/tuning_results.csv, najlepsza konfiguracja do best_config.json.
    Zbiór testowy nie jest tu używany.
    """
    log = get_logger()
    keys = list(TUNE_GRID)
    combinations = list(itertools.product(*TUNE_GRID.values()))
    log.info(f"tune: sprawdzam {len(combinations)} kombinacji po {TUNE_EPOCHS} epok: {TUNE_GRID}")

    results = []
    for number, values in enumerate(combinations, start=1):
        chosen = dict(zip(keys, values))
        params = {**DEFAULT_PARAMS, **chosen, "epochs": TUNE_EPOCHS}
        log.info(f"tune: kombinacja {number}/{len(combinations)}: {chosen}")

        _, history = train_model(splits, params)  # bez run_dir = bez zapisu modelu
        best_val_auc = float(max(history["val_auc"]))
        results.append({**chosen, "best_val_auc": best_val_auc})
        log.info(f"tune: kombinacja {number}: najlepszy val_auc={best_val_auc:.4f}")
        keras.backend.clear_session()  # zwalnia pamięć po poprzednim modelu

    results_table = pd.DataFrame(results).sort_values("best_val_auc", ascending=False)
    results_table.to_csv(run_dir / "metrics" / "tuning_results.csv", index=False)

    best = max(results, key=lambda row: row["best_val_auc"])
    best_params = {key: best[key] for key in keys}
    save_json(best_params, run_dir / "best_config.json")
    log.info(f"tune: najlepsza konfiguracja: {best_params} (val_auc={best['best_val_auc']:.4f})")
    return best_params
