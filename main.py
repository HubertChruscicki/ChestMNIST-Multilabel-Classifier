"""Punkt wejścia projektu. Tryb pracy wybieramy flagą --mode (opis w README.md)."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # mniej komunikatów TensorFlow w konsoli

import matplotlib

matplotlib.use("Agg")  # wykresy tylko zapisujemy do plików, bez otwierania okienek

from tensorflow import keras

from src.dataset import load_splits
from src.evaluate import evaluate_model
from src.preprocessing import run_preprocessing
from src.train import DEFAULT_PARAMS, train_model
from src.tune import run_tuning
from src.utils import create_run_dir, find_latest_model_dir, get_logger, save_json, setup_logger

MODES = ["preprocessing", "train", "evaluate", "tune", "all"]


def parse_args() -> argparse.Namespace:
    """Czyta flagi z linii poleceń."""
    parser = argparse.ArgumentParser(description="ChestMNIST: klasyfikacja wieloetykietowa chorób płuc")
    parser.add_argument("--mode", choices=MODES, required=True, help="co uruchomić")
    parser.add_argument("--with-tune", action="store_true", help="tylko z --mode all: dodaje tuning przed treningiem")
    parser.add_argument("--config", type=Path, help="plik JSON z parametrami, np. best_config.json z tuningu")
    parser.add_argument("--run", type=Path, help="folder runu z modelem dla --mode evaluate (domyślnie najnowszy)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--filters", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.with_tune and args.mode != "all":
        parser.error("--with-tune działa tylko z --mode all (osobno służy --mode tune)")
    return args


def build_params(args: argparse.Namespace) -> dict:
    """Składa parametry: domyślne, potem z pliku --config, na końcu z flag (flagi wygrywają)."""
    params = dict(DEFAULT_PARAMS)
    if args.config:
        params.update(json.loads(args.config.read_text(encoding="utf-8")))
    from_flags = {
        "epochs": args.epochs, "learning_rate": args.learning_rate, "dropout": args.dropout,
        "filters": args.filters, "batch_size": args.batch_size,
    }
    params.update({key: value for key, value in from_flags.items() if value is not None})
    return params


def run_pipeline(args: argparse.Namespace, params: dict, model_dir, run_dir: Path) -> None:
    """Wykonuje kolejne etapy zależnie od trybu."""
    log = get_logger()
    splits = load_splits()
    model = None

    if args.mode in ("preprocessing", "all"):
        run_preprocessing(splits, run_dir)

    if args.mode == "tune" or (args.mode == "all" and args.with_tune):
        best_params = run_tuning(splits, run_dir)
        params = {**params, **best_params}  # trening użyje najlepszych parametrów z tuningu
        save_json({"mode": args.mode, "with_tune": args.with_tune, "params": params}, run_dir / "config.json")

    if args.mode in ("train", "all"):
        model, _ = train_model(splits, params, run_dir)

    if args.mode == "evaluate":
        log.info(f"evaluate: ładuję model z {model_dir}")
        model = keras.models.load_model(model_dir / "model.keras", compile=False)

    if args.mode in ("evaluate", "all"):
        evaluate_model(model, splits, run_dir)


def main() -> None:
    args = parse_args()
    params = build_params(args)

    model_dir = None
    if args.mode == "evaluate":
        model_dir = args.run or find_latest_model_dir()
        if model_dir is None or not (model_dir / "model.keras").exists():
            raise SystemExit("Nie znaleziono zapisanego modelu. Najpierw uruchom --mode train (albo all).")

    run_dir = create_run_dir(args.mode)
    log = setup_logger(run_dir)
    save_json({"mode": args.mode, "with_tune": args.with_tune, "params": params,
               "model_source": str(model_dir) if model_dir else None}, run_dir / "config.json")
    log.info(f"start: tryb {args.mode}, wyniki w {run_dir}")

    try:
        run_pipeline(args, params, model_dir, run_dir)
    except Exception:
        log.exception("BŁĄD: program zatrzymał się z wyjątkiem")  # ślad trafia też do log.txt
        raise
    log.info(f"koniec: wyniki w {run_dir}")


if __name__ == "__main__":
    main()
