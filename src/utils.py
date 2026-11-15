"""Wspólne narzędzia: ścieżki, logowanie i foldery runów."""
import json
import logging
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / "outputs" / "runs"

LOGGER_NAME = "chestmnist"


def create_run_dir(mode: str) -> Path:
    """Tworzy nowy folder runu, np. outputs/runs/20261108_163012_train.

    Każde uruchomienie ma własny folder, więc stare wyniki się nie nadpisują.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_{mode}"
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "metrics").mkdir(parents=True)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    """Włącza logowanie jednocześnie do konsoli i do pliku log.txt w folderze runu."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def get_logger() -> logging.Logger:
    """Zwraca logger projektu (działa po wywołaniu setup_logger)."""
    return logging.getLogger(LOGGER_NAME)


def save_json(data: dict, path: Path) -> None:
    """Zapisuje słownik do pliku JSON (z polskimi znakami)."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def find_latest_model_dir() -> Path | None:
    """Zwraca folder najnowszego runu, w którym jest zapisany model (albo None)."""
    models = sorted(RUNS_DIR.glob("*/model.keras"))  # nazwy zaczynają się od daty, więc sort = chronologicznie
    return models[-1].parent if models else None
