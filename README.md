# ChestMNIST: klasyfikacja wieloetykietowa chorób płuc

Prosty projekt studencki (Uczenie Maszynowe w Diagnostyce Medycznej). Sieć CNN w TensorFlow/Keras rozpoznaje 14 chorób na zdjęciach RTG klatki piersiowej (zbiór ChestMNIST z MedMNIST, obrazy 28x28). Jedno zdjęcie może mieć kilka chorób naraz.

## Instalacja

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Dane pobierają się same do `data/` przy pierwszym uruchomieniu.

## Uruchomienie

```
python main.py --mode all                # preprocessing -> trening -> ewaluacja
python main.py --mode all --with-tune    # to samo, ale z tuningiem hiperparametrów przed treningiem
python main.py --mode preprocessing      # tylko kontrola danych i wykresy
python main.py --mode tune               # tylko tuning (zapisuje best_config.json)
python main.py --mode train              # tylko trening (domyślne parametry)
python main.py --mode evaluate           # bez treningu: ewaluacja najnowszego modelu
```

Dodatkowe flagi:
- `--epochs`, `--learning-rate`, `--dropout`, `--filters`, `--batch-size` nadpisują parametry treningu.
- `--config <plik.json>` wczytuje parametry z pliku, np. `outputs/runs/<run>/best_config.json`.
- `--run <folder>` wskazuje run z modelem dla `--mode evaluate` (domyślnie najnowszy).

Każde uruchomienie tworzy nowy folder w `outputs/runs/`, więc nic się nie nadpisuje. Opis wyników: `outputs/README.md`.

## Struktura

```
main.py              # wejście, wybór trybu flagą
requirements.txt
data/                # pobrany zbiór (poza gitem)
outputs/runs/        # wyniki każdego uruchomienia
src/
  dataset.py         # pobieranie, wczytanie, skalowanie
  preprocessing.py   # kontrola danych, statystyki, wykresy
  model.py           # definicja CNN
  train.py           # trening
  tune.py            # tuning hiperparametrów (grid search)
  evaluate.py        # progi, metryki, analiza błędów
  utils.py           # logger, foldery runów
NOTATKI_PROJEKTU.md  # notatki pod raport
```
