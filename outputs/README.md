# Wyniki uruchomień

Każde uruchomienie `main.py` tworzy folder `runs/<data_godzina>_<tryb>/`, np. `runs/20261108_163012_all/`. Stare foldery zostają, więc runy można porównywać.

## Co jest w folderze runu

Zawartość zależy od trybu (`preprocessing`, `tune`, `train`, `evaluate`, `all`).

| Plik | Co zawiera | Tryb |
|---|---|---|
| `log.txt` | pełny zapis etapów: liczby zdjęć, kształty, statystyki, wyniki epok, metryki | każdy |
| `config.json` | użyty tryb i parametry | każdy |
| `model.keras` | zapisany model | train, all |
| `history.csv` | loss i AUC po każdej epoce | train, all |
| `best_config.json` | najlepsze parametry z tuningu | tune, all --with-tune |
| `thresholds.json` | progi decyzyjne dobrane dla każdej choroby na walidacji | evaluate, all |
| `figures/class_distribution.png` | odsetek zdjęć z daną chorobą (niezbalansowanie klas) | preprocessing, all |
| `figures/pixel_histogram.png` | histogram jasności pikseli | preprocessing, all |
| `figures/sample_images.png` | 16 losowych zdjęć z etykietami | preprocessing, all |
| `figures/learning_curves.png` | loss i AUC w epokach (train kontra val) | train, all |
| `figures/per_class_f1.png` | F1 każdej choroby: próg 0.5 kontra progi dla klas | evaluate, all |
| `figures/error_gallery.png` | 16 losowych zdjęć sklasyfikowanych z błędem | evaluate, all |
| `metrics/preprocessing_summary.json` | kształty, typy, zakresy, braki, etykiety | preprocessing, all |
| `metrics/class_counts.csv` | liczba zdjęć z każdą chorobą w train/val/test | preprocessing, all |
| `metrics/tuning_results.csv` | wynik (val AUC) każdej kombinacji parametrów | tune, all --with-tune |
| `metrics/test_metrics.json` | metryki zbiorcze na teście | evaluate, all |
| `metrics/per_class.csv` | tabela chorób (opis poniżej) | evaluate, all |
| `metrics/misclassified.csv` | lista zdjęć z błędem: indeks, prawdziwe i przewidziane choroby | evaluate, all |

## Jak czytać `per_class.csv`

Jeden wiersz to jedna choroba, posortowane od najsłabszej (wg F1).

- `support`: ile zdjęć z tą chorobą jest w teście.
- `auc`: jak dobrze model odróżnia tę chorobę od reszty, niezależnie od progu.
- `threshold`: próg dobrany dla tej choroby na walidacji.
- `precision`: z uznanych za chorych, ile faktycznie było chorych.
- `recall`: z faktycznie chorych, ile model wykrył.
- `f1`: jedna liczba łącząca precision i recall (bliżej 1 = lepiej).
- kolumny z `_05` to te same miary przy standardowym progu 0.5, do porównania.

## Jak czytać `log.txt`

Każda linia ma godzinę i prefiks etapu: `dataset`, `preprocessing`, `tune`, `train`, `evaluate`. Liczby zdjęć na każdym etapie widać w liniach `dataset` i `preprocessing`. Wynik każdej epoki to linia `train: epoka N`.

## Zasada: walidacja i test

Parametry i progi dobieramy na zbiorze walidacyjnym. Zbiór testowy służy tylko do końcowej oceny.
