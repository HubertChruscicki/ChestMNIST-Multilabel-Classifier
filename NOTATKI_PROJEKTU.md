# Notatki projektu (pod raport)

## Decyzje projektowe

- **Dane:** ChestMNIST z biblioteki `medmnist`, obrazy 28x28 w skali szarości, 14 etykiet, klasyfikacja wieloetykietowa. Używane oficjalne podziały train/val/test.
- **Przetwarzanie:** obrazy w uint8 (0–255) są dzielone przez 255 do zakresu 0–1 i dostają kanał, bo sieć oczekuje tensora 4D.
- **Model:** mała CNN w Keras (2 bloki Conv+MaxPool, Dense 64 z Dropout, 14 wyjść sigmoid), funkcja straty BinaryCrossentropy, optymalizator Adam, augmentacja (lekki obrót i zoom).
- **Metryki:** główna to macro AUC. Dodatkowo F1, precision i recall dla każdej klasy. Accuracy nie jest miarą główną, bo przy wielu etykietach i rzadkich chorobach jest myląca.
- **Niezbalansowanie klas:** próg decyzyjny dobierany osobno dla każdej choroby na zbiorze walidacyjnym, na podstawie krzywej Precision-Recall (maksymalne F1). Powód: próg 0.5 zakłada zbalansowane klasy, a przy rzadkich chorobach prawdopodobieństwa pozytywów są niskie, więc recall spada. Krzywa PR pokazuje kompromis między precision a recall. Przy wielu negatywach ROC daje zawyżony obraz jakości. Zbiór testowy nie służy do doboru progów.
- **Raportowanie:** wynik podajemy dla progu 0.5 i dla progów dobranych. Różnica pokazuje wpływ progów.
- **Tuning hiperparametrów:** siatka (learning rate, dropout, liczba filtrów), krótki trening, ocena na walidacji. Uruchamiany flagą `--with-tune` w trybie `all` albo osobnym trybem `tune`.
- **Odrzucone:** `class_weight` (Keras nie obsługuje go poprawnie dla wielu etykiet naraz), SHAP i LIME (za ciężkie na ten projekt).

## Log postępów

- 08.11.2026: ustalenie zakresu, struktury projektu i metod.
- 08.11.2026: implementacja pipeline'u (preprocessing, train, tune, evaluate, flagi, logi, foldery runów).
- 08.11.2026: dane po sprawdzeniu: train 78468, val 11219, test 22433 zdjęć 28x28, brak braków, etykiety tylko 0/1, brak całkiem czarnych obrazów, piksele 0–253 (uint8). 54% zdjęć bez żadnej choroby, średnio 0.72 choroby na zdjęcie. Najczęstsza klasa (infiltration, 13914) jest 97x częstsza niż najrzadsza (hernia, 144).
- 08.11.2026: run 1 (`all`, domyślne parametry: filtry 32, dropout 0.5, lr 1e-3, 15 epok, bez tuningu): val AUC 0.702 po 15 epokach i nadal rośnie, test macro AUC 0.702. Przy progu 0.5 model nie wskazuje żadnej choroby (F1 = 0, accuracy 0.947, bo większość etykiet to 0). Progi dobrane na walidacji dają macro F1 0.159 (recall 0.38, precision 0.11). Najsłabsze klasy: hernia, pneumonia, fibrosis.
- 08.11.2026: tuning (grid 3x3x3x3 = 81 kombinacji: lr {5e-3, 2e-3, 1e-3}, dropout {0.3, 0.4, 0.5}, filtry {32, 48, 64}, batch {64, 128, 256}, po 8 epok, ocena na val AUC, ok. 3 godziny na CPU). Najlepsza konfiguracja: lr 2e-3, dropout 0.3, filtry 48, batch 64 (val AUC 0.7065). Wyniki wszystkich kombinacji mieszczą się w 0.66–0.71. Trendy: dropout 0.3 lepszy od 0.4 i 0.5, batch 64 lepszy od większych, lr 5e-3 najsłabszy. Wszystkie 5 najlepszych kombinacji ma dropout 0.3 i batch 64.
- 08.11.2026: run 2 (`all --with-tune --epochs 30`, najlepsza konfiguracja z tuningu): EarlyStopping po 12 epokach, najlepsza epoka 9 (val AUC 0.7101). Test macro AUC 0.709 (run 1: 0.702). Przy progu 0.5 nadal F1 = 0. Z progami dla klas macro F1 0.165 (run 1: 0.159), recall 0.31, precision 0.12. Najsłabsze klasy bez zmian: hernia, pneumonia, fibrosis. Zysk z tuningu jest mały. AUC na train (0.686) jest niższe niż na val (0.710), co wskazuje na niedouczenie (augmentacja i dropout hamują naukę małej sieci), a nie przeuczenie.
