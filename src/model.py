"""Definicja sieci CNN."""
from tensorflow import keras
from tensorflow.keras import layers

from src.dataset import NUM_CLASSES


def build_model(filters: int, dropout: float, learning_rate: float) -> keras.Model:
    """Buduje małą sieć CNN, która ocenia 14 chorób jednocześnie.

    filters: liczba filtrów w pierwszym bloku (drugi ma ich dwa razy więcej).
    dropout: jaka część neuronów jest losowo wyłączana podczas treningu.
    learning_rate: jak duże kroki robi optymalizator Adam przy uczeniu.
    """
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),  # obraz 28x28, jeden kanał (skala szarości)

        # Augmentacja działa tylko podczas treningu (przy predykcji jest wyłączona)
        layers.RandomRotation(0.05),  # losowy lekki obrót, do ok. 18 stopni
        layers.RandomZoom(0.1),       # losowe przybliżenie lub oddalenie do 10%

        # Blok 1: proste cechy, np. krawędzie i jasne/ciemne plamy
        layers.Conv2D(filters, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling2D(),  # zmniejsza obraz dwa razy, zostawia najsilniejsze sygnały

        # Blok 2: bardziej złożone kształty, stąd więcej filtrów
        layers.Conv2D(filters * 2, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),                        # rozwija obraz w jedną długą listę liczb
        layers.Dense(64, activation="relu"),     # 64 neurony łączą cechy w decyzję
        layers.Dropout(dropout),                 # chroni przed wkuwaniem zbioru treningowego

        # 14 wyjść sigmoid: każda choroba ma własne prawdopodobieństwo 0-1, niezależnie od reszty
        layers.Dense(NUM_CLASSES, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",  # błąd liczony osobno dla każdej choroby (tak/nie)
        # AUC liczone dla każdej choroby i uśrednione (macro), nie zależy od progu
        metrics=[keras.metrics.AUC(multi_label=True, num_labels=NUM_CLASSES, name="auc")],
    )
    return model
