"""Загрузка готовой модели MNIST; при неудаче — обучение CNN на tensorflow.keras.datasets.mnist."""

import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "mnist_cnn.keras"

MODEL_URLS = (
    "https://raw.githubusercontent.com/kj7kunal/MNIST-Keras/master/saved_models/MNIST_keras_CNN.h5",
    "https://raw.githubusercontent.com/hassaan-munir/MNIST/main/mnist_cnn_model.h5",
    "https://raw.githubusercontent.com/robpalgit/DEEP_LEARNING---MNIST_image_classification/master/cnn_mnist_final_model.h5",
    "https://raw.githubusercontent.com/nrmkarthi/MNIST-CNN/main/mnist_cnn_model.h5",
    "https://huggingface.co/spaces/ayaanzaveri/mnist/resolve/c959fe1db8b15ed643b91856cb2db4e2a3125938/mnist-model.h5",
)

INPUT_SHAPE = (28, 28, 1)
OUTPUT_SHAPE = (10,)


def _log(message):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(message, flush=True)


def _download(url, timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    if len(data) < 10_000:
        raise OSError(f"файл слишком мал ({len(data)} байт)")
    return data


def _load_and_validate(path):
    """Загружает модель и проверяет, что она принимает 28x28 и выдаёт 10 классов."""
    from tensorflow import keras

    model = keras.models.load_model(path, compile=False)
    if not model.inputs or not model.outputs:
        raise ValueError("модель без входов или выходов")
    if tuple(model.inputs[0].shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"вход {tuple(model.inputs[0].shape)}, ожидалось (None, 28, 28, 1)")
    if tuple(model.outputs[0].shape[1:]) != OUTPUT_SHAPE:
        raise ValueError(f"выход {tuple(model.outputs[0].shape)}, ожидалось (None, 10)")
    return model


def train_cnn(epochs=3):
    """Обучает простую CNN на MNIST. Возвращает готовую модель."""
    from tensorflow import keras
    from tensorflow.keras import layers

    _log("[model] Загружаю датасет MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, -1)
    x_test = np.expand_dims(x_test.astype(np.float32) / 255.0, -1)

    model = keras.Sequential(
        [
            layers.Input(INPUT_SHAPE),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    _log(f"[model] Точность на тесте: {accuracy:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    _log(f"[model] Модель сохранена: {MODEL_PATH}")
    return model


def get_model():
    """Кэшированная модель -> скачивание готовой -> обучение с нуля."""
    from tensorflow import keras

    if MODEL_PATH.exists():
        try:
            return keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as exc:
            _log(f"[model] Кэш не читается ({exc}), пробую скачать заново")

    for url in MODEL_URLS:
        _log(f"[model] Пробую скачать: {url}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = MODELS_DIR / "candidate.h5"
        try:
            tmp_path.write_bytes(_download(url))
            model = _load_and_validate(tmp_path)
        except Exception as exc:
            _log(f"[model]   не подошла: {exc}")
            tmp_path.unlink(missing_ok=True)
            continue
        model.save(MODEL_PATH)
        tmp_path.unlink(missing_ok=True)
        _log("[model] Готово, модель скачана и проверена.")
        return model

    _log("[model] Ни одна готовая модель не подошла, обучаю CNN на MNIST.")
    return train_cnn()


def predict(model, batch):
    """Возвращает вектор вероятностей длины 10 для батча (1, 28, 28, 1).

    Прямой вызов модели вместо model.predict: на такой маленькой сети
    обёртка predict даёт ~40 мс против ~7 мс, потому что каждый вызов
    пересобирает итератор подачи данных.
    """
    output = model(batch, training=False)
    return np.asarray(output, dtype=np.float64)[0]