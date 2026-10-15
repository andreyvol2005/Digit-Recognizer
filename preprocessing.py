"""Преобразование холста pygame в тензор 28x28, пригодный для MNIST.

Холст: белая «бумага», чёрные штрихи. Инверсия выполняется ровно один раз,
в to_mnist_polarity: чёрные чернила -> 1.0, белая бумага -> 0.0.
"""

import numpy as np
import pygame
from PIL import Image

SIZE = 28
INK_THRESHOLD = 0.1
MAX_DIGIT = 20
TARGET_CENTER = SIZE / 2.0

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_LANCZOS = Image.LANCZOS


def canvas_to_gray(canvas):
    """Уменьшает Surface до 28x28 и переводит в оттенки серого."""
    small = pygame.transform.smoothscale(canvas, (SIZE, SIZE))
    rgb = np.transpose(pygame.surfarray.array3d(small), (1, 0, 2))
    return Image.fromarray(rgb).convert("L")


def to_mnist_polarity(gray):
    """Инверсия: чёрные чернила -> 1.0, белая бумага -> 0.0."""
    return 1.0 - np.asarray(gray, dtype=np.float32) / 255.0


def _to_image(arr):
    """float в полярности MNIST -> изображение, где чернила белые."""
    return Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255))


def _from_image(image):
    """Изображение с белыми чернилами -> float в полярности MNIST (без инверсии)."""
    return np.asarray(image, dtype=np.float32) / 255.0


def _bbox(arr):
    mask = arr > INK_THRESHOLD
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def is_blank(gray):
    return _bbox(to_mnist_polarity(gray)) is None


def _fit_to_canvas(gray):
    """Обрезает по bounding box и вписывает цифру в 20x20 по центру поля."""
    arr = to_mnist_polarity(gray)
    box = _bbox(arr)
    if box is None:
        return np.zeros((SIZE, SIZE), dtype=np.float32)

    top, left, bottom, right = box
    width, height = right - left, bottom - top
    scale = MAX_DIGIT / max(width, height)
    new_w = max(1, min(MAX_DIGIT, int(round(width * scale))))
    new_h = max(1, min(MAX_DIGIT, int(round(height * scale))))

    digit = _to_image(arr[top:bottom, left:right]).resize((new_w, new_h), RESAMPLE_LANCZOS)
    canvas = Image.new("L", (SIZE, SIZE), 0)
    canvas.paste(digit, ((SIZE - new_w) // 2, (SIZE - new_h) // 2))
    return _from_image(canvas)


def _center_of_mass(arr):
    total = float(arr.sum())
    if total <= 0.0:
        return TARGET_CENTER, TARGET_CENTER
    grid_y, grid_x = np.mgrid[0:SIZE, 0:SIZE]
    return float((grid_y * arr).sum() / total), float((grid_x * arr).sum() / total)


def _shift(arr, dy, dx):
    return _from_image(
        _to_image(arr).transform(
            (SIZE, SIZE),
            Image.AFFINE,
            (1, 0, -dx, 0, 1, -dy),
            resample=RESAMPLE_BILINEAR,
            fillcolor=0,
        )
    )


def center_by_mass(gray, iterations=3):
    """Сдвигает цифру так, чтобы центр масс оказался в геометрическом центре поля.

    Итерации компенсируют небольшой сдвиг массы, который вносит билинейная
    интерполяция при субпиксельном сдвиге и обрезание по краям поля.
    """
    arr = _fit_to_canvas(gray)
    for _ in range(iterations):
        cy, cx = _center_of_mass(arr)
        dy, dx = TARGET_CENTER - cy, TARGET_CENTER - cx
        if abs(dy) < 0.02 and abs(dx) < 0.02:
            break
        arr = _shift(arr, dy, dx)
    return np.clip(arr, 0.0, 1.0)


def prepare(gray):
    """Возвращает батч (1, 28, 28, 1) float32 со значениями 0..1."""
    return center_by_mass(gray).reshape(1, SIZE, SIZE, 1).astype(np.float32)


def to_ascii(arr):
    """Текстовое представление 28x28 полотна — для отладки."""
    ramp = " .:-=+*#%@"
    flat = np.clip(arr, 0.0, 1.0).reshape(SIZE, SIZE)
    return "\n".join("".join(ramp[int(v * (len(ramp) - 1))] for v in row) for row in flat)