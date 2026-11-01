import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

import cv2

LUMINANCE = np.array([0.299, 0.587, 0.114])


def transform_original(image):
    """Image originale, aucune transformation."""
    return image


def transform_grayscale(image):
    """Image en niveaux de gris, reconvertie en RGB (3 canaux)."""
    return image.convert("L").convert("RGB")


def transform_brightness_normalized(image, target_mean=0.5):
    """
    Normalise la luminosité moyenne de l'image.

    Chaque canal RGB est multiplié par target_mean / luminosité_moyenne.
    Même principe que normalize_mean_brightness de montage_grayscale.py.
    """
    array = np.asarray(image).astype(np.float32) / 255.0

    luminance = array @ LUMINANCE
    current_mean = luminance.mean()

    if current_mean == 0:
        return image

    factor = target_mean / current_mean

    normalized = np.clip(array * factor, 0, 1)

    return Image.fromarray(
        (normalized * 255).astype(np.uint8)
    )


def transform_contrast_normalized(image):
    """
    Normalise le contraste : (luminance - mean) / std, ramené à [0, 1]
    pour la visualisation. L'image de sortie est en niveaux de gris.
    Même principe que normalize_contrast de montage_grayscale.py.
    """
    array = np.asarray(image).astype(np.float32) / 255.0

    luminance = array @ LUMINANCE
    mean = luminance.mean()
    std = luminance.std()

    if std == 0:
        return image

    normalized = (luminance - mean) / std

    min_value = normalized.min()
    max_value = normalized.max()

    if max_value == min_value:
        return image

    scaled = (normalized - min_value) / (max_value - min_value)
    gray = np.stack([scaled] * 3, axis=-1)

    return Image.fromarray(
        (gray * 255).astype(np.uint8)
    )


def transform_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applique CLAHE (Contrast Limited Adaptive Histogram Equalization)
    sur l'image en niveaux de gris, puis reconvertit en RGB.
    Mêmes paramètres que montage_grayscale.py.
    """
    gray = np.asarray(image.convert("L"))

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    result = clahe.apply(gray)

    return Image.fromarray(result).convert("RGB")


def transform_histogram_equalized(image):
    """
    Applique une histogram equalization GLOBALE (cv2.equalizeHist) sur
    l'image en niveaux de gris, puis reconvertit en RGB.

    Contrairement à CLAHE, l'égalisation est calculée sur l'histogramme
    global de l'image (pas par tuiles locales).
    """
    gray = np.asarray(image.convert("L"))

    result = cv2.equalizeHist(gray)

    return Image.fromarray(result).convert("RGB")