import numpy as np
from PIL import Image
import cv2

LUMINANCE = np.array([0.299, 0.587, 0.114])


def to_luminance(image):
    """
    Convertit une image PIL en luminance Rec. 601
    (tableau HxW de floats dans [0, 1]).
    """
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return array @ LUMINANCE


def to_rgb(luminance):
    """
    Convertit une luminance (HxW dans [0, 1]) en image PIL
    à 3 canaux identiques (R = G = B).
    """
    gray = np.round(np.clip(luminance, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(gray).convert("RGB")



def grayscale(image):
    """Luminance Rec. 601 de l'image, en 3 canaux identiques."""
    return to_rgb(to_luminance(image))


def brightness_normalized(image, target_mean=0.5):
    """
    Normalise la luminosité moyenne de la luminance.

    La luminance est multipliée par target_mean / luminance_moyenne,
    puis clippée à [0, 1]. Sortie : 3 canaux identiques.
    """
    luminance = to_luminance(image)

    current_mean = luminance.mean()

    if current_mean == 0:
        return to_rgb(luminance)

    normalized = np.clip(
        luminance * (target_mean / current_mean),
        0,
        1,
    )

    return to_rgb(normalized)


def contrast_normalized(image):
    """
    Normalise le contraste : (luminance - mean) / std, ramené à [0, 1].
    Sortie : 3 canaux identiques.
    """
    luminance = to_luminance(image)

    mean = luminance.mean()
    std = luminance.std()

    if std == 0:
        return to_rgb(luminance)

    normalized = (luminance - mean) / std

    min_value = normalized.min()
    max_value = normalized.max()

    if max_value == min_value:
        return to_rgb(luminance)

    scaled = (normalized - min_value) / (max_value - min_value)

    return to_rgb(scaled)


def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applique CLAHE (Contrast Limited Adaptive Histogram Equalization)
    sur la luminance. Sortie : 3 canaux identiques.
    """
    gray = np.round(to_luminance(image) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    result = clahe.apply(gray)

    return Image.fromarray(result).convert("RGB")


def histogram_equalized(image):
    """
    Applique une égalisation d'histogramme GLOBALE (cv2.equalizeHist)
    sur la luminance. Sortie : 3 canaux identiques.
    """
    gray = np.round(to_luminance(image) * 255).astype(np.uint8)

    result = cv2.equalizeHist(gray)

    return Image.fromarray(result).convert("RGB")
