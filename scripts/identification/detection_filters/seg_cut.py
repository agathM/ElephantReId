"""For ear segmentation only
filter cut ears on three parameters :
the max cut score of all side cut score
side cut score being the length of the biggest consecutive number of pixels (horizontal or vertical) on the sum of pixels of one side (horizontal or vertical)
the horizontal and vertical cut ratio corresponding to the length of the side of the maximum cut score on the perpendicular dimension (height or width)"""

import numpy as np
import pandas as pd
from PIL import Image
import os

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FEATURE_COLS = ["max_cut_score", "cut_line", "cut_ratio","ratio", "area"]

MAX_CUT_SCORE_THRESH = 0.8
MAX_CUT_RATIO = {"horizontal": 1, "vertical": 1.6}

def get_mask(image):
    """Retourne un masque des pixels dont au moins un canal est non nul."""
    array = np.asarray(image)
    if array.ndim == 3:
        return np.any(array != 0, axis=2)
    return array != 0


def largest_contiguous(values):
    """Retourne la longueur de la plus longue suite de valeurs True."""
    max_run = 0
    current_run = 0
    for value in values:
        if value:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def compute_cut_scores(mask):
    """Retourne les cut scores (left, right, top, bottom) d'un masque."""
    height, width = mask.shape
    vertical = int(mask.sum(axis=0).max())
    horizontal = int(mask.sum(axis=1).max())
    rows_with_pixels = np.any(mask, axis=1)
    cols_with_pixels = np.any(mask, axis=0)

    if vertical == 0 or horizontal == 0:
        return (0.0, 0.0, 0.0, 0.0)

    first_col = int(np.argmax(cols_with_pixels))
    last_col = width - 1 - int(np.argmax(cols_with_pixels[::-1]))
    first_row = int(np.argmax(rows_with_pixels))
    last_row = height - 1 - int(np.argmax(rows_with_pixels[::-1]))

    return (
        largest_contiguous(mask[:, first_col]) / vertical, #left
        largest_contiguous(mask[:, last_col]) / vertical, #right
        largest_contiguous(mask[first_row, :]) / horizontal, #up
        largest_contiguous(mask[last_row, :]) / horizontal, #bottom
    )


def round_to_two_significant_digits(value):
    return float(f"{value:.2g}")


def keep_image(image_path):
    """Return True if image is to be kept"""
    if image_path is None:
        return pd.Series(
            {column: np.nan for column in FEATURE_COLS}
        )
    with Image.open(image_path) as image:
        width, height = image.size
        mask = get_mask(image)

    max_score = round_to_two_significant_digits(max(compute_cut_scores(mask)))
    cut_line = "horizontal" if np.argmax(compute_cut_scores(mask)) > 1 else "vertical"
    cut_ratio = round_to_two_significant_digits(width / height) if np.argmax(compute_cut_scores(mask)) > 1 \
        else round_to_two_significant_digits(height / width)
    if max_score>MAX_CUT_SCORE_THRESH:
        return MAX_CUT_RATIO[cut_line]>cut_ratio
    return True

def main(df, crops_dir, model):
    def get_image_path(row):
        img_path = crops_dir / model / row["detection"] / row["image"]
        if not os.path.isfile(img_path):
            img_path = crops_dir / model / row["image"]

            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

        return img_path

    to_keep = df.apply(
        lambda row: keep_image(get_image_path(row)),
        axis=1
    )

    return df.loc[to_keep].copy()

