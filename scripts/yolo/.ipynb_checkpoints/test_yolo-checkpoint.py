from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================================================
# PATHS
# =========================================================
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/yolo/v4")
MODEL_PATH = RESULTS_PATH / "weights/best.pt"
YOLO_PATH = "/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/yolo"

VAL_IMG_DIR = Path(YOLO_PATH) / "images/val"
VAL_LABEL_DIR = Path(YOLO_PATH) / "labels/val"

OUT_DIR = RESULTS_PATH / Path("errors")
os.makedirs(OUT_DIR,exist_ok=True)

CLASS_NAMES = {0: "Head", 1: "Left Ear", 2: "Right Ear"}

# =========================================================
# LOAD MODEL
# =========================================================
model = YOLO(MODEL_PATH)

# IF NEW VALIDATION SET
validation_set_version = "v1"
metrics = model.val(
    data=os.path.join(YOLO_PATH, "data.yaml"),
    split="val",
    project=RESULTS_PATH,
    name=os.path.join('val',validation_set_version),
    plots=True,
    visualize=True,
    show_labels=True,
    show_conf=True
)