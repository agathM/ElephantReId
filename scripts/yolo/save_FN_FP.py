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
# metrics = model.val(
#     data=os.path.join(YOLO_PATH, "data.yaml"),
#     split="val",
#     project=RESULTS_PATH,
#     name=os.path.join('val',validation_set_version),
#     plots=True,
# )

# =========================================================
# IOU FUNCTION
# =========================================================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# =========================================================
# READ YOLO LABELS
# =========================================================
def read_labels(label_path, w, h):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            boxes.append({
                "cls": int(cls),
                "box": [x1, y1, x2, y2]
            })

    return boxes

# =========================================================
# ANALYSE
# =========================================================
IOU_THRESH = 0.45
CONF_THRESH = 0.25

fp_files = []
fn_files = []

image_paths = sorted(VAL_IMG_DIR.glob("*.jpg"))

for img_path in image_paths:

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    label_path = VAL_LABEL_DIR / f"{img_path.stem}.txt"
    gt_boxes = read_labels(label_path, w, h)

    result = model.predict(
        source=str(img_path),
        conf=CONF_THRESH,
        verbose=False
    )[0]

    pred_boxes = []

    if result.boxes is not None:
        for b in result.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            pred_boxes.append({
                "cls": cls,
                "box": [x1, y1, x2, y2],
                "conf" : float(b.conf[0].item())
            })

    # =========================================================
    # MATCH ALL GT / PRED AND SAVE CONFUSION ERRORS
    # =========================================================

    
    matches = []
    
    # all possible matches based only on IoU
    for i_gt, gt in enumerate(gt_boxes):
        for i_pred, pred in enumerate(pred_boxes):
    
            current_iou = iou(gt["box"], pred["box"])
    
            if current_iou >= IOU_THRESH:
                matches.append(
                    (current_iou, i_gt, i_pred)
                )
                
    
    # highest IoU first
    matches.sort(reverse=True, key=lambda x:x[0])
    
    
    matched_gt = set()
    matched_pred = set()
    
    
    errors = []
    FP = []
    FN = []
    
    
    for current_iou, i_gt, i_pred in matches:
    
        if i_gt in matched_gt or i_pred in matched_pred:
            continue
    
        gt = gt_boxes[i_gt]
        pred = pred_boxes[i_pred]
    
        matched_gt.add(i_gt)
        matched_pred.add(i_pred)
    
        # classification error
        if gt["cls"] != pred["cls"]:
    
            errors.append(
                (
                    CLASS_NAMES[gt["cls"]],
                    CLASS_NAMES[pred["cls"]]
                )
            )
    
    
    # -----------------------------
    # False negatives
    # -----------------------------
    for i_gt, gt in enumerate(gt_boxes):
    
        if i_gt not in matched_gt:
    
            errors.append(
                (
                    CLASS_NAMES[gt["cls"]],
                    "Background"
                )
            )
    
    
    # -----------------------------
    # False positives
    # -----------------------------
    for i_pred, pred in enumerate(pred_boxes):
    
        if i_pred not in matched_pred:
    
            errors.append(
                (
                    "Background",
                    CLASS_NAMES[pred["cls"]]
                )
            )
    
    
    # =========================================================
    # DRAW IMAGE IF ERROR EXISTS
    # =========================================================
    
    if len(errors) > 0:
    
        vis = img.copy()
    
    
        # -------- GT green --------
        for gt in gt_boxes:
    
            x1,y1,x2,y2 = gt["box"]
    
            cv2.rectangle(
                vis,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )
    
            cv2.putText(
                vis,
                f"GT {CLASS_NAMES[gt['cls']]}",
                (x1,max(20,y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )
    
    
        # -------- Predictions red --------
        for pred in pred_boxes:
    
            x1,y1,x2,y2 = pred["box"]
    
            cv2.rectangle(
                vis,
                (x1,y1),
                (x2,y2),
                (0,0,255),
                2
            )
    
            cv2.putText(
                vis,
                f"Pred {CLASS_NAMES[pred['cls']]} {pred['conf']:.2f}",
                (x1,min(h-5,y2+20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,255),
                2
            )
    

        for gt,pred in errors:
            out_dir = OUT_DIR / f"{gt}_to_{pred}"
            os.makedirs(out_dir,exist_ok=True)
            cv2.imwrite(
                        str(out_dir / f"{img_path.stem}.jpg"),
                        vis
                    )