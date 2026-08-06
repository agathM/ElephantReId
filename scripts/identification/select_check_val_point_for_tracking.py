import os
import yaml
import argparse
import torch
from pathlib import Path

MODEL_CONFIG = Path("/pbs/home/a/ameurisse/ElephantReId/config/train_classifiers")
TRACK_CONFIG = Path("/pbs/home/a/ameurisse/ElephantReId/config/TrackIn") / 

with open(MODEL_CONFIG / args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

with open(TRACK_CONFIG / args.config, "r") as f:
        track_cfg = yaml.safe_load(f)
    CHECKPOINTS = track_cfg["checkpoints"]


# ============================================================
# Paths from config
# ============================================================


version = cfg['version']
crop_type = cfg["crop_type"]



RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers")


results_folder = (
    RESULTS_PATH
    / version
    / crop_type
)


PREDICTIONS_FILE = (
    results_folder
    / "val_predictions.pt"
)

print(f"Loading predictions: {PREDICTIONS_FILE}")


# ============================================================
# Load predictions
# ============================================================

data = torch.load(
    PREDICTIONS_FILE,
    map_location="cpu"
)


outputs = data["outputs"]
labels = data["labels"]
class_names = data["class_names"]
val_loss = data["val_loss"]


# ============================================================
# Compute predictions
# ============================================================

predicted = torch.argmax(
    outputs,
    dim=1
)

# ============================================================
# Select validation points
# ============================================================

selected = []

allowed_individuals = cfg.get(
    "trackin_individuals",
    []
)


for i, (pred, gt, out) in enumerate(
    zip(predicted, labels, outputs)
):

    pred = pred.item()
    gt = gt.item()


    # Ground truth filtering

    if allowed_individuals:

        gt_name = class_names[gt]

        if gt_name not in allowed_individuals:
            continue


    probs = torch.softmax(
        out,
        dim=0
    )


    confidence = probs[pred].item()


    top2 = torch.topk(
        probs,
        2
    ).values


    margin = (
        top2[0]-top2[1]
    ).item()


    # Prediction filtering

    if cfg["correct_prediction_only"]:

        if pred != gt:
            continue


    if cfg["wrong_prediction_only"]:

        if pred == gt:
            continue


    # Confidence filtering

    if confidence < cfg["conf_thresh"]:
        continue


    if margin < cfg["margin_thresh"]:
        continue


    selected.append(i)

torch.save(selected, results_folder / "val_indices_for_tracking.pt")

# ============================================================
# Select checkpoints
# ============================================================

diff_losses = [val_loss[0]] + [val_loss[i+1] - val_loss[i] for i in range(len(val_loss)-1)]
print(diff_losses)

    