import torch
from torchvision import models, transforms
from pathlib import Path
import argparse
import yaml
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report,
)
import pandas as pd

from elephantreid.identification import *

# PARAMETERS
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers")
FEATURES_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features")
CROP_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops")
MODEL_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/models/identification")
CONFIG_PATH = Path("/pbs/home/a/ameurisse/ElephantReId/config/train_classifiers")

model_name = "ResNet50_v3"


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate classifier"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="crop type config file"
    )
    args = parser.parse_args()

    with open(CONFIG_PATH / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    features_folder = FEATURES_PATH / model_name / cfg["features_folder"]

    selection_type = cfg["selection_type"]
    crop_type = cfg["crop_type"]
    date = cfg["date"]

    results_folder = (
        RESULTS_PATH
        / model_name
        / selection_type
        / date
        / crop_type
    )

    pred_file = results_folder / "val_predictions.pt"

    if not pred_file.exists():
        print("No saved predictions found.")
        return

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------

    data = torch.load(pred_file, map_location="cpu")

    outputs = data["outputs"]
    labels = data["labels"].numpy()
    class_names = data["class_names"]

    predictions = outputs.argmax(dim=1).numpy()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    accuracy = accuracy_score(labels, predictions)

    f1_macro = f1_score(
        labels,
        predictions,
        average="macro"
    )

    f1_weighted = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"F1 macro    : {f1_macro:.4f}")
    print(f"F1 weighted : {f1_weighted:.4f}")

    # ------------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------------

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=3,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        results_folder / "classification_report.csv"
    )

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    cm = confusion_matrix(labels, predictions)

    save_confusion_matrix(
        cm,
        class_names,
        results_folder,
        police_size
    )

    print("\nEvaluation saved to:")
    print(results_folder)


if __name__ == "__main__":
    main()