import os
import argparse
import yaml
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd

from elephantreid.identification import *

# ============================================================
# PARAMETERS
# ============================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers")
FEATURES_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features")
CROP_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops")
MODEL_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/models/identification")
CONFIG_PATH = Path("/pbs/home/a/ameurisse/ElephantReId/config/train_classifiers")

model_name = "ResNet50_v3"

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Generate Grad-CAM figures")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Training config file",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    selection_type = cfg["selection_type"]
    crop_type = cfg["crop_type"]
    date = cfg["date"]
    crop_version = cfg["crop_version"]
    split_version = cfg["split_version"]
    mode = cfg["mode"]
    version = cfg["version"]
    
    crop_path = CROP_PATH / crop_type / crop_version
    split_file = CROP_PATH / cfg["split_file"]
    
    features_folder = FEATURES_PATH / version / crop_type
    
    results_folder = RESULTS_PATH / version / crop_type
    
    model_path = (
        MODEL_PATH
        / version
        / crop_type
        / "model.pth"
    )

    if not model_path.exists():
        print("Model not found.")
        return

    # ============================================================
    # LOAD MODEL
    # ============================================================

    base_model = models.resnet50(weights="DEFAULT")

    train_features_dataset, val_features_dataset, train_data, val_data, _ = load_data(
        base_model,
        input_transform,
        crop_path,
        split_file,
        crop_type,
        DEVICE,
        features_folder,
        mode,
    )

    class_names = train_data.classes


    base_model.fc = ClassificationHead(
        dim_in=2048,
        dim_out=train_data.classes_nb,
    )

    state_dict = torch.load(model_path, map_location=DEVICE)

    base_model.load_state_dict(state_dict)

    base_model.to(DEVICE)
    base_model.eval()

    # ============================================================
    # TRAIN GRADCAM
    # ============================================================

    df = pd.read_excel(
        split_file,
        sheet_name="samples",
        engine="odf",
    )
    if mode:
        df = df[df["mode"] == mode].reset_index(drop=True)

    df["crop_path"] = (
        crop_path
        / df["crop_id"]
    )

    train_df = df[df["split"] == "train"].copy()

    train_df = (
        train_df.groupby("indiv_name", group_keys=False)
        .apply(lambda g: g.iloc[::5])
        .reset_index(drop=True)
    )

    for _, row in train_df.iterrows():

        img_path = Path(row["crop_path"])

        img = Image.open(img_path).convert("RGB")
        img = input_transform(img)

        label = train_data.label_map[row["indiv_name"]]

        out_dir = (
            results_folder
            / "grad_cam"
            / "train"
            / class_names[label]
        )

        os.makedirs(out_dir, exist_ok=True)

        outfile = out_dir / f"{img_path.parent.name}_{img_path.name}"

        save_grad_cam(
            base_model,
            class_names,
            img.unsqueeze(0),
            label,
            out_name=outfile,
        )

    # ============================================================
    # VALIDATION GRADCAM
    # ============================================================

    df = pd.read_csv(results_folder / "val_pred_metadata.csv")

    df["crop_path"] = (
        crop_path
        / df["crop_id"]
    )

    df["conf_pred_label"] = df["conf_pred_label"].astype(float)

    val_df = df[df["conf_pred_label"] > 0.2]

    val_df = (
        val_df.groupby(
            ["indiv_name", "pred_label"],
            group_keys=False,
        )
        .sample(frac=0.3, random_state=SEED)
        .reset_index(drop=True)
    )

    for _, row in val_df.iterrows():

        img_path = Path(row["crop_path"])

        img = Image.open(img_path).convert("RGB")
        img = input_transform(img)

        label = val_data.label_map[row["indiv_name"]]

        pred_label = row["pred_label"]

        out_dir = (
            results_folder
            / "grad_cam"
            / "val"
            / "pred_to_gt"
            / pred_label
            / class_names[label]
        )

        os.makedirs(out_dir, exist_ok=True)

        outfile = out_dir / f"{img_path.parent.name}_{img_path.name}"

        save_grad_cam(
            base_model,
            class_names,
            img.unsqueeze(0),
            label,
            out_name=outfile,
        )


if __name__ == "__main__":
    main()