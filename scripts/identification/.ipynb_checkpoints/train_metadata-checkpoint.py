import os
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import yaml
from torchvision import models, transforms

from elephantreid.identification import *

# ============================================================
# PARAMETERS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers")
FEATURES_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features")
CROP_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers")
MODEL_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/models/identification")
MODEL_CONFIG = Path("/pbs/home/a/ameurisse/ElephantReId/config/train_classifiers")

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

input_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normalize,
])

# ============================================================
# CONFIG
# ============================================================

import argparse

parser = argparse.ArgumentParser(
    description="Generate train predictions from a trained classifier"
)
parser.add_argument(
    "--config",
    "-c",
    required=True,
    help="classifier config file",
)
arg = parser.parse_args()

with open(MODEL_CONFIG / arg.config, "r", encoding="utf-8") as f:
    model_cfg = yaml.safe_load(f)

selection_type = model_cfg["selection_type"]
crop_type = model_cfg["crop_type"]
date = model_cfg["date"]
split_version = model_cfg["split_version"]
crop_version = model_cfg["crop_version"]
mode = model_cfg["mode"]
version = model_cfg["version"]

crop_path = CROP_PATH / crop_type / crop_version
split_file = CROP_PATH / model_cfg["split_file"]

results_folder = RESULTS_PATH / version / crop_type
model_file = MODEL_PATH / version / crop_type / "model.pth"
features_folder = FEATURES_PATH / version / crop_type

# ============================================================
# LOAD DATA
# ============================================================

base_model = models.resnet50(weights="DEFAULT")

train_features_dataset, _, train_data, _, _ = load_data(
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
num_classes = train_data.classes_nb

base_model.fc = ClassificationHead(
    dim_in=2048,
    dim_out=num_classes,
)

base_model.load_state_dict(
    torch.load(model_file, map_location=DEVICE)
)
base_model.to(DEVICE)
base_model.eval()

# ============================================================
# INFERENCE ON TRAIN SET
# ============================================================

train_loader = torch.utils.data.DataLoader(
    train_features_dataset,
    batch_size=256,
    shuffle=False,
)

all_outputs = []
all_labels = []
all_idx = []

with torch.no_grad():
    for feat, label, idx in train_loader:
        feat = feat.to(DEVICE)

        output = base_model.fc(feat)

        all_outputs.append(output.cpu())
        all_labels.append(label.cpu())
        all_idx.append(idx.cpu())

all_outputs = torch.cat(all_outputs)
all_labels = torch.cat(all_labels)
all_idx = torch.cat(all_idx)

all_probs = torch.softmax(all_outputs, dim=1)

# ============================================================
# LOAD TRAIN METADATA
# ============================================================

out_df = pd.read_excel(
    split_file,
    sheet_name="samples",
    engine="odf",
)

out_df = (
    out_df[
        (out_df["mode"] == mode)
        & (out_df["split"] == "train")
    ]
    .reset_index(drop=True)
)

assert len(out_df) == len(all_outputs)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

out_df["logits"] = [
    [round(v, 2) for v in x.tolist()]
    for x in all_outputs.numpy()
]

out_df["pred"] = [
    [round(v, 2) for v in x.tolist()]
    for x in all_probs.numpy()
]

out_df["conf_pred_label"] = (
    all_probs.max(dim=1)
    .values
    .numpy()
    .round(2)
)

top2 = torch.topk(all_probs, k=2, dim=1).values

out_df["margin"] = (
    (top2[:, 0] - top2[:, 1])
    .numpy()
    .round(2)
)

entropy = -(all_probs * torch.log(all_probs + 1e-8)).sum(dim=1)

out_df["entropy"] = (
    entropy.numpy()
    .round(2)
)

pred_idx = all_probs.argmax(dim=1).numpy()

out_df["pred_label"] = [
    class_names[i]
    for i in pred_idx
]

out_df.to_csv(
    results_folder / "train_pred_metadata.csv",
    index=False,
    float_format="%.2f",
)

print(f"Saved {len(out_df)} predictions to:")
print(results_folder / "train_pred_metadata.csv")