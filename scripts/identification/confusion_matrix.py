"""
Plot confusion matrix from saved validation predictions.

This script:
- Loads configuration from a YAML file.
- Loads saved model outputs and labels from a .pt file.
- Computes predicted classes using argmax.
- Computes and saves the confusion matrix.
"""

import os
import yaml
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
import matplotlib.patches as patches


# ============================================================
# Plot function
# ============================================================
  
def save_confusion_matrix(cm, class_names,out_dir,police_size=8):
    fig, ax = plt.subplots(figsize=(15, 15))
    cm_sum = cm.sum(axis=1, keepdims=True)
    print("Number of crops per individuals : ")
    for i in range(len(class_names)):
        print(f"{class_names[i]}:{cm_sum.item(i)}")
    cm_norm = np.zeros_like(cm, dtype=float)
    mask = cm_sum.squeeze() != 0
    cm_norm[mask] = cm[mask] / cm_sum[mask]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"pad": 0.15}  # espace pour les labels de droite
    )
    
    # Labels du bas
    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right"
    )
    
    # Labels de gauche
    ax.set_yticklabels(
        class_names,
        rotation=0
    )
    
    # Axe secondaire en haut
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(np.arange(len(class_names)) + 0.5)
    ax_top.set_xticklabels(
        class_names,
        rotation=45,
        ha="left"
    )
    
    # Axe secondaire à droite
    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(np.arange(len(class_names)) + 0.5)
    ax_right.set_yticklabels(
        class_names,
        rotation=0
    )
    
    # Ajustement des marges
    plt.subplots_adjust(
        left=0.25,
        right=0.30,
        bottom=0.20,
        top=0.85
    )
    # sns.heatmap(
    #     cm_norm,
    #     annot=True,
    #     fmt=".2f",
    #     cmap="Blues",
    #     vmin=0,
    #     vmax=1,
    #     annot_kws={"size": police_size}, #taille police
    #     xticklabels=class_names,
    #     yticklabels=class_names
    # )
    
    # Highlight diagonal cells
    for i in range(len(class_names)):
        rect = patches.Rectangle(
            (i, i), 1, 1,
            fill=False,
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(rect)
        
    # ax.tick_params(
    #     axis="x",
    #     labeltop=True,
    #     labelbottom=True,
    #     top=True,
    #     bottom=True
    # )
    
    # ax.tick_params(
    #     axis="y",
    #     labelleft=True,
    #     labelright=True,
    #     left=True,
    #     right=True
    # )
    
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Normalized confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'confusion_matrix_normalized'), dpi=300, bbox_inches="tight")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": police_size},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"pad": 0.15}  # espace pour les labels de droite
    )
    
    # Highlight diagonal cells
    for i in range(len(class_names)):
        rect = patches.Rectangle(
            (i, i), 1, 1,
            fill=False,
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(rect)
    # Labels du bas
    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right"
    )
    
    # Labels de gauche
    ax.set_yticklabels(
        class_names,
        rotation=0
    )
    
    # Axe secondaire en haut
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(np.arange(len(class_names)) + 0.5)
    ax_top.set_xticklabels(
        class_names,
        rotation=45,
        ha="left"
    )
    
    # Axe secondaire à droite
    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(np.arange(len(class_names)) + 0.5)
    ax_right.set_yticklabels(
        class_names,
        rotation=0
    )
    
    # Ajustement des marges
    plt.subplots_adjust(
        left=0.25,
        right=0.30,
        bottom=0.20,
        top=0.85)
    # ax.tick_params(
    #     axis="x",
    #     labeltop=True,
    #     labelbottom=True,
    #     top=True,
    #     bottom=True
    # )
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
    
    # ax.tick_params(
    #     axis="y",
    #     labelleft=True,
    #     labelright=True,
    #     left=True,
    #     right=True
    # )
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'confusion_matrix'), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Plot confusion matrix from validation predictions"
)

parser.add_argument(
    "--config",
    "-c",
    required=True,
    help="Classifier configuration file"
)

args = parser.parse_args()


# ============================================================
# Load configuration
# ============================================================

CONFIG_PATH = Path("/pbs/home/a/ameurisse/ElephantReId/config/train_classifiers")

with open(CONFIG_PATH / args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# ============================================================
# Paths from config
# ============================================================

model_name = "ResNet50_v4"

selection_type = cfg["selection_type"]
crop_type = cfg["crop_type"]
date = cfg["date"]
split_version = cfg['split_version']


RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers")


results_folder = (
    RESULTS_PATH
    / model_name
    / selection_type
    / date
    / crop_type
    / split_version
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


# ============================================================
# Compute predictions
# ============================================================

predicted = torch.argmax(
    outputs,
    dim=1
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true=labels.numpy(),
    y_pred=predicted.numpy(),
    labels=np.arange(len(class_names))
)
cm_sum = cm.sum(axis=1, keepdims=True)
print("Number of crops per individuals : ")
for i in range(len(class_names)):
    print(f"{class_names[i]}:{cm_sum.item(i)}")


# ============================================================
# Save result
# ============================================================

# save_confusion_matrix(
#     cm,
#     class_names,
#     results_folder
# )


print(
    f"Confusion matrix saved in: {results_folder}"
)