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
import pandas as pd


# ============================================================
# Plot function
# ============================================================
  
def save_confusion_matrix(cm, class_names,out_dir,police_size=8):
    fig, ax = plt.subplots(figsize=(15, 15))
    cm_sum = cm.sum(axis=1, keepdims=True)
    # print("Number of crops per individuals : ")
    # for i in range(len(class_names)):
    #     print(f"{class_names[i]}:{cm_sum.item(i)}")
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
    
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Normalized confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'confusion_normalized'), dpi=300, bbox_inches="tight")
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
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'confusion'), dpi=300, bbox_inches="tight")
    plt.close()

def save_confusion_matrix_rect(cm, class_names, out_dir, police_size=8):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as patches

    # Remove individuals absent from validation set (empty Actual rows)
    mask = cm.sum(axis=1) != 0

    missing_indiv = np.array(class_names)[~mask]
    if len(missing_indiv) > 0:
        print("Individuals absent from validation:")
        print(missing_indiv.tolist())

    # Keep all predicted classes (columns)
    cm = cm[mask, :]
    
    actual_class_names = np.array(class_names)[mask].tolist()
    predicted_class_names = class_names

    # =========================
    # Normalized confusion matrix
    # =========================
    fig, ax = plt.subplots(figsize=(15, 15))

    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm, dtype=float)

    mask_norm = cm_sum.squeeze() != 0
    cm_norm[mask_norm] = cm[mask_norm] / cm_sum[mask_norm]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=predicted_class_names,
        yticklabels=actual_class_names,
        annot_kws={"size": police_size},
        ax=ax,
        cbar_kws={"pad": 0.15}
    )

    ax.set_xticklabels(
        predicted_class_names,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        actual_class_names,
        rotation=0
    )

    # Top labels
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(np.arange(len(predicted_class_names)) + 0.5)
    ax_top.set_xticklabels(
        predicted_class_names,
        rotation=45,
        ha="left"
    )

    # Right labels
    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(np.arange(len(actual_class_names)) + 0.5)
    ax_right.set_yticklabels(
        actual_class_names,
        rotation=0
    )

    # Highlight diagonal when class exists in both actual and predicted
    for i, name in enumerate(actual_class_names):
        if name in predicted_class_names:
            j = predicted_class_names.index(name)
            rect = patches.Rectangle(
                (j, i), 1, 1,
                fill=False,
                edgecolor="black",
                linewidth=1.5
            )
            ax.add_patch(rect)

    plt.subplots_adjust(
        left=0.25,
        right=0.75,
        bottom=0.20,
        top=0.85
    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Normalized confusion Matrix")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "rect_confusion_normalized"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


    # =========================
    # Raw confusion matrix
    # =========================
    fig, ax = plt.subplots(figsize=(15, 15))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": police_size},
        xticklabels=predicted_class_names,
        yticklabels=actual_class_names,
        ax=ax,
        cbar_kws={"pad": 0.15}
    )

    # Highlight diagonal
    for i, name in enumerate(actual_class_names):
        if name in predicted_class_names:
            j = predicted_class_names.index(name)
            rect = patches.Rectangle(
                (j, i), 1, 1,
                fill=False,
                edgecolor="black",
                linewidth=1.5
            )
            ax.add_patch(rect)

    ax.set_xticklabels(
        predicted_class_names,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        actual_class_names,
        rotation=0
    )

    # Top labels
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(np.arange(len(predicted_class_names)) + 0.5)
    ax_top.set_xticklabels(
        predicted_class_names,
        rotation=45,
        ha="left"
    )

    # Right labels
    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(np.arange(len(actual_class_names)) + 0.5)
    ax_right.set_yticklabels(
        actual_class_names,
        rotation=0
    )

    plt.subplots_adjust(
        left=0.25,
        right=0.75,
        bottom=0.20,
        top=0.85
    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "rect_confusion"),
        dpi=300,
        bbox_inches="tight"
    )
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
np.savez(
    f"{version}_{crop_type}_confusion.npz",
    cm=cm,
    class_names=np.array(class_names),
)

# ============================================================
# Save result
# ============================================================

# save_confusion_matrix_rect(
#     cm,
#     class_names,
#     results_folder
# )


# print(
#     f"Confusion matrix saved in: {results_folder}"
# )


# ============================================================
# Individual contribution to global accuracy
# ============================================================

# global_accuracy = (
#     (predicted == labels).sum().item()
#     /
#     len(labels)
# )

# print(
#     f"\nGlobal validation accuracy: {global_accuracy:.3f}"
# )


# individual_results = []


# for idx, indiv in enumerate(class_names):

#     mask = labels == idx

#     nb_crop = mask.sum().item()

#     if nb_crop == 0:
#         continue


#     nb_correct = (
#         (predicted[mask] == idx)
#         .sum()
#         .item()
#     )


#     indiv_accuracy = nb_correct / nb_crop


#     # Difference with expected correct predictions
#     # if this individual had the global accuracy
#     expected_correct = (
#         nb_crop * global_accuracy
#     )


#     contribution = (
#         nb_correct - expected_correct
#     )


#     individual_results.append(
#         {
#             "indiv_name": indiv,
#             "nb_crop": nb_crop,
#             "nb_correct": nb_correct,
#             "accuracy": indiv_accuracy,
#             "contribution": contribution
#         }
#     )



# # dataframe only for sorting
# individual_results = pd.DataFrame(
#     individual_results
# )



# # ============================================================
# # Individuals increasing accuracy
# # ============================================================

# print(
#     "\n=============================="
# )
# print(
#     "Individuals increasing accuracy"
# )
# print(
#     "=============================="
# )


# positive = individual_results[
#     individual_results["contribution"] > 0
# ]


# positive = positive.sort_values(
#     "contribution",
#     ascending=False
# )


# print(
#     positive.to_string(
#         index=False,
#         formatters={
#             "accuracy": "{:.2f}".format,
#             "contribution": "{:.0f}".format
#         }
#     )
# )



# # ============================================================
# # Individuals decreasing accuracy
# # ============================================================

# print(
#     "\n=============================="
# )
# print(
#     "Individuals decreasing accuracy"
# )
# print(
#     "=============================="
# )


# negative = individual_results[
#     individual_results["contribution"] < 0
# ]


# negative = negative.sort_values(
#     "contribution",
#     ascending=True
# )


# print(
#     negative.to_string(
#         index=False,
#         formatters={
#             "accuracy": "{:.2f}".format,
#             "contribution": "{:.0f}".format
#         }
#     )
# )