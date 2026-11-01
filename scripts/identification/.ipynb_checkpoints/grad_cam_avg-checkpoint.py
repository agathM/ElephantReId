import os
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from elephantreid.identification import *


# ============================================================
# PARAMETERS
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULTS_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers"
)

FEATURES_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features"
)

CROP_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers"
)

MODEL_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/models/identification"
)

CONFIG_PATH = Path(
    "/pbs/home/a/ameurisse/ElephantReId/config/train_classifiers"
)


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
# GET LAST CONVOLUTIONAL LAYER
# ============================================================

def get_last_conv(model: nn.Module) -> nn.Conv2d:

    last_conv = None

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError("No Conv2d layer found in the model.")

    return last_conv


# ============================================================
# COMPUTE GRAD-CAM
# ============================================================

def compute_grad_cam(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: int,
    last_conv: nn.Conv2d,
) -> torch.Tensor:
    """
    Compute Grad-CAM for one image and one class.

    Returns
    -------
    grad_cam : torch.Tensor
        Normalized Grad-CAM at the native resolution of the
        last convolutional layer, typically 7x7 for ResNet50.
    """

    device = next(model.parameters()).device

    image = image.to(device)

    activations = None

    def forward_hook(module, inputs, output):
        nonlocal activations
        activations = output

    handle = last_conv.register_forward_hook(forward_hook)

    try:

        # ------------------------------------------------------
        # Forward
        # ------------------------------------------------------

        output = model(image)

        if activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        # ------------------------------------------------------
        # Gradient of the target class
        # ------------------------------------------------------

        model.zero_grad(set_to_none=True)

        gradient = torch.autograd.grad(
            outputs=output[0, class_idx],
            inputs=activations,
            retain_graph=False,
            create_graph=False,
        )[0]

        # ------------------------------------------------------
        # Grad-CAM weights
        # ------------------------------------------------------

        alpha = gradient.mean(
            dim=(2, 3),
            keepdim=True,
        )

        grad_cam = (
            alpha * activations
        ).sum(dim=1)

        grad_cam = F.relu(grad_cam)

        grad_cam = grad_cam[0]

        # ------------------------------------------------------
        # Normalize
        # ------------------------------------------------------

        min_val = grad_cam.min()
        max_val = grad_cam.max()

        if max_val > min_val:
            grad_cam = (
                grad_cam - min_val
            ) / (max_val - min_val)
        else:
            grad_cam.zero_()

        return grad_cam.detach().cpu()

    finally:
        handle.remove()


# ============================================================
# LOAD AND NORMALIZE IMAGE FOR DISPLAY
# ============================================================

def prepare_display_image(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a normalized tensor to a displayable RGB image.
    """

    plot_image = (
        image[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .clone()
    )

    plot_image -= plot_image.min()

    if plot_image.max() > 0:
        plot_image /= plot_image.max()

    return plot_image


# ============================================================
# SAVE AVERAGED GRAD-CAM
# ============================================================

def save_average_grad_cam(
    average_grad_cam: torch.Tensor,
    best_image: torch.Tensor,
    title: str,
    output_path: Path,
) -> None:
    """
    Save the average Grad-CAM interpolated over the best image.

    The Grad-CAM is kept at its native resolution (7x7) and
    matplotlib performs bilinear interpolation when displaying it
    over the 224x224 image.
    """

    plot_image = prepare_display_image(best_image)

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(5, 5),
    )

    ax.imshow(
        plot_image,
        extent=(0, 224, 0, 224),
    )

    ax.imshow(
        average_grad_cam,
        cmap="viridis",
        alpha=0.5,
        interpolation="bilinear",
        extent=(0, 224, 0, 224),
    )

    ax.set_title(title)

    ax.axis("off")

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate averaged Grad-CAM figures."
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Training config file.",
    )

    args = parser.parse_args()

    # ========================================================
    # CONFIG
    # ========================================================

    with open(
        CONFIG_PATH / args.config,
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    selection_type = cfg["selection_type"]
    crop_type = cfg["crop_type"]
    date = cfg["date"]
    crop_version = cfg["crop_version"]
    split_version = cfg["split_version"]
    mode = cfg["mode"]
    version = cfg["version"]

    crop_path = (
        CROP_PATH
        / crop_type
        / crop_version
    )

    split_file = (
        CROP_PATH
        / "split"
        / cfg["split_file"]
    )

    features_folder = (
        FEATURES_PATH
        / version
        / crop_type
    )

    results_folder = (
        RESULTS_PATH
        / version
        / crop_type
    )

    model_path = (
        MODEL_PATH
        / version
        / crop_type
        / "model.pth"
    )

    if not model_path.exists():
        print(
            f"Model not found: {model_path}"
        )
        return

    # ========================================================
    # LOAD MODEL
    # ========================================================

    base_model = models.resnet50(
        weights="DEFAULT"
    )

    (
        train_features_dataset,
        val_features_dataset,
        train_data,
        val_data,
        _,
    ) = load_data(
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

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
    )

    base_model.load_state_dict(
        state_dict
    )

    base_model.to(DEVICE)
    base_model.eval()

    # --------------------------------------------------------
    # Last convolutional layer
    # --------------------------------------------------------

    last_conv = get_last_conv(base_model)

    print(
        f"Last convolutional layer: {last_conv}"
    )

    # ========================================================
    # LOAD SPLIT FILE
    # ========================================================

    df = pd.read_excel(
        split_file,
        sheet_name="samples",
        engine="odf",
    )

    if mode:
        df = (
            df[df["mode"] == mode]
            .reset_index(drop=True)
        )

    df["crop_path"] = df["crop_id"].apply(
        lambda x: crop_path / str(x)
    )

    # ========================================================
    # ========================================================
    # TRAIN
    # ========================================================
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAIN AVERAGED GRAD-CAM")
    print("=" * 70)

    train_df = df[
        df["split"] == "train"
    ].copy()

    # --------------------------------------------------------
    # Containers
    #
    # Each individual contains:
    #
    #   {
    #       "grad_cams": [...],
    #       "best_margin": ...,
    #       "best_conf": ...,
    #       "best_image": ...
    #   }
    # --------------------------------------------------------

    train_grad_cams = {}

    # --------------------------------------------------------
    # Process every training image
    # --------------------------------------------------------

    for _, row in train_df.iterrows():

        img_path = Path(
            row["crop_path"]
        )

        indiv_name = row["indiv_name"]

        # ----------------------------------------------------
        # GT label
        # ----------------------------------------------------

        if indiv_name not in train_data.label_map:
            continue

        label = train_data.label_map[
            indiv_name
        ]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        img = Image.open(
            img_path
        ).convert("RGB")

        img = input_transform(img)

        image_batch = img.unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():
            output = base_model(
                image_batch
            )

            probs = torch.softmax(
                output,
                dim=1,
            )

            values, indexes = torch.topk(
                probs,
                k=2,
                dim=1,
            )

        pred_label = int(
            indexes[0, 0]
        )

        confidence = float(
            values[0, 0]
        )

        margin = float(
            values[0, 0]
            - values[0, 1]
        )

        # ----------------------------------------------------
        # Keep only correct predictions
        # ----------------------------------------------------

        if pred_label != label:
            continue

        # ----------------------------------------------------
        # Grad-CAM for predicted class
        # ----------------------------------------------------

        grad_cam = compute_grad_cam(
            base_model,
            img.unsqueeze(0),
            pred_label,
            last_conv,
        )

        # ----------------------------------------------------
        # Initialize individual
        # ----------------------------------------------------

        if indiv_name not in train_grad_cams:

            train_grad_cams[indiv_name] = {
                "grad_cams": [],
                "best_margin": -float("inf"),
                "best_conf": -float("inf"),
                "best_image": None,
            }

        data = train_grad_cams[
            indiv_name
        ]

        data["grad_cams"].append(
            grad_cam
        )

        # ----------------------------------------------------
        # Best image:
        #
        # 1. highest margin
        # 2. highest confidence if same margin
        # ----------------------------------------------------

        is_better = (
            margin > data["best_margin"]
            or (
                margin == data["best_margin"]
                and confidence > data["best_conf"]
            )
        )

        if is_better:

            data["best_margin"] = margin
            data["best_conf"] = confidence
            data["best_image"] = (
                img.unsqueeze(0).clone()
            )

    # --------------------------------------------------------
    # Average and save
    # --------------------------------------------------------

    for indiv_name, data in train_grad_cams.items():

        if not data["grad_cams"]:
            continue

        # ----------------------------------------------------
        # Average Grad-CAM
        # ----------------------------------------------------

        average_grad_cam = torch.stack(
            data["grad_cams"],
            dim=0,
        ).mean(dim=0)

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = train_data.label_map[
            indiv_name
        ]

        class_name = class_names[
            label
        ]

        # ----------------------------------------------------
        # Output
        #
        # grad_cam_avg/
        #     train/
        #         Baymax/
        #             Baymax.png
        # ----------------------------------------------------

        output_dir = (
            results_folder
            / "grad_cam_avg"
            / "train"
        )

        output_path = (
            output_dir
            / f"{class_name}.png"
        )

        title = (
            f"{class_name}\n"
            f"n = {len(data['grad_cams'])} | "
            f"conf = {data['best_conf']:.2f} | "
            f"margin = {data['best_margin']:.2f}"
        )

        save_average_grad_cam(
            average_grad_cam,
            data["best_image"],
            title,
            output_path,
        )

        print(
            f"[TRAIN] {class_name}: "
            f"{len(data['grad_cams'])} images -> "
            f"{output_path}"
        )

    # ========================================================
    # ========================================================
    # VALIDATION
    # ========================================================
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION AVERAGED GRAD-CAM")
    print("=" * 70)

    val_pred_file = (
        results_folder
        / "val_pred_metadata.csv"
    )

    val_df = pd.read_csv(
        val_pred_file
    )

    val_df["crop_path"] = val_df[
        "crop_id"
    ].apply(
        lambda x: crop_path / str(x)
    )

    val_df["conf_pred_label"] = (
        val_df["conf_pred_label"]
        .astype(float)
    )

    # --------------------------------------------------------
    # Containers
    #
    # Key:
    #
    #     (GT individual, predicted individual)
    #
    # Example:
    #
    #     ("Baymax", "Oolong")
    #
    # --------------------------------------------------------

    val_grad_cams = {}

    # --------------------------------------------------------
    # Process every validation image
    # --------------------------------------------------------

    for _, row in val_df.iterrows():

        img_path = Path(
            row["crop_path"]
        )

        gt_indiv = row["indiv_name"]
        pred_indiv = row["pred_label"]

        # ----------------------------------------------------
        # Make sure labels exist
        # ----------------------------------------------------

        if gt_indiv not in val_data.label_map:
            continue

        if pred_indiv not in class_names:
            continue

        gt_label = val_data.label_map[
            gt_indiv
        ]

        pred_label = class_names.index(
            pred_indiv
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            row["conf_pred_label"]
        )

        # ----------------------------------------------------
        # Margin
        #
        # If it is already present in the CSV, use it.
        # Otherwise calculate it from the model.
        # ----------------------------------------------------

        if "margin" in row.index:

            margin = float(
                row["margin"]
            )

        elif "margin_pred_label" in row.index:

            margin = float(
                row["margin_pred_label"]
            )

        else:

            # ------------------------------------------------
            # Need a forward pass to calculate margin
            # ------------------------------------------------

            img = Image.open(
                img_path
            ).convert("RGB")

            img = input_transform(img)

            with torch.no_grad():

                output = base_model(
                    img.unsqueeze(0).to(DEVICE)
                )

                probs = torch.softmax(
                    output,
                    dim=1,
                )

                values, indexes = torch.topk(
                    probs,
                    k=2,
                    dim=1,
                )

                predicted_class = int(
                    indexes[0, 0]
                )

                # ------------------------------------------------
                # The prediction in the CSV should match
                # the model prediction.
                # ------------------------------------------------

                if predicted_class != pred_label:
                    continue

                margin = float(
                    values[0, 0]
                    - values[0, 1]
                )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        img = Image.open(
            img_path
        ).convert("RGB")

        img = input_transform(img)

        # ----------------------------------------------------
        # Grad-CAM for predicted class
        # ----------------------------------------------------

        grad_cam = compute_grad_cam(
            base_model,
            img.unsqueeze(0),
            pred_label,
            last_conv,
        )

        # ----------------------------------------------------
        # Group by GT / prediction
        # ----------------------------------------------------

        key = (
            gt_indiv,
            pred_indiv,
        )

        if key not in val_grad_cams:

            val_grad_cams[key] = {
                "grad_cams": [],
                "best_margin": -float("inf"),
                "best_conf": -float("inf"),
                "best_image": None,
            }

        data = val_grad_cams[key]

        data["grad_cams"].append(
            grad_cam
        )

        # ----------------------------------------------------
        # Best image:
        #
        # 1. highest margin
        # 2. highest confidence if same margin
        # ----------------------------------------------------

        is_better = (
            margin > data["best_margin"]
            or (
                margin == data["best_margin"]
                and confidence > data["best_conf"]
            )
        )

        if is_better:

            data["best_margin"] = margin
            data["best_conf"] = confidence
            data["best_image"] = (
                img.unsqueeze(0).clone()
            )

    # --------------------------------------------------------
    # Average and save validation Grad-CAMs
    # --------------------------------------------------------

    for (
        gt_indiv,
        pred_indiv,
    ), data in val_grad_cams.items():

        if not data["grad_cams"]:
            continue

        # ----------------------------------------------------
        # Average Grad-CAM
        # ----------------------------------------------------

        average_grad_cam = torch.stack(
            data["grad_cams"],
            dim=0,
        ).mean(dim=0)

        # ----------------------------------------------------
        # Output
        #
        # grad_cam_avg/
        #     val/
        #         pred_to_gt/
        #             Oolong/
        #                 Baymax/
        #                     Oolong_Baymax.png
        # ----------------------------------------------------

        output_dir = (
            results_folder
            / "grad_cam_avg"
            / "val"
            / "pred_to_gt"
        )

        output_path = (
            output_dir
            / f"{pred_indiv}_{gt_indiv}.png"
        )

        title = (
            f"{pred_indiv} → {gt_indiv}\n"
            f"n = {len(data['grad_cams'])} | "
            f"conf = {data['best_conf']:.2f} | "
            f"margin = {data['best_margin']:.2f}"
        )

        save_average_grad_cam(
            average_grad_cam,
            data["best_image"],
            title,
            output_path,
        )

        print(
            f"[VAL] "
            f"{pred_indiv} → {gt_indiv}: "
            f"{len(data['grad_cams'])} images -> "
            f"{output_path}"
        )

    print("\nDone.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()