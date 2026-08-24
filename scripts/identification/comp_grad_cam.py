import os
import argparse
from pathlib import Path

import yaml
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
# LOAD MODEL + DATA
# ============================================================

def load_model_and_data(config_name):
    """
    Load a classifier and all information needed to generate
    Grad-CAMs.
    """

    config_path = CONFIG_PATH / config_name

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    crop_type = cfg["crop_type"]
    crop_version = cfg["crop_version"]
    split_file = cfg["split_file"]
    mode = cfg["mode"]
    version = cfg["version"]

    crop_path = CROP_PATH / crop_type / crop_version

    split_path = CROP_PATH / "split" / split_file

    features_folder = (
        FEATURES_PATH
        / version
        / crop_type
    )

    result_folder = (
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
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    # --------------------------------------------------------
    # Base ResNet
    # --------------------------------------------------------

    base_model = models.resnet50(weights="DEFAULT")

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
        split_path,
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

    base_model.load_state_dict(state_dict)

    base_model.to(DEVICE)
    base_model.eval()

    # --------------------------------------------------------
    # Load samples
    # --------------------------------------------------------

    df = pd.read_excel(
        split_path,
        sheet_name="samples",
        engine="odf",
    )

    if mode:
        df = df[
            df["mode"] == mode
        ].reset_index(drop=True)

    df["crop_path"] = df["crop_id"].apply(
        lambda x: crop_path / str(x)
    )

    return {
        "model": base_model,
        "class_names": class_names,
        "train_data": train_data,
        "val_data": val_data,
        "df": df,
        "crop_path": crop_path,
        "result_folder": result_folder,
        "config": cfg,
    }


# ============================================================
# GRAD-CAM
# ============================================================

def compute_grad_cam(model, image, target_class):
    """
    Compute Grad-CAM for one image and one target class.

    Parameters
    ----------
    model : torch.nn.Module
    image : Tensor
        Shape [1, 3, H, W]
    target_class : int

    Returns
    -------
    grad_cam : numpy array
        Shape [H, W]
    """

    activations = None
    gradients = None

    # --------------------------------------------------------
    # Find last convolutional layer
    # --------------------------------------------------------

    last_conv = None

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise RuntimeError(
            "No Conv2d layer found in model."
        )

    # --------------------------------------------------------
    # Hooks
    # --------------------------------------------------------

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    handle_forward = last_conv.register_forward_hook(
        forward_hook
    )

    handle_backward = last_conv.register_full_backward_hook(
        backward_hook
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    model.zero_grad(set_to_none=True)

    output = model(image)

    score = output[0, target_class]

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    score.backward()

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    # Global average pooling of gradients
    alpha = gradients.mean(
        dim=(2, 3),
        keepdim=True,
    )

    # Weighted activations
    grad_cam = F.relu(
        (alpha * activations).sum(dim=1)
    )

    grad_cam = grad_cam[0]

    # Normalize
    grad_cam -= grad_cam.min()

    max_value = grad_cam.max()

    if max_value > 0:
        grad_cam /= max_value

    grad_cam = grad_cam.detach().cpu().numpy()

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    handle_forward.remove()
    handle_backward.remove()

    return grad_cam


# ============================================================
# PREDICTION + GRAD-CAM
# ============================================================

def get_prediction_and_gradcam(
    model,
    image,
):
    """
    Predict an image and compute Grad-CAM for the
    predicted class.
    """

    device = next(model.parameters()).device

    image = image.to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():
        output = model(image)

    pred_class = output.argmax(dim=1).item()

    probabilities = torch.softmax(
        output,
        dim=1,
    )

    confidence = probabilities[
        0,
        pred_class,
    ].item()

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    grad_cam = compute_grad_cam(
        model,
        image,
        pred_class,
    )

    return (
        pred_class,
        confidence,
        grad_cam,
    )


# ============================================================
# ORIGINAL IMAGE
# ============================================================

def load_original_image(img_path):
    """
    Load image for display.

    The image is not normalized/resized here.
    """

    return Image.open(img_path).convert("RGB")


# ============================================================
# COMPARISON FIGURE
# ============================================================

def save_comparison(
    original_image,
    gradcam1,
    gradcam2,
    pred1,
    conf1,
    pred2,
    conf2,
    class_names1,
    class_names2,
    output_path,
):
    """
    Save a 3-column comparison:

        Original | Model 1 Grad-CAM | Model 2 Grad-CAM
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(original_image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    # --------------------------------------------------------
    # Model 1
    # --------------------------------------------------------

    axes[1].imshow(original_image)
    axes[1].imshow(
        gradcam1,
        cmap="viridis",
        alpha=0.5,
        interpolation="bilinear",
        extent=(
            0,
            original_image.width,
            original_image.height,
            0,
        ),
    )

    axes[1].set_title(
        f"Model 1\n"
        f"{class_names1[pred1]} : {conf1:.2f}"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Model 2
    # --------------------------------------------------------

    axes[2].imshow(original_image)
    axes[2].imshow(
        gradcam2,
        cmap="viridis",
        alpha=0.5,
        interpolation="bilinear",
        extent=(
            0,
            original_image.width,
            original_image.height,
            0,
        ),
    )

    axes[2].set_title(
        f"Model 2\n"
        f"{class_names2[pred2]} : {conf2:.2f}"
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
# BUILD MATCHED DATA
# ============================================================

def build_comparison_dataframe(
    data1,
    data2,
    split,
):
    """
    Match samples from model 1 and model 2 using video and frame.

    This allows the two models to be compared on exactly
    the same samples.
    """

    df1 = data1["df"]

    df2 = data2["df"]

    df1 = df1[
        df1["split"] == split
    ].copy()

    df2 = df2[
        df2["split"] == split
    ].copy()

    # --------------------------------------------------------
    # Rename columns to distinguish the two data sources
    # --------------------------------------------------------

    df1 = df1.rename(
        columns={
            "crop_path": "crop_path_1"
        }
    )
    df1["frame"] = df1["crop_id"].str.extract(
    r"frame_(\d+)")[0].astype(int)

    df2 = df2.rename(
        columns={
            "crop_path": "crop_path_2",
        }
    )
    df2["frame"] = df2["crop_id"].str.extract(
    r"frame_(\d+)")[0].astype(int)

    # --------------------------------------------------------
    # Match using crop_id
    # --------------------------------------------------------

    common = pd.merge(
        df1[
            [
                "crop_path_1",
                "indiv_name",
                "video",
                "frame"
            ]
        ],
        df2[
            [
                "crop_path_2",
                "indiv_name",
                "video",
                "frame"
            ]
        ],
        on=["video","frame","indiv_name"],
        how="inner",
    )

    return common


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split,
    data1,
    data2,
    output_root,
):
    """
    Generate comparison Grad-CAMs for one split.
    """

    print(
        f"\nProcessing {split}..."
    )

    comparison_df = build_comparison_dataframe(
        data1,
        data2,
        split,
    )

    print(
        f"Found {len(comparison_df)} common samples."
    )

    model1 = data1["model"]
    model2 = data2["model"]

    class_names1 = data1["class_names"]
    class_names2 = data2["class_names"]

    for idx, row in comparison_df.iterrows():

        crop_path1 = Path(
            row["crop_path_1"]
        )

        crop_path2 = Path(
            row["crop_path_2"]
        )

        # ----------------------------------------------------
        # Check both images
        # ----------------------------------------------------

        if not crop_path1.exists():
            print(
                f"Missing model 1 image: {crop_path1}"
            )
            continue

        if not crop_path2.exists():
            print(
                f"Missing model 2 image: {crop_path2}"
            )
            continue

        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        original_image = load_original_image(
            crop_path1
        )

        image1 = input_transform(
            original_image
        ).unsqueeze(0)

        # If the two crop sources are not exactly the same
        # image, model 2 receives its own crop.
        image2_original = load_original_image(
            crop_path2
        )

        image2 = input_transform(
            image2_original
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Model 1
        # ----------------------------------------------------

        (
            pred1,
            conf1,
            gradcam1,
        ) = get_prediction_and_gradcam(
            model1,
            image1,
        )

        # ----------------------------------------------------
        # Model 2
        # ----------------------------------------------------

        (
            pred2,
            conf2,
            gradcam2,
        ) = get_prediction_and_gradcam(
            model2,
            image2,
        )

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        indiv_name = row["indiv_name"]
        
        output_dir = (
            output_root
            / split
            / str(indiv_name)
        )
            

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / f"{crop_path1.name}"
        )

        # ----------------------------------------------------
        # Save comparison
        # ----------------------------------------------------

        save_comparison(
            original_image=original_image,
            gradcam1=gradcam1,
            gradcam2=gradcam2,
            pred1=pred1,
            conf1=conf1,
            pred2=pred2,
            conf2=conf2,
            class_names1=class_names1,
            class_names2=class_names2,
            output_path=output_path,
        )

        if (idx + 1) % 50 == 0:
            print(
                f"  {idx + 1}/{len(comparison_df)}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare Grad-CAM of two ResNet50 classifiers."
        )
    )

    parser.add_argument(
        "--config1",
        required=True,
        help="Config of model 1",
    )

    parser.add_argument(
        "--config2",
        required=True,
        help="Config of model 2",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load both models
    # --------------------------------------------------------

    print(
        f"Loading model 1: {args.config1}"
    )

    data1 = load_model_and_data(
        args.config1
    )

    print(
        f"Loading model 2: {args.config2}"
    )

    data2 = load_model_and_data(
        args.config2
    )

    # --------------------------------------------------------
    # Output
    #
    # Use model 1's result folder as the common result folder.
    # --------------------------------------------------------

    result_folder = data1["result_folder"]

    output_root = (
        result_folder
        / "comp_grad_cam"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nGrad-CAM comparison output:\n"
        f"{output_root}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    process_split(
        split="train",
        data1=data1,
        data2=data2,
        output_root=output_root,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    process_split(
        split="val",
        data1=data1,
        data2=data2,
        output_root=output_root,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()