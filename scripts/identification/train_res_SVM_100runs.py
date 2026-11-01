# This script extracts ResNet152 features, reduces their dimensionality with PCA,
# and trains a Linear SVM independently for each train/validation split.
# PCA is fitted only on the training set of each run to avoid data leakage.

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC


# ============================================================
# PARAMETERS
# ============================================================

SEED = 42

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

OUT_DIR_NAME = "100_runs_splits_resnet150_svm_v2"

version = "v9"
crop_type = "Ear"
mode = "Day"

RESULTS_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers/"
)

FEATURES_PATH = Path(
        "/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features/resnet152/v9/Ear/brightness_normalized"
)

MODEL_PATH = (
    Path(
        "/sps/mnhn/ameurisse/ElephantReId/models/identification"
    )
    / OUT_DIR_NAME
)

os.makedirs(MODEL_PATH, exist_ok=True)


# ------------------------------------------------------------
# SVM / PCA parameters
# ------------------------------------------------------------

# SVM regularization parameter.
# Larger C = stronger penalty for classification errors.
SVM_C = 1.0

# Number of PCA components.
# Example: 256 means 2048 -> 256 dimensions.
#
# Set to None to disable dimensionality reduction.
DIM_REDUC = 256

# Maximum number of iterations for the Linear SVM.
SVM_MAX_ITER = 10000


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

SPLIT_FILE = Path(
    "/sps/mnhn/ameurisse/ElephantReId/"
    "data/processed/crops/classifiers/split/"
    "all_0.8_h_1_v_1.6_nc_2.7.csv"
)

RUNS = [97, 57, 98, 94, 6]

CROPS_FILE = Path(
    "/sps/mnhn/ameurisse/ElephantReId/"
    "data/processed/crops/classifiers/split/crops_0.8_h_1_v_1.6_nc_2.7.csv"
)

SUMMARY_FILE = Path(
    "/sps/mnhn/ameurisse/ElephantReId/"
    "data/processed/crops/classifiers/raw/"
    "v2_summary_videos.ods"
)

crop_path = (
    Path(
        "/sps/mnhn/ameurisse/ElephantReId/"
        "data/processed/crops/classifiers/"
    )
    / crop_type
    / "v5"
)
entry_col = "entry_day"

batch_size = 64


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

LUMINANCE = np.array([0.299, 0.587, 0.114])


def transform_brightness_normalized(
    image,
    target_mean=0.5,
):
    """
    Normalize the mean brightness of an image.
    """

    array = np.asarray(image).astype(np.float32) / 255.0

    luminance = array @ LUMINANCE
    current_mean = luminance.mean()

    if current_mean == 0:
        return image

    factor = target_mean / current_mean

    normalized = np.clip(array * factor, 0, 1)

    return Image.fromarray(
        (normalized * 255).astype(np.uint8)
    )


TRANSFORMATIONS = [
    (
        "brightness_normalized",
        transform_brightness_normalized,
    ),
]


# ============================================================
# INPUT PIPELINE
# ============================================================

def build_input_transform(image_transform):
    """
    Build the complete ResNet input pipeline.
    """

    transforms_list = []

    transforms_list.append(image_transform)

    transforms_list.extend([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(transforms_list)


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int) -> None:

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASETS
# ============================================================

class GlobalCropDataset(Dataset):
    """
    Dataset used to compute ResNet features.
    """

    def __init__(
        self,
        crop_path: Path,
        crop_ids: list,
        transform,
    ):
        self.crop_path = crop_path
        self.crop_ids = crop_ids
        self.transform = transform

    def __len__(self):
        return len(self.crop_ids)

    def __getitem__(self, idx):

        image = Image.open(
            self.crop_path / self.crop_ids[idx]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0, idx


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def _compute_global_features(
    model,
    crop_path: Path,
    crop_ids: list,
    device: torch.device,
    input_transform,
    batch_size: int = 64,
) -> tuple:

    """
    Compute ResNet152 features before the final fully connected layer.

    Output dimension:
        2048 features per image.
    """

    dataset = GlobalCropDataset(
        crop_path,
        crop_ids,
        input_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.to(device)
    model.eval()

    batch_features = []
    batch_idxs = []

    def hook(module, input, output):

        # Input to the final FC layer has shape [B, 2048].
        batch_features.append(
            input[0].detach().cpu()
        )

    handle = model.fc.register_forward_hook(hook)

    with torch.no_grad():

        for images, _, idx in loader:

            images = images.to(device)

            _ = model(images)

            batch_idxs.append(idx)

    handle.remove()

    features = torch.cat(
        batch_features,
        dim=0,
    )

    idxs = torch.cat(
        batch_idxs,
        dim=0,
    )

    # Restore the original crop order.
    order = idxs.argsort()

    features = features[order]

    ordered_crop_ids = [
        crop_ids[i]
        for i in order.tolist()
    ]

    return features, ordered_crop_ids


def compute_or_load_global_features(
    model,
    crop_path: Path,
    crop_ids: list,
    cache_dir: Path,
    device: torch.device,
    input_transform,
    batch_size: int = 64,
) -> tuple:

    """
    Compute or load cached ResNet152 features.

    Features are computed once for each image and transformation.
    """

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_cache = cache_dir / "features.pt"
    crop_ids_cache = cache_dir / "crop_ids.pt"

    if (
        features_cache.exists()
        and crop_ids_cache.exists()
    ):

        print(
            f"Loading cached global features "
            f"({features_cache})..."
        )

        features = torch.load(
            features_cache,
            weights_only=False,
            map_location="cpu"
        )

        cached_crop_ids = torch.load(
            crop_ids_cache,
            weights_only=False,
        )

        print(
            f"  {features.shape[0]} features, "
            f"{len(cached_crop_ids)} crop ids"
        )

    else:

        features = None
        cached_crop_ids = []

    cached_set = set(cached_crop_ids)

    missing_crop_ids = [
        cid
        for cid in crop_ids
        if cid not in cached_set
    ]

    if missing_crop_ids:

        print(
            f"Computing features for "
            f"{len(missing_crop_ids)} images..."
        )

        new_features, ordered_new_ids = (
            _compute_global_features(
                model,
                crop_path,
                missing_crop_ids[:1],
                device,
                input_transform,
                batch_size,
            )
        )
        new_features = new_features.cpu()
        print(new_features.shape)

        if features is None:
            features = new_features

        else:
            features = torch.cat(
                [
                    features,
                    new_features,
                ],
                dim=0,
            )

        cached_crop_ids.extend(
            ordered_new_ids
        )

        torch.save(
            features,
            features_cache,
        )

        torch.save(
            cached_crop_ids,
            crop_ids_cache,
        )

        print(
            f"  Cache updated at {cache_dir}"
        )

    else:

        print(
            "  All requested crop ids already "
            "exist in cache."
        )

    return features, cached_crop_ids


# ============================================================
# METADATA
# ============================================================

def convert_metadata_value(value):

    if torch.is_tensor(value):
        return value.item()

    if isinstance(value, np.generic):
        return value.item()

    return value


# ============================================================
# SVM PREDICTIONS
# ============================================================

def build_prediction_rows(
    svm,
    X,
    metadata_rows,
    class_names,
):
    """
    Build prediction rows from SVM decision scores.
    """

    decision_scores = svm.decision_function(X)

    predictions = svm.predict(X)

    rows = []

    num_classes = len(class_names)

    for i in range(len(X)):

        scores = decision_scores[i]

        # LinearSVC returns one score per class.
        if num_classes == 2 and scores.ndim == 0:

            # This case is unlikely with the current multiclass
            # elephant identification setup, but is handled explicitly.
            score = float(scores)

            pred_label = int(predictions[i])

            sorted_scores = np.array(
                [-score, score]
            )

        else:

            sorted_indices = np.argsort(
                scores
            )[::-1]

            pred_label = int(
                predictions[i]
            )

            sorted_scores = scores[
                sorted_indices
            ]

        if num_classes > 1:

            margin = float(
                sorted_scores[0]
                - sorted_scores[1]
            )

        else:

            margin = 1.0

        row_metadata = {
            key: convert_metadata_value(value)
            for key, value in metadata_rows[i].items()
        }

        row = {
            key: row_metadata[key]
            for key in row_metadata
        }

        row.update({
            "decision_scores": scores.tolist()
            if np.ndim(scores) > 0
            else [float(scores)],

            "decision_score_pred": float(
                scores[pred_label]
                if np.ndim(scores) > 0
                else scores
            ),

            "pred": pred_label,

            "margin": margin,

            "pred_label": class_names[pred_label],
        })

        rows.append(row)

    return rows


# ============================================================
# TRAIN ONE SVM
# ============================================================

def train_one_run(
    seed: int,
    train_features,
    train_labels,
    val_features,
    val_labels,
    train_metadata,
    val_metadata,
    class_names,
    num_classes,
):

    """
    Fit PCA and a Linear SVM for one train/validation split.

    PCA is fitted only on the training data.
    """

    set_seed(seed)

    # --------------------------------------------------------
    # Convert PyTorch tensors to NumPy arrays
    # --------------------------------------------------------

    X_train = train_features.cpu().numpy()
    y_train = train_labels.cpu().numpy()

    X_val = val_features.cpu().numpy()
    y_val = val_labels.cpu().numpy()

    original_dim = X_train.shape[1]

    print(
        f"  Original feature dimension: "
        f"{original_dim}"
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    if DIM_REDUC is not None:

        if DIM_REDUC >= original_dim:

            raise ValueError(
                f"DIM_REDUC={DIM_REDUC} must be "
                f"smaller than the original dimension "
                f"{original_dim}."
            )

        print(
            f"  PCA: {original_dim} -> "
            f"{DIM_REDUC}"
        )

        pca = PCA(
            n_components=DIM_REDUC,
            random_state=seed,
        )

        # IMPORTANT:
        # PCA is fitted only on the training set.
        X_train_reduced = pca.fit_transform(
            X_train
        )

        X_val_reduced = pca.transform(
            X_val
        )

        explained_variance = (
            pca.explained_variance_ratio_.sum()
            * 100
        )

        print(
            f"  PCA explained variance: "
            f"{explained_variance:.2f}%"
        )

    else:

        pca = None

        X_train_reduced = X_train
        X_val_reduced = X_val

        print(
            "  PCA disabled."
        )

    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    print(
        f"  Linear SVM: C={SVM_C}"
    )

    svm = LinearSVC(
        C=SVM_C,
        max_iter=SVM_MAX_ITER,
        random_state=seed,
    )

    svm.fit(
        X_train_reduced,
        y_train,
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    train_pred = svm.predict(
        X_train_reduced
    )

    val_pred = svm.predict(
        X_val_reduced
    )

    train_accuracy = (
        np.mean(train_pred == y_train)
        * 100
    )

    val_accuracy = (
        np.mean(val_pred == y_val)
        * 100
    )

    print(
        f"  Train accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"  Val accuracy: "
        f"{val_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Prediction rows
    # --------------------------------------------------------

    val_prediction_rows = build_prediction_rows(
        svm,
        X_val_reduced,
        val_metadata,
        class_names,
    )

    train_prediction_rows = build_prediction_rows(
        svm,
        X_train_reduced,
        train_metadata,
        class_names,
    )

    return {
        "svm": svm,
        "pca": pca,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "train_predictions": train_prediction_rows,
        "val_predictions": val_prediction_rows,
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    transform_name,
    runs,
    split_dfs,
    crop_to_global_idx,
    all_features,
    out_dir,
    model_dir,
):

    """
    Train one PCA + SVM for every train/validation split.
    """

    metadata_cols = [
        "run",
        "indiv_name",
        "camera",
        "mode",
        "video",
        entry_col,
        "split",
        "crop_id",
    ]

    all_val_rows = []
    all_train_rows = []

    run_results = []

    for run in runs:

        print(
            f"\n===== "
            f"[{transform_name}] "
            f"Run {run} "
            f"====="
        )

        df = split_dfs[run]

        train_df = df[
            df["split"] == "train"
        ].copy()

        val_df = df[
            df["split"] == "val"
        ].copy()

        # Keep only crops for which ResNet features exist.
        train_df = train_df[
            train_df["crop_id"].isin(
                crop_to_global_idx
            )
        ]

        val_df = val_df[
            val_df["crop_id"].isin(
                crop_to_global_idx
            )
        ]

        # ----------------------------------------------------
        # Label mapping
        # ----------------------------------------------------

        class_names = sorted(
            df["indiv_name"].unique()
        )

        label_map = {
            name: idx
            for idx, name
            in enumerate(class_names)
        }

        num_classes = len(class_names)

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        train_gidx = torch.tensor(
            [
                crop_to_global_idx[cid]
                for cid in train_df["crop_id"]
            ],
            dtype=torch.long,
        )

        val_gidx = torch.tensor(
            [
                crop_to_global_idx[cid]
                for cid in val_df["crop_id"]
            ],
            dtype=torch.long,
        )

        train_features = all_features[
            train_gidx
        ]

        val_features = all_features[
            val_gidx
        ]

        train_labels = torch.tensor(
            [
                label_map[name]
                for name
                in train_df["indiv_name"]
            ],
            dtype=torch.long,
        )

        val_labels = torch.tensor(
            [
                label_map[name]
                for name
                in val_df["indiv_name"]
            ],
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        train_metadata = (
            train_df[
                metadata_cols
            ]
            .to_dict("records")
        )

        val_metadata = (
            val_df[
                metadata_cols
            ]
            .to_dict("records")
        )

        print(
            f"  Classes: {num_classes}, "
            f"Train: {len(train_features)}, "
            f"Val: {len(val_features)}"
        )

        # ----------------------------------------------------
        # Train PCA + SVM
        # ----------------------------------------------------

        result = train_one_run(
            seed=SEED,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            class_names=class_names,
            num_classes=num_classes,
        )

        # ----------------------------------------------------
        # Save models
        # ----------------------------------------------------

        model_data = {
            "svm": result["svm"],
            "pca": result["pca"],
            "class_names": class_names,
            "label_map": label_map,
            "dim_reduc": DIM_REDUC,
            "svm_C": SVM_C,
        }

        model_path = (
            model_dir
            / f"model_run_{run}.pkl"
        )

        import pickle

        with open(
            model_path,
            "wb",
        ) as f:
            pickle.dump(
                model_data,
                f,
            )

        print(
            f"  Model saved: {model_path}"
        )

        # ----------------------------------------------------
        # Collect predictions
        # ----------------------------------------------------

        all_val_rows.extend(
            result["val_predictions"]
        )

        all_train_rows.extend(
            result["train_predictions"]
        )

        run_results.append({
            "run": run,
            "n_train": len(train_features),
            "n_val": len(val_features),
            "n_classes": num_classes,
            "train_accuracy": result[
                "train_accuracy"
            ],
            "val_accuracy": result[
                "val_accuracy"
            ],
        })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    columns = metadata_cols + [
        "decision_scores",
        "decision_score_pred",
        "pred",
        "margin",
        "pred_label",
    ]

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    val_pred_df = pd.DataFrame(
        all_val_rows,
        columns=columns,
    )

    val_pred_path = (
        out_dir / "val_pred.csv"
    )

    val_pred_df.to_csv(
        val_pred_path,
        index=False,
    )

    print(
        f"\nValidation predictions saved: "
        f"{val_pred_path}"
    )

    # --------------------------------------------------------
    # Train predictions
    # --------------------------------------------------------

    train_pred_df = pd.DataFrame(
        all_train_rows,
        columns=columns,
    )

    train_pred_path = (
        out_dir / "train_pred.csv"
    )

    train_pred_df.to_csv(
        train_pred_path,
        index=False,
    )

    print(
        f"Train predictions saved: "
        f"{train_pred_path}"
    )

    # --------------------------------------------------------
    # Accuracy summary
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        run_results
    )

    results_path = (
        out_dir / "svm_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(
        f"Run results saved: "
        f"{results_path}"
    )

    print(
        "\nMean validation accuracy: "
        f"{results_df['val_accuracy'].mean():.2f}%"
    )

    print(
        "Std validation accuracy: "
        f"{results_df['val_accuracy'].std():.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    set_seed(SEED)

    results_folder = (
        RESULTS_PATH
        / version
        / crop_type
    )

    out_dir = (
        results_folder
        / OUT_DIR_NAME
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    print(
        f"{crop_type} - {mode}"
    )

    print(
        f"SVM C = {SVM_C}"
    )

    print(
        f"PCA dimensions = {DIM_REDUC}"
    )

    # ========================================================
    # LOAD RUNS
    # ========================================================

    split_df = pd.read_csv(
        SPLIT_FILE
    ).filter(['run',entry_col,'split']).drop_duplicates()

    for col in (
        "run",
        entry_col,
        "split",
    ):

        if col not in split_df.columns:

            raise ValueError(
                f"Column '{col}' absent from "
                f"{SPLIT_FILE}"
            )

    runs = sorted(
        split_df["run"].unique()
    )

    NUM_RUNS = len(runs)

    if NUM_RUNS == 0:

        raise ValueError(
            f"No run found in {SPLIT_FILE}."
        )

    print(
        f"Number of runs: {NUM_RUNS}"
    )

    # ========================================================
    # CROPS + ENTRY
    # ========================================================

    crops_df = (
        pd.read_csv(CROPS_FILE)
        .dropna(
            subset=[
                "indiv_name",
                "crop_id",
            ]
        )
    )

    crops_df = crops_df[
        crops_df.indiv_name != "X"
    ]

    # summary_df = (
    #     pd.read_excel(
    #         SUMMARY_FILE,
    #         engine="odf",
    #         sheet_name="clips",
    #     )
    #     .filter([
    #         "video",
    #         entry_col,
    #         "mode",
    #         "camera",
    #     ])
    # )

    # crops_df = crops_df.merge(
    #     summary_df,
    #     on="video",
    # )

    if mode and "mode" in crops_df.columns:

        crops_df = crops_df[
            crops_df["mode"] == mode
        ]

    # One crop belongs to one entry.
    crops_df = crops_df.drop_duplicates(
        subset=["crop_id"]
    )

    # ========================================================
    # CROP-LEVEL DATAFRAME FOR EACH RUN
    # ========================================================

    split_dfs = {}

    all_crop_ids = set()

    for run in runs:

        df = (
            split_df[
                split_df["run"] == run
            ][
                [
                    "run",
                    entry_col,
                    "split",
                ]
            ]
            .merge(
                crops_df,
                on=entry_col,
                how="inner",
            )
        )

        split_dfs[run] = df

        all_crop_ids.update(
            df["crop_id"]
        )

    # ========================================================
    # RESNET152 BASE MODEL
    # ========================================================

    print(
        "\nLoading ResNet152..."
    )

    base_model = models.resnet152(
        weights="DEFAULT"
    )

    valid_crop_ids = sorted(
        cid
        for cid in all_crop_ids
        if (
            crop_path / cid
        ).exists()
    )

    print(
        f"Valid crop IDs: "
        f"{len(valid_crop_ids)} / "
        f"{len(all_crop_ids)}"
    )

    # ========================================================
    # TRANSFORMATIONS
    # ========================================================

    for (
        transform_name,
        image_transform,
    ) in TRANSFORMATIONS:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"TRANSFORMATION: "
            f"{transform_name}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Output directories
        # ----------------------------------------------------

        exp_out_dir = (
            out_dir
            / transform_name
        )

        exp_model_dir = (
            MODEL_PATH
            / transform_name
        )

        os.makedirs(
            exp_out_dir,
            exist_ok=True,
        )

        os.makedirs(
            exp_model_dir,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # ResNet input pipeline
        # ----------------------------------------------------

        input_transform = (
            build_input_transform(
                image_transform
            )
        )

        # ----------------------------------------------------
        # Feature cache
        # ----------------------------------------------------

        cache_name = (
            "all"
            if transform_name == "original"
            else transform_name
        )

        # global_cache_dir = (
        #     FEATURES_PATH
        #     / version
        #     / crop_type
        #     / cache_name
        # )

        all_features, cached_crop_ids = (
            compute_or_load_global_features(
                base_model,
                crop_path,
                valid_crop_ids,
                FEATURES_PATH,
                DEVICE,
                input_transform,
                batch_size,
            )
        )

        crop_to_global_idx = {
            cid: i
            for i, cid
            in enumerate(cached_crop_ids)
        }

        # ----------------------------------------------------
        # PCA + SVM for all runs
        # ----------------------------------------------------

        run_experiment(
            transform_name,
            runs,
            split_dfs,
            crop_to_global_idx,
            all_features,
            exp_out_dir,
            exp_model_dir,
        )

    print(
        "\nExperiment completed."
    )


if __name__ == "__main__":
    main()