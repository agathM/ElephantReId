import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import pdb

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.utils.data import Dataset, TensorDataset
from PIL import Image
from transformers import AutoModel
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

import cv2

# ============================================================
# PARAMETERS
# ============================================================
SEEDS = [1,10,42,50,60]  # seed fixe — seule la partition train/val change entre les runs

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_name = "cornwell_resnet50"
head = "SVM"
DIM_REDUC = 2000
SVM_C = 1
SVM_MAX_ITER = 10000
OUT_DIR_NAME = features_name + head
if DIM_REDUC:
    OUT_DIR_NAME = f"{OUT_DIR_NAME}_{DIM_REDUC}_C_{SVM_C}"
version = "v9"
crop_type = "Ear"
mode = "Day"
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers/") / version / crop_type / OUT_DIR_NAME
os.makedirs(RESULTS_PATH,exist_ok=True)
FEATURES_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features") / version / crop_type / features_name
os.makedirs(FEATURES_PATH,exist_ok=True)
MODEL_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/models/identification") / version / crop_type / OUT_DIR_NAME
os.makedirs(MODEL_PATH, exist_ok=True)

# CSV des runs : colonnes run, entry, split (train/val)
SPLIT_FILE = Path(
    "/sps/mnhn/ameurisse/ElephantReId/"
    "data/processed/crops/classifiers/split/"
    "cornwell_ears_split.csv"
)

CROPS_FILE = Path(
    "/sps/mnhn/ameurisse/ElephantReId/"
    "data/processed/crops/classifiers/split/cornwell_ears_split.csv"
)

SUMMARY_FILE = Path(
    "/sps/mnhn/ameurisse/ElephantReId/"
    "data/processed/crops/classifiers/raw/"
    "v3_summary_videos.ods"
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
# hyperparamètres d'entraînement
batch_size = 64
num_epoch = 30
lr = 0.001

# early stopping : seuils d'amélioration d'accuracy train à tester.
# On entraîne TOUJOURS jusqu'à num_epoch, puis on sauvegarde les
# prédictions de validation à l'epoch où l'on se serait arrêté pour
# chaque seuil (et à la dernière epoch).
MIN_ACC_IMPROVEMENTS = [1.0, 2.0, 5.0, 10.0]


# ============================================================
# TRANSFORMATIONS D'IMAGES
# ============================================================
# Chaque transformation prend une image PIL (RGB) et renvoie une image
# PIL (RGB). Elle est appliquée AVANT le Resize + ToTensor + Normalize
# de la pipeline ResNet50 (voir build_input_transform).

# Coefficients de luminance (Rec. 601) — mêmes que montage_grayscale.py.
LUMINANCE = np.array([0.299, 0.587, 0.114])

# Classification head with dropout regularization.
class ClassificationHead(nn.Module):
    def __init__(self, dim_out, dim_in, dropout=0.5):
        super(ClassificationHead, self).__init__()

        self.out_features = dim_out
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(
            in_features=dim_in,
            out_features=self.out_features
        )

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
        
def transform_brightness_normalized(image, target_mean=0.5):
    """
    Normalise la luminosité moyenne de l'image.

    Chaque canal RGB est multiplié par target_mean / luminosité_moyenne.
    Même principe que normalize_mean_brightness de montage_grayscale.py.
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



# Liste des 6 transformations testées : (nom, fonction d'image).
TRANSFORMATIONS = [
    ("brightness_normalized", transform_brightness_normalized),
]


# ============================================================
# PIPELINE D'ENTRÉE PAR TRANSFORMATION
# ============================================================

def build_input_transform(image_transform):
    """
    Pipeline ResNet50 complète pour une transformation d'image :

        image_transform -> Resize(224) -> ToTensor -> Normalize(ImageNet)

    `image_transform` est appliquée sur l'image PIL avant le redimensionnement.
    """
    transforms_list = []

    transforms_list.append(image_transform)

    # transforms_list.extend([
    #     transforms.Resize([224, 224]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    transforms_list = [
    transforms.Resize((224, 224)),
    # transforms.Pad(
    #     padding=(108, 108, 108, 108),
    #     fill=0,
    # ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
    return transforms.Compose(transforms_list)

def convert_metadata_value(value):

    if torch.is_tensor(value):
        return value.item()

    if isinstance(value, np.generic):
        return value.item()

    return value

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class GlobalCropDataset(Dataset):
    """Simple dataset over a list of crop_ids. Labels are dummy — only the
    images are needed to precompute features."""

    def __init__(self, crop_path: Path, crop_ids: list, transform):
        self.crop_path = crop_path
        self.crop_ids = crop_ids
        self.transform = transform

    def __len__(self):
        return len(self.crop_ids)

    def __getitem__(self, idx):
        image = Image.open(self.crop_path / self.crop_ids[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0, idx


class MetadataTensorDataset(Dataset):
    """Dataset of (features, label, metadata) where `metadata` is a dict of
    crop metadata (run, indiv_name, camera, ..., crop_id) needed for
    val_pred.csv. The collator turns the per-sample dicts into one dict of
    batches, which is exactly what train_one_run expects for validation."""

    def __init__(self, features, labels, metadata_rows):
        self.features = features
        self.labels = labels
        self.metadata_rows = metadata_rows  # list of dicts, aligned with features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.metadata_rows[idx]


def _compute_global_features(
    model,
    crop_path: Path,
    crop_ids: list,
    device: torch.device,
    input_transform,
    batch_size: int = 64,
) -> tuple:
    """Compute ResNet50 features (input of fc, 2048-d) for `crop_ids`
    after applying `input_transform` (qui inclut la transformation d'image).

    Returns
    -------
    (features, crop_ids)
        features  : torch.Tensor (N, 2048)
        crop_ids  : list[str] parallel to `features`
    """
    dataset = GlobalCropDataset(crop_path, crop_ids, input_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.to(device)
    model.eval()

    batch_features = []
    batch_idxs = []

    def hook(module, input, output):
        # input[0] : [B, 2048] tensor fed to fc (same as precompute_features)
        batch_features.append(input[0].detach().cpu())

    handle = model.fc.register_forward_hook(hook)

    with torch.no_grad():
        for images, _, idx in loader:
            images = images.to(device)
            _ = model(images)
            batch_idxs.append(idx)

    handle.remove()

    features = torch.cat(batch_features, dim=0)
    idxs = torch.cat(batch_idxs, dim=0)

    # Reorder by dataset index (loader may not keep global order in all cases)
    order = idxs.argsort()
    features = features[order]
    ordered_crop_ids = [crop_ids[i] for i in order.tolist()]

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
    Precompute ResNet50 features (input of fc, 2048-d) for all the crop ids
    shared by every run of ONE transformation.

    Since all the runs use the same images, the features are the same for
    every run — we compute them once and reuse them. If a cache already
    exists but some crop_ids are missing from it, only the missing images
    are computed and appended to the existing cache.

    Returns
    -------
    (features, crop_ids)
        features  : torch.Tensor (N, 2048)
        crop_ids  : list[str] parallel to `features`
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_cache = cache_dir / "features.pt"
    crop_ids_cache = cache_dir / "crop_ids.pt"

    if features_cache.exists() and crop_ids_cache.exists():
        print(f"Loading cached global features ({features_cache})...")
        features = torch.load(features_cache, weights_only=False)
        cached_crop_ids = torch.load(crop_ids_cache, weights_only=False)
        print(f"  {features.shape[0]} features, {len(cached_crop_ids)} crop ids")
    else:
        features = None
        cached_crop_ids = []

    cached_set = set(cached_crop_ids)
    missing_crop_ids = [cid for cid in crop_ids if cid not in cached_set]

    if missing_crop_ids:
        print(
            f"Computing features for {len(missing_crop_ids)} images "
            f"missing from the cache (once)..."
        )
        new_features, ordered_new_ids = _compute_global_features(
            model,
            crop_path,
            missing_crop_ids,
            device,
            input_transform,
            batch_size,
        )
        if features is None:
            features = new_features
        else:
            features = torch.cat([features, new_features], dim=0)
        cached_crop_ids.extend(ordered_new_ids)

        torch.save(features, features_cache)
        torch.save(cached_crop_ids, crop_ids_cache)
        print(
            f"  Cache updated at {cache_dir} "
            f"({features.shape[0]} features, {len(cached_crop_ids)} crop ids)"
        )
    elif features is not None:
        print("  All requested crop ids already in cache; nothing to recompute.")

    return features, cached_crop_ids
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
    split_df,
    crop_to_global_idx,
    all_features,
    out_dir,
    model_dir,
):

    """
    Train one PCA + SVM for every train/validation split.
    """

    metadata_cols = [
        "indiv_name",
        "split",
        "crop_id",
    ]

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]

    # Ne garder que les crop_ids dont on a les features
    train_df = train_df[train_df["crop_id"].isin(crop_to_global_idx)]
    val_df = val_df[val_df["crop_id"].isin(crop_to_global_idx)]
    # Label map de ce run (ordre alphabétique des individus)
    class_names = sorted(split_df["indiv_name"].unique())
    label_map = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    # Récupérer les features déjà calculées + les labels
    train_gidx = torch.tensor(
        [crop_to_global_idx[cid] for cid in train_df["crop_id"]]
    )
    val_gidx = torch.tensor(
        [crop_to_global_idx[cid] for cid in val_df["crop_id"]]
    )

    train_features = all_features[train_gidx]
    val_features = all_features[val_gidx]
    train_labels = torch.tensor(
        [label_map[name] for name in train_df["indiv_name"]]
    )
    val_labels = torch.tensor(
        [label_map[name] for name in val_df["indiv_name"]]
    )

    train_features_dataset = TensorDataset(
        train_features, train_labels, torch.arange(len(train_features))
    )
    # Les métadonnées (alignées avec val_features / val_labels) servent à
    # construire val_pred.csv ; train_one_run attend un dict en 3e élément.
    val_metadata = val_df[metadata_cols].to_dict("records")
    train_metadata = (train_df[metadata_cols].to_dict("records"))
    val_features_dataset = MetadataTensorDataset(
        val_features, val_labels, val_metadata
    )

    dim_in = train_features.shape[1]

    print(
        f"  Classes: {num_classes}, Train: {len(train_features_dataset)}, "
        f"Val: {len(val_features_dataset)}"
    )

    for seed in SEEDS:
        set_seed(seed)
        # ----------------------------------------------------
        # Train PCA + SVM
        # ----------------------------------------------------

        result = train_one_run(
            seed=seed,
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
            "seed": seed,
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
        "seed"
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

    print(f"{crop_type} - {mode}")

    # ============================================================
    # LOAD RUNS (run, entry, split)
    # ============================================================
    split_df = pd.read_csv(CROPS_FILE)
    all_crop_ids = set(split_df.crop_id)
    # ============================================================
    # MODÈLE DE BASE (UNE SEULE FOIS, commun à toutes les transformations)
    # ============================================================
    # model_tag = f"conservationxlabs/miewid-msv3"
    # base_model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
    base_model = models.resnet50(pretrained="True")
    valid_crop_ids = sorted(
        cid for cid in all_crop_ids if (crop_path / cid).exists()
    )
    print(f"\nCrop IDs valides : {len(valid_crop_ids)} / {len(all_crop_ids)}")

    # ============================================================
    # BOUCLE SUR LES 6 TRANSFORMATIONS
    # ============================================================
    for transform_name, image_transform in TRANSFORMATIONS:

        print("\n" + "=" * 70)
        print(f"TRANSFORMATION : {transform_name}")
        print("=" * 70)

        # Pipeline d'entrée : transformation d'image + Resize + Normalize
        input_transform = build_input_transform(image_transform)

        # Cache de features DÉDIÉ à cette transformation.
        # Pour "original" on réutilise le cache "all" existant (identique
        # à la pipeline de train_10_runs_splits.py).

        all_features, cached_crop_ids = compute_or_load_global_features(
            base_model,
            crop_path,
            valid_crop_ids,
            FEATURES_PATH,
            DEVICE,
            input_transform,
        )
        crop_to_global_idx = {cid: i for i, cid in enumerate(cached_crop_ids)}

        # Lancement des 100 runs pour cette transformation
        run_experiment(
            transform_name,
            input_transform,
            split_df,
            crop_to_global_idx,
            all_features,
            RESULTS_PATH,
            MODEL_PATH,
        )

    print(f"\nPour les visualisations et les stats globales, lancer : scripts/analyze_splits.py")


if __name__ == "__main__":
    main()