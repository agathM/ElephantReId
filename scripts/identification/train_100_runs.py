"""
Entraîne la tête de classification (features ResNet50 précalculées) sur
tous les runs définis par un CSV de splits, pour CHAQUE transformation
d'image.

Reprend exactement la logique de train_10_runs_splits.py, mais ajoute
une boucle externe sur les transformations d'images :

    pour chaque transformation :
        pour chaque run (100 runs) :
            entraîne la tête de classification sur les features
            ResNet50 précalculées sur les images TRANSFORMÉES

Le CSV de splits (SPLIT_FILE) contient les colonnes :
    run, entry, split (train/val)
    - run   : numéro du run (les runs partagent les mêmes images ; seule
              la partition train/val des `entry` change)
    - entry : identifiant d'une entrée (groupe de crops)
    - split : 'train' ou 'val' pour ce run

Les crops associés à chaque entry sont résolus via le fichier des crops
(CROPS_FILE, ears_detections_day_with_names.csv) qui contient maintenant
une colonne `entry` (ajoutée en mergeant avec summary_all.csv sur
video + indiv_name). Il fournit, pour chaque crop, sa entry, son individu
(indiv_name), sa caméra, son mode et sa vidéo.

Pour chaque transformation, le script produit, dans un SOUS-DOSSIER de
<out_dir> nommé d'après la transformation :

    <out_dir>/<transform>/train_curv_run_<run>.png   courbes loss/accuracy
    <out_dir>/<transform>/val_pred.csv               prédictions de
                                                     validation (dernière
                                                     epoch, tous les runs)
    <out_dir>/<transform>/val_pred_min_acc_<X>.csv   prédictions à l'epoch
                                                     où l'on se serait
                                                     arrêté pour chaque
                                                     seuil X de
                                                     MIN_ACC_IMPROVEMENT

Les features ResNet50 sont précalculées UNE FOIS par transformation (et
mises en cache par transformation) : toutes les runs d'une transformation
partagent les mêmes features.
"""

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

import cv2

# ============================================================
# PARAMETERS
# ============================================================
SEED = 42  # seed fixe — seule la partition train/val change entre les runs

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_name = "brightness_normalize_pad_miewid"
head = "_entry_day_FC_drop0.2"
OUT_DIR_NAME = features_name + head
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
    "split_by_videos.csv"
)

RUNS = [97, 57, 98, 94, 6]

CROPS_FILE = Path(
    "/sps/mnhn/ameurisse/ElephantReId/"
    "data/processed/crops/classifiers/split/crops_0.8_h_1_v_1.6_nc_2.7.csv"
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
entry_col = "video"
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
    transforms_list.extend([
    transforms.Resize((224, 224)),
    transforms.Pad(
        padding=(108, 108, 108, 108),
        fill=0,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    return transforms.Compose(transforms_list)


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
        batch_features.append(output.detach().cpu())

    handle = model.bn.register_forward_hook(hook)

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


def find_early_stop_epoch(train_accuracies: list, min_acc_improvement: float) -> int:
    """Indice de l'epoch (0-based) où l'early stopping se serait déclenché.

    On se serait arrêté à la première epoch `e` (e >= 1) dont l'amélioration
    d'accuracy train par rapport à l'epoch précédente est strictement
    inférieure à `min_acc_improvement` %. Si aucun seuil ne déclenche
    l'arrêt, on retourne l'indice de la dernière epoch.
    """
    for epoch in range(1, len(train_accuracies)):
        if train_accuracies[epoch - 1] > 0:
            improvement = (
                (train_accuracies[epoch] - train_accuracies[epoch - 1])
                / train_accuracies[epoch - 1]
                * 100
            )
            if improvement < min_acc_improvement:
                return epoch
    return len(train_accuracies) - 1


def save_train_curves(
    out_dir: Path,
    run: int,
    train_error: list,
    val_error: list,
    train_accuracy: list,
    val_accuracy: list,
    num_classes: int,
) -> Path:
    """Enregistre les courbes d'entraînement et de validation pour un run.

    Reprend le tracé de train_with_config.py : loss (haut) et accuracy
    avec le niveau "random" 1/num_classes (bas).
    """
    epochs = range(1, len(train_accuracy) + 1)

    plt.figure(figsize=(4, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_error, color="blue", label="train")
    plt.plot(epochs, val_error, color="orange", label="validation")
    plt.title("Loss")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracy, color="blue", label="train")
    plt.plot(epochs, val_accuracy, color="orange", label="validation")
    plt.plot(
        epochs,
        [1 / num_classes * 100 for _ in epochs],
        color="red",
        label="random",
    )
    plt.title("Accuracy")
    plt.tight_layout()

    path = out_dir / f"train_curv_run_{run}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def train_one_run(
    seed: int,
    train_features_dataset,
    val_features_dataset,
    class_names: list,
    num_classes: int,
    dim_in: int,
    device: torch.device,
):
    """Train the classification head once with the given seed.

    L'entraînement parcourt TOUJOURS tous les epochs (num_epoch), sans
    early stopping : les courbes et les prédictions de validation de chaque
    epoch sont collectées pour décider après coup des points de sauvegarde.

    Returns
    -------
    dict
        - train_error / val_error : list[float] (loss par epoch)
        - train_accuracy / val_accuracy : list[float] (accuracy % par epoch)
        - epoch_val_predictions : dict[int, list[dict]]
            Prédictions de validation (rows prêtes pour val_pred.csv) par
            indice d'epoch (0-based).
    """

    set_seed(seed)

    head = ClassificationHead(
        dim_out=num_classes,
        dim_in=dim_in,
    ).to(device)

    if head.out_features != num_classes:
        raise AssertionError(
            "Classification head output size mismatch."
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_features_dataset,
        shuffle=True,
        generator=generator,
        batch_size=batch_size,
    )

    val_loader = DataLoader(
        val_features_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    num_train = len(train_features_dataset)
    num_val = len(val_features_dataset)

    train_error = []
    val_error = []
    train_accuracy = []
    val_accuracy = []

    # Prédictions de validation par epoch {epoch_index: [rows]}
    epoch_val_predictions = {}

    for epoch in range(num_epoch):

        # ====================================================
        # TRAINING
        # ====================================================

        head.train()

        error = 0.0
        corrects = 0

        for feat, label, _ in train_loader:

            feat = feat.to(device)
            label = label.to(device)

            output = head(feat)

            loss = criterion(
                output,
                label,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            error += loss.item()

            _, predicted = torch.max(
                output,
                1,
            )

            corrects += (
                predicted == label
            ).sum().item()

        error /= len(train_loader)

        train_error.append(error)

        train_accuracy.append(
            corrects * 100 / num_train
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        head.eval()

        error = 0.0
        corrects = 0

        val_prediction_rows = []

        with torch.no_grad():

            for feat, label, metadata in val_loader:

                feat = feat.to(device)
                label = label.to(device)

                output = head(feat)

                loss = criterion(
                    output,
                    label,
                )

                error += loss.item()

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )

                predicted = torch.argmax(
                    output,
                    dim=1,
                )

                corrects += (
                    predicted == label
                ).sum().item()

                # --------------------------------------------
                # Save predictions for this epoch
                # --------------------------------------------

                logits = output.cpu()
                predicted_cpu = predicted.cpu()
                probabilities_cpu = probabilities.cpu()

                for i in range(len(label)):

                    prob = probabilities_cpu[i]

                    # Probability of predicted class
                    conf_pred_label = (
                        prob[predicted_cpu[i]].item()
                    )

                    # Top two probabilities
                    sorted_probs, _ = torch.sort(
                        prob,
                        descending=True,
                    )

                    if num_classes > 1:
                        margin = (
                            sorted_probs[0].item()
                            - sorted_probs[1].item()
                        )
                    else:
                        margin = 1.0

                    # Entropy
                    entropy = -torch.sum(
                        prob * torch.log(
                            prob.clamp(min=1e-12)
                        )
                    ).item()

                    # Convert metadata to a dictionary
                    if isinstance(metadata, dict):

                        row_metadata = {
                            key: metadata[key][i]
                            if torch.is_tensor(metadata[key])
                            else metadata[key][i]
                            if isinstance(
                                metadata[key],
                                (list, tuple, np.ndarray, pd.Series),
                            )
                            else metadata[key]
                            for key in metadata
                        }

                    else:
                        raise TypeError(
                            "Expected the third element returned "
                            "by the dataset to be a dictionary "
                            "containing crop metadata."
                        )

                    # Convert tensors / numpy values to Python values
                    for key, value in row_metadata.items():

                        if torch.is_tensor(value):
                            row_metadata[key] = value.item()

                        elif isinstance(
                            value,
                            np.generic,
                        ):
                            row_metadata[key] = value.item()

                    pred_label = predicted_cpu[i].item()

                    row = {
                        key: row_metadata[key]
                        for key in row_metadata
                    }
                    row.update({
                        "logits": logits[i].tolist(),
                        "pred": pred_label,
                        "conf_pred_label": conf_pred_label,
                        "margin": margin,
                        "entropy": entropy,
                        "pred_label": class_names[pred_label],
                    })

                    val_prediction_rows.append(row)

        error /= len(val_loader)

        val_error.append(error)

        val_accuracy.append(
            corrects * 100 / num_val
        )

        scheduler.step(error)

        # On conserve les prédictions de validation de CETTE epoch
        # pour pouvoir générer val_pred.csv à n'importe quel point
        # d'arrêt (early stopping simulé a posteriori).
        epoch_val_predictions[epoch] = val_prediction_rows

    return {
        "train_error": train_error,
        "val_error": val_error,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "epoch_val_predictions": epoch_val_predictions,
        "model_dict": head.state_dict()
    }


# ============================================================
# EXPÉRIENCE : UNE TRANSFORMATION + TOUS LES RUNS
# ============================================================

def run_experiment(
    transform_name,
    input_transform,
    runs,
    split_dfs,
    crop_to_global_idx,
    all_features,
    out_dir,
    model_dir,
):
    """Entraîne tous les runs pour UNE transformation d'images.

    out_dir  : dossier de sortie (sous-dossier de out_dir principal)
    model_dir: dossier où sont sauvegardés les state_dict par run
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

    # Lignes de prédiction de la dernière epoch (tous runs confondus)
    all_rows_last_epoch = []
    # {min_acc_improvement: lignes prédites à l'epoch d'arrêt simulé}
    min_acc_rows = {m: [] for m in MIN_ACC_IMPROVEMENTS}

    for run in RUNS:
        print(f"\n===== [{transform_name}] Run {run}/{len(runs)} — {run} =====")

        df = split_dfs[run]
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]

        # Ne garder que les crop_ids dont on a les features
        train_df = train_df[train_df["crop_id"].isin(crop_to_global_idx)]
        val_df = val_df[val_df["crop_id"].isin(crop_to_global_idx)]
        # Label map de ce run (ordre alphabétique des individus)
        class_names = sorted(df["indiv_name"].unique())
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
        val_features_dataset = MetadataTensorDataset(
            val_features, val_labels, val_metadata
        )

        dim_in = train_features.shape[1]

        print(
            f"  Classes: {num_classes}, Train: {len(train_features_dataset)}, "
            f"Val: {len(val_features_dataset)}"
        )

        result = train_one_run(
            SEED,
            train_features_dataset,
            val_features_dataset,
            class_names,
            num_classes,
            dim_in,
            DEVICE,
        )
        train_error = result["train_error"]
        val_error = result["val_error"]
        train_accuracy = result["train_accuracy"]
        val_accuracy = result["val_accuracy"]
        epoch_val_predictions = result["epoch_val_predictions"]
        model_state_dict = result["model_dict"]
        torch.save(model_state_dict, model_dir / f"model_state_dict_{run}.pth")
        print(f"  Val accuracy (dernière epoch) : {val_accuracy[-1]:.2f}%")

        # Courbes d'entraînement et de validation (comme train_with_config.py)
        curv_path = save_train_curves(
            out_dir,
            run,
            train_error,
            val_error,
            train_accuracy,
            val_accuracy,
            num_classes,
        )
        print(f"  Courbes sauvegardées : {curv_path}")

        # -- Prédictions de la dernière epoch --------------------------
        last_epoch = len(epoch_val_predictions) - 1
        last_rows = epoch_val_predictions[last_epoch]

        # -- Early stopping simulé pour chaque MIN_ACC_IMPROVEMENT -----
        stop_epochs = {}
        for min_acc in MIN_ACC_IMPROVEMENTS:
            stop_epoch = find_early_stop_epoch(train_accuracy, min_acc)
            stop_epochs[min_acc] = stop_epoch
            print(
                f"  MIN_ACC_IMPROVEMENT={min_acc}: arrêt simulé à l'epoch "
                f"{stop_epoch + 1}"
            )

        all_rows_last_epoch.extend(last_rows)
        for min_acc in MIN_ACC_IMPROVEMENTS:
            min_acc_rows[min_acc].extend(
                epoch_val_predictions[stop_epochs[min_acc]]
            )

    # ============================================================
    # SAUVEGARDE : val_pred.csv AVEC LA COLONNE `run`
    # ============================================================
    columns = metadata_cols + [
        "logits",
        "pred",
        "conf_pred_label",
        "margin",
        "entropy",
        "pred_label",
    ]

    # 1) Dernière epoch (comportement d'origine)
    val_pred_df = pd.DataFrame(all_rows_last_epoch, columns=columns)
    val_pred_path = out_dir / "val_pred.csv"
    val_pred_df.to_csv(val_pred_path, index=False)
    print(f"\nPrédictions de validation (dernière epoch, tous les runs) : {val_pred_path}")
    print(f"  {len(val_pred_df)} lignes, {val_pred_df['run'].nunique()} runs")

    # 2) Une époque d'arrêt simulée par MIN_ACC_IMPROVEMENT
    for min_acc in MIN_ACC_IMPROVEMENTS:
        pred_df = pd.DataFrame(min_acc_rows[min_acc], columns=columns)
        pred_path = out_dir / f"val_pred_min_acc_{min_acc}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(
            f"Prédictions de validation (MIN_ACC_IMPROVEMENT={min_acc}, "
            f"tous les runs) : {pred_path}"
        )
        print(f"  {len(pred_df)} lignes, {pred_df['run'].nunique()} runs")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(f"{crop_type} - {mode}")

    # ============================================================
    # LOAD RUNS (run, entry, split)
    # ============================================================
    split_df = pd.read_csv(SPLIT_FILE).filter(['entry_col','run','split']).drop_duplicates()
    for col in ("run", entry_col, "split"):
        if col not in split_df.columns:
            raise ValueError(
                f"Colonne '{col}' absente de {SPLIT_FILE}"
            )

    runs = sorted(split_df["run"].unique())
    NUM_RUNS = len(runs)
    if NUM_RUNS == 0:
        raise ValueError(f"Aucun run trouvé dans {SPLIT_FILE}.")
    print(f"Nombre de runs trouvés dans {SPLIT_FILE.name} : {NUM_RUNS}")

    # ============================================================
    # CROPS AVEC ENTRY (entry fournie par summary_all, via CROPS_FILE)
    # ============================================================
    crops_df = pd.read_csv(CROPS_FILE).dropna(subset=['indiv_name', 'crop_id'])
    crops_df = crops_df[crops_df.indiv_name != "X"]
    # crops_df = crops_df_
    # summary_df = pd.read_excel(SUMMARY_FILE, engine='odf', sheet_name='clips').filter(['video', 'entry'])
    # crops_df = crops_df.merge(summary_df, on='video')
    if mode and "mode" in crops_df.columns:
        crops_df = crops_df[crops_df["mode"] == mode]
    # Un crop n'appartient qu'à une seule entry : on retire les doublons
    crops_df = crops_df.drop_duplicates(subset=["crop_id"])

    # ============================================================
    # CROP-LEVEL DF PAR RUN
    # ============================================================
    split_dfs = {}
    all_crop_ids = set()
    for run in runs:
        df = split_df[split_df["run"] == run][["run", entry_col, "split"]].merge(
            crops_df,
            on=entry_col,
            how="inner",
        )
        split_dfs[run] = df
        all_crop_ids.update(df["crop_id"])

    # ============================================================
    # MODÈLE DE BASE (UNE SEULE FOIS, commun à toutes les transformations)
    # ============================================================
    model_tag = f"conservationxlabs/miewid-msv3"
    base_model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
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

        # Dossier de sortie dédié à cette transformation (sous-dossier
        # de out_dir) et dossier des model_state_dict dédié.
        # exp_out_dir = RESULTS_PATH / transform_name
        # exp_model_dir = MODEL_PATH / transform_name
        # os.makedirs(exp_out_dir, exist_ok=True)
        # os.makedirs(exp_model_dir, exist_ok=True)

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
            runs,
            split_dfs,
            crop_to_global_idx,
            all_features,
            RESULTS_PATH,
            MODEL_PATH,
        )

    print(f"\nPour les visualisations et les stats globales, lancer : scripts/analyze_splits.py")


if __name__ == "__main__":
    main()