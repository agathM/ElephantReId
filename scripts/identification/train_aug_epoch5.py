"""
Entraîne la tête de classification (features ResNet50 précalculées) sur
tous les runs définis par un CSV de splits, pour CHAQUE groupe
d'augmentation d'image testé séparément.

Reprend exactement la logique de train_10_runs_splits_transforms.py,
avec deux différences :

1. PREPROCESSING : brightness_normalize est appliqué en amont de TOUT
   (toutes les images passent par transform_brightness_normalized, que
   ce soit pour les features de base ou pour les features augmentées).

2. AUGMENTATION ACTIVE À PARTIR DE LA 5e EPOCH SEULEMENT :
       - epochs 1-4 : la tête est entraînée sur les features de base
         (brightness_normalize seul) ;
       - epochs 5-30 : la tête est entraînée sur les features calculées
         sur les images brightness_normalize + augmentation du groupe
         courant.
   La validation utilise TOUJOURS les features de base (non augmentées).

Chaque groupe d'augmentation est testé dans une expérience SÉPARÉE
(sous-dossier de <out_dir> nommé d'après le groupe) :

    affine                RandomAffine(degrees=10, translate=(0.05, 0.05),
                                       scale=(0.95, 1.05))
    gaussian_blur         RandomApply([GaussianBlur(5, sigma=(0.1, 1.5))],
                                      p=0.2)
    perspective_bilinear  RandomPerspective(distortion_scale=0.25 / 0.5
                          0.75, interpolation=BILINEAR)  [7 variants]
    perspective_nearest   RandomPerspective(distortion_scale=0.25 / 0.5,
                          interpolation=NEAREST)

Par groupe, le script produit dans <out_dir>/<groupe> :
    train_curv_run_<run>.png   courbes loss/accuracy
    val_pred.csv               prédictions de validation (dernière epoch,
                               tous les runs)
    val_pred_min_acc_<X>.csv   prédictions aux points d'arrêt simulés

Comme dans le script d'origine, les features ResNet50 sont précalculées
UNE SEULE FOIS par groupe (cache disque dédié) et partagées par les 100
runs : seule la partition train/val change entre les runs.

N.B. — TIRAGE DES AUGMENTATIONS :
    Le pipeline étant à base de features précalculées, l'augmentation est
    appliquée par tirage AVANT l'entraînement, pas batch par batch (on ne
    peut pas relancer ResNet50 dans la boucle d'entraînement).
    Avec REDRAW_PER_EPOCH = True (recommandé sur GPU), les features
    augmentées sont re-tirées à CHAQUE epoch : l'epoch e (>= 5) dispose
    de son propre jeu d'images augmentées, précalculé une seule fois et
    partagé par les 100 runs. La tête voit des versions qui changent
    d'epoch en epoch, ce qui correspond à l'effet de régularisation
    attendu de l'augmentation.
    Coût : (num_epoch - AUG_START_EPOCH + 1) forward ResNet50 par groupe
    (~1 min sur GPU pour ~4500 images) + ~37 Mo de cache disque et RAM
    par epoch et par groupe.
    Avec REDRAW_PER_EPOCH = False : une seule version augmentée, figée,
    utilisée des epochs 5 à 30 (coût minimal — à n'utiliser que sur une
    machine sans GPU ni place disque).
"""

import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from collections.abc import Callable

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.utils.data import Dataset, TensorDataset
from PIL import Image

# ============================================================
# PARAMETERS
# ============================================================
SEED = 42  # seed fixe — seule la partition train/val change entre les runs

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR_NAME = "100_runs_splits_aug_from_epoch5v2"
version = "v9"
crop_type = "Ear"
mode = "Day"
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers/")
FEATURES_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features")
MODEL_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/models/identification") / OUT_DIR_NAME
os.makedirs(MODEL_PATH, exist_ok=True)

# CSV des runs : colonnes run, entry, split (train/val)
SPLIT_FILE = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/split/splits_summary2.csv")
RUNS = [29,3,35,91,42]
# Fichier des crops (avec colonne `entry` ajoutée via summary_all.csv)
CROPS_FILE = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/raw/ears_day.csv")
SUMMARY_FILE = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/raw/v2_summary_videos.ods")
crop_path = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/") / crop_type / 'v5'

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
# AUGMENTATION : début et mode de tirage
# ============================================================
# L'augmentation n'est appliquée qu'à partir de cette epoch (1-based) :
# les AUG_START_EPOCH - 1 premières époques apprennent sur les features
# de base.
AUG_START_EPOCH = 5

# True  : les features augmentées sont re-tirées à CHAQUE epoch à partir
#         de l'epoch 5 (effet d'augmentation réel). Chaque epoch a son
#         propre cache disque ; le tirage est partagé par les 100 runs.
#         Coût : 1 forward ResNet50 par epoch et par groupe (rapide sur
#         GPU) + ~37 Mo de disque/RAM par epoch et par groupe.
# False : une seule version augmentée, figée, utilisée des epochs 5 à 30
#         (coût minimal — à n'utiliser que sans GPU).
REDRAW_PER_EPOCH = True


# ============================================================
# PREPROCESSING ET TRANSFORMATIONS D'IMAGES
# ============================================================
# Coefficients de luminance (Rec. 601) — mêmes que montage_grayscale.py.
LUMINANCE = np.array([0.299, 0.587, 0.114])

class ClassificationHead(nn.Module):
    def __init__(self,dim_out,dim_in):
        super(ClassificationHead, self).__init__()
        self.out_features = dim_out
        # self.linear1 = nn.Linear(in_features=dim_in, out_features=512)
        # self.linear2 = nn.Linear(in_features=512, out_features=128)
        # self.relu = nn.ReLU()
        # self.linear3 = nn.Linear(in_features=128, out_features=self.out_features)
        self.linear = nn.Linear(in_features=dim_in, out_features=self.out_features)

    def forward(self, x):
        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu(x)
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


# Chaque groupe d'augmentation est testé SÉPARÉMENT (sous-dossier dédié).
# L'augmentation est active uniquement à partir de AUG_START_EPOCH.
# (Décommenter "none" si l'on veut aussi le baseline brightness seule.)
AUGMENTATION_GROUPS = [
    # ("none", None),
    (
        "affine",
        transforms.RandomAffine(
            degrees=15,
            # translate=(0.3, 0.3),
            scale=(0.9, 1.25),
        ),
    ),
    (
        "gaussian_blur",
        transforms.RandomApply(
            [transforms.GaussianBlur(
                kernel_size=5,
                sigma=(0.1, 10),
            )],
            p=0.2,
        ),
    ),
    # Perspective avec interpolation bilinear : distortion_scale 0.25, 0.5
    # et 0.75 (p=0.5 par défaut dans torchvision).
    (
        "perspective_bilinear_025",
        transforms.RandomPerspective(
            distortion_scale=0.25,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ),
    (
        "perspective_bilinear_050",
        transforms.RandomPerspective(
            distortion_scale=0.5,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ),
    (
        "perspective_bilinear_075",
        transforms.RandomPerspective(
            distortion_scale=0.75,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ),
    # Perspective avec interpolation nearest : distortion_scale 0.25 et 0.5.
    (
        "perspective_nearest_025",
        transforms.RandomPerspective(
            distortion_scale=0.25,
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
    ),
    (
        "perspective_nearest_050",
        transforms.RandomPerspective(
            distortion_scale=0.5,
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
    ),
]


# ============================================================
# PIPELINE D'ENTRÉE
# ============================================================

def build_input_transform(aug_transform=None):
    """
    Pipeline ResNet50 complète :

        brightness_normalize -> (augmentation) ->
        Resize(224) -> ToTensor -> Normalize(ImageNet)

    `aug_transform` (si fournie) s'applique sur l'image PIL après le
    preprocessing brightness_normalize, avant le redimensionnement.
    """
    transforms_list = [transform_brightness_normalized]

    if aug_transform is not None:
        transforms_list.append(aug_transform)

    transforms_list.extend([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return transforms.Compose(transforms_list)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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
        batch_features.append(input[0].to(device))

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
    shared by every run of ONE transformation / variant.

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
    make_train_dataset: Callable[[int], Dataset],
    val_features_dataset,
    class_names: list,
    num_classes: int,
    dim_in: int,
    device: torch.device,
):
    """Train the classification head once with the given seed.

    `make_train_dataset(epoch)` (1-based) renvoie le dataset d'entraînement
    à utiliser pour l'epoch donnée : features de base avant AUG_START_EPOCH,
    features augmentées ensuite (voir run_experiment).

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

    val_loader = DataLoader(
        val_features_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

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

        train_features_dataset = make_train_dataset(epoch + 1)

        train_loader = DataLoader(
            train_features_dataset,
            shuffle=True,
            generator=generator,
            batch_size=batch_size,
        )

        num_train = len(train_features_dataset)

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
# EXPÉRIENCE : UN GROUPE D'AUGMENTATION + TOUS LES RUNS
# ============================================================

def run_experiment(
    group_name,
    runs,
    split_dfs,
    crop_to_global_idx,
    base_features,
    get_epoch_features,
    out_dir,
    model_dir,
):
    """Entraîne tous les runs pour UN groupe d'augmentation.

    base_features         : features de base (brightness_normalize seul),
                            alignées sur crop_to_global_idx
    get_epoch_features    : callable(epoch_1based) -> Tensor (N, 2048)
                            aligné sur crop_to_global_idx ; précharge à la
                            demande (et met en cache) les features
                            augmentées de l'epoch donnée (>= AUG_START_EPOCH)
    out_dir   : dossier de sortie (sous-dossier de out_dir principal)
    model_dir : dossier où sont sauvegardés les state_dict par run
    """
    metadata_cols = [
        "run",
        "indiv_name",
        "camera",
        "mode",
        "video",
        "entry",
        "split",
        "crop_id",
    ]

    # Lignes de prédiction de la dernière epoch (tous runs confondus)
    all_rows_last_epoch = []
    # {min_acc_improvement: lignes prédites à l'epoch d'arrêt simulé}
    min_acc_rows = {m: [] for m in MIN_ACC_IMPROVEMENTS}

    for run in runs:
        print(f"\n===== [{group_name}] Run {run}/{len(runs)} — {run} =====")

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

        base_train_features = base_features[train_gidx]
        val_features = base_features[val_gidx]
        train_labels = torch.tensor(
            [label_map[name] for name in train_df["indiv_name"]]
        )
        val_labels = torch.tensor(
            [label_map[name] for name in val_df["indiv_name"]]
        )

        # --------------------------------------------------------
        # Dataset d'entraînement PAR EPOCH :
        #   - epochs 1..AUG_START_EPOCH-1 : features de base ;
        #   - epochs AUG_START_EPOCH..    : features augmentées,
        #     re-tirées à chaque epoch si REDRAW_PER_EPOCH.
        # --------------------------------------------------------
        def make_train_dataset(
            epoch_1based,
            _train_gidx=train_gidx,
            _train_labels=train_labels,
            _base_train_features=base_train_features,
        ):
            if epoch_1based < AUG_START_EPOCH:
                train_feats = _base_train_features
            else:
                train_feats = get_epoch_features(epoch_1based)[_train_gidx]
            return TensorDataset(
                train_feats,
                _train_labels,
                torch.arange(len(train_feats)),
            )

        # Les métadonnées (alignées avec val_features / val_labels) servent à
        # construire val_pred.csv ; train_one_run attend un dict en 3e élément.
        val_metadata = val_df[metadata_cols].to_dict("records")
        val_features_dataset = MetadataTensorDataset(
            val_features, val_labels, val_metadata
        )

        dim_in = base_features.shape[1]

        print(
            f"  Classes: {num_classes}, Train: {len(train_gidx)}, "
            f"Val: {len(val_features_dataset)}"
        )

        result = train_one_run(
            SEED,
            make_train_dataset,
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
    set_seed(SEED)

    results_folder = RESULTS_PATH / version / crop_type
    out_dir = results_folder / OUT_DIR_NAME
    os.makedirs(out_dir, exist_ok=True)

    print(f"{crop_type} - {mode}")

    # ============================================================
    # LOAD RUNS (run, entry, split)
    # ============================================================
    split_df = pd.read_csv(SPLIT_FILE).filter(['run','entry','split']).drop_duplicates()
    for col in ("run", "entry", "split"):
        if col not in split_df.columns:
            raise ValueError(
                f"Colonne '{col}' absente de {SPLIT_FILE}"
            )
    split_df = split_df[split_df.run.isin(RUNS)]
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
    summary_df = pd.read_excel(SUMMARY_FILE, engine='odf', sheet_name='clips').filter(['video', 'entry', 'mode', 'camera'])
    crops_df = crops_df.merge(summary_df, on='video')
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
        df = split_df[split_df["run"] == run][["run", "entry", "split"]].merge(
            crops_df,
            on="entry",
            how="inner",
        )
        split_dfs[run] = df
        all_crop_ids.update(df["crop_id"])

    # ============================================================
    # MODÈLE DE BASE (UNE SEULE FOIS)
    # ============================================================
    base_model = models.resnet50(weights='DEFAULT')
    valid_crop_ids = sorted(
        cid for cid in all_crop_ids if (crop_path / cid).exists()
    )
    print(f"\nCrop IDs valides : {len(valid_crop_ids)} / {len(all_crop_ids)}")

    # ============================================================
    # FEATURES DE BASE — brightness_normalize SEUL (une seule fois,
    # partagées par tous les groupes). Réutilise le cache existant.
    # ============================================================
    base_input_transform = build_input_transform()
    base_cache_dir = FEATURES_PATH / version / crop_type / "brightness_normalized"
    base_features, base_crop_ids = compute_or_load_global_features(
        base_model,
        crop_path,
        valid_crop_ids,
        base_cache_dir,
        DEVICE,
        base_input_transform,
    )
    crop_to_global_idx = {cid: i for i, cid in enumerate(base_crop_ids)}
    print(f"  Ordre des features de base : {len(base_crop_ids)} crop ids")

    # ============================================================
    # BOUCLE SUR LES GROUPES D'AUGMENTATION (testés séparément)
    # ============================================================
    for group_idx, (group_name, aug_transform) in enumerate(AUGMENTATION_GROUPS):

        print("\n" + "=" * 70)
        print(f"GROUPE D'AUGMENTATION : {group_name}")
        print("=" * 70)

        if aug_transform is None:
            # Baseline : brightness_normalize seul, pas d'augmentation.
            get_epoch_features = lambda _epoch: base_features
        else:
            aug_input_transform = build_input_transform(aug_transform)

            # Chargeur de features augmentées PAR EPOCH avec cache en
            # mémoire (partagé par les 100 runs d'un même groupe). Cache
            # disque dédié par groupe et par epoch, sous
            # brightness_normalized_aug/<groupe>/epoch_<XX>.
            epoch_cache_root = (
                FEATURES_PATH / version / crop_type /
                "brightness_normalized_aug" / group_name
            )
            epoch_cache = {}

            def get_epoch_features(epoch_1based):
                if epoch_1based < AUG_START_EPOCH:
                    return base_features
                epoch_key = epoch_1based if REDRAW_PER_EPOCH else AUG_START_EPOCH
                if epoch_key not in epoch_cache:
                    cache_dir = epoch_cache_root / f"epoch_{epoch_key:02d}"
                    # Seed dépendant du groupe et de l'epoch : le tirage
                    # d'augmentation est reproductible d'une exécution à
                    # l'autre.
                    set_seed(SEED + 1000 * (group_idx + 1) + epoch_key)
                    print(
                        f"\n[{group_name}] Tirage d'augmentation de l'epoch "
                        f"{epoch_key} ({cache_dir})"
                    )
                    feats, ids = compute_or_load_global_features(
                        base_model,
                        crop_path,
                        base_crop_ids,
                        cache_dir,
                        DEVICE,
                        aug_input_transform,
                    )
                    # Réalignement défensif sur l'ordre des features de base.
                    if list(ids) != list(base_crop_ids):
                        pos = {cid: i for i, cid in enumerate(ids)}
                        reorder = torch.tensor(
                            [pos[cid] for cid in base_crop_ids]
                        )
                        feats = feats[reorder]
                    epoch_cache[epoch_key] = feats
                return epoch_cache[epoch_key]

        # Dossier de sortie dédié à ce groupe (sous-dossier de out_dir)
        # et dossier des model_state_dict dédié.
        exp_out_dir = out_dir / group_name
        exp_model_dir = MODEL_PATH / group_name
        os.makedirs(exp_out_dir, exist_ok=True)
        os.makedirs(exp_model_dir, exist_ok=True)

        # Lancement des 100 runs pour ce groupe
        run_experiment(
            group_name,
            runs,
            split_dfs,
            crop_to_global_idx,
            base_features,
            get_epoch_features,
            exp_out_dir,
            exp_model_dir,
        )

    print(f"\nPour les visualisations et les stats globales, lancer : scripts/analyze_splits.py")


if __name__ == "__main__":
    main()