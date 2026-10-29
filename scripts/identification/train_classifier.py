"""
Entraînement modulaire d'une tête de classification (ClassificationHead) sur
un extracteur de features (MiewID / MegaDescriptor / ResNet152).

PRINCIPES
---------
* La tête de classification est la classe `ClassificationHead` déjà codée dans
  `scripts/network.py` : elle est importée, pas réécrite.
* La loss est une CrossEntropy.
* L'extracteur de features peut être :
    - entièrement gelé (extraction de features à la volée) ;
    - partiellement fine-tuné : les couches comprises ENTRE `last_frozen_layer`
      et `last_layer` sont entraînées à partir de `defreeze_epoch`.
* L'augmentation de données (RandomAffine + ColorJitter) est appliquée À LA
  VOLÉE par le DataLoader à partir de `aug_epoch` : chaque image est
  re-transformée à chaque epoch (comportement PyTorch classique, aucun
  pré-calcul, rien n'est enregistré sur disque).
* Le nombre de runs est celui présent dans le fichier de splits.

USAGE
-----
    python -m scripts.train_classifier -c config/pipeline.yaml \
        --feature-extractor ResNet152 --last-layer fc --last-frozen-layer layer3

ou, depuis `pipeline.py`, via `run_training(args)`.

SORTIES (dans `--out-dir`)
--------------------------
    train_curv_run_<run>.png   courbes loss / accuracy du run
    val_pred.csv               prédictions de validation (dernière epoch)
    val_pred_early_stop.csv    prédictions à l'epoch d'arrêt simulée
    val_pred_run_<run>.csv     prédictions du run (dernière epoch)
    runs_summary.csv           métriques par run (train_acc, val_acc, ...)
"""

from __future__ import annotations

import argparse
import ast
import random
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset, TensorDataset  # noqa: E402
from torchvision import transforms  # noqa: E402

# ============================================================
# BOOTSTRAP : le script doit pouvoir être importé comme
# `scripts.train_classifier` (racine du dépôt dans sys.path)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = Path('/pbs/home/a/ameurisse')

from scripts.network import ClassificationHead  # noqa: E402


# ============================================================
# TABLE DE CORRESPONDANCE  last_layer / last_frozen_layer
# ------------------------------------------------------------
# C'est ICI que l'on remplit la correspondance entre le NOM LOGIQUE
# choisi sur la ligne de commande (--last-layer / --last-frozen-layer)
# et le module RÉEL du modèle.
#
#   "module" : chemin du module dans le modèle ("fc" = model.fc,
#              "head.global_pool" = model.head.global_pool)
#   "take"   : "output" -> on prend la SORTIE du module
#              "input"  -> on prend l'ENTRÉE du module
#                          (= sortie de la couche précédente)
#
# `last_layer` doit désigner la dernière couche AVANT la classification :
#   - "output" : la sortie du module EST le vecteur de features ;
#   - "input"  : l'entrée du module de classification EST le vecteur
#                de features (ex. l'entrée de model.fc pour ResNet).
#
# ATTENTION MiewID : le forward() de MiewIdNet s'arrête AVANT `final`
# (la couche de classification) : `final` n'est donc jamais appelé et ne
# peut pas être capturé par un hook. Il faut utiliser "bn" (take=output).
# ============================================================
# par défault : on prend tjs l'output de la couche
LAYER_MAPPING: dict[str, dict[str, dict[str, str]]] = {
    "ResNet152": {
        "layer1": {"module": "layer1"},
        "layer2": {"module": "layer2"},
        "layer3": {"module": "layer3"}, # de layer3.0.relu layer3.35.relu
        "layer4": {"module": "layer4"}, # de layer4.0.relu à layer4.2.relu -> ajouter avgpool
        "avgpool": {"module": "avgpool"},
    },
    "ResNet50": {   #16 blocks et dans chaque block 3 couches de convolution suivi de batch normalization puis relu à la fin
        "layer1": {"module": "layer1"},
        "layer2": {"module": "layer2"},
        "layer3": {"module": "layer3"},  # de layer3.0.relu layer3.5.relu
        "layer4": {"module": "layer4"},  # de layer4.0.relu à layer4.2.relu -> ajouter avgpool
        "avgpool": {"module": "avgpool"},
    },
    "MegaDescriptor": {
        "norm": {"module": "norm"},
    },
    "MiewID": {                     #0 	3 	6 	15 	30 	60 	90 	120 	150 	180 unfreezing layers
        "bn": {"module": "bn"},     #backbone.blocks.5.9,  backbone.blocks.4.9, backbone.blocks.3.7, backbone.blocks.3.0 -> test simple regarde pas activations au sein des blocks
    },
}
# for name, param in model.named_parameters():
#     param.requires_grad = False
#
# for name, param in model.named_parameters():
#     if name.startswith("backbone.blocks.4.6.") or \
#        name.startswith("backbone.blocks.4.7."):
#         param.requires_grad = True

# ============================================================
# REGISTRE DES EXTRACTEURS DE FEATURES
# ------------------------------------------------------------
# input_size / mean / std : prétraitement attendu par chaque modèle.
# classification_layer : module de classification d'origine (remplacé par
# la ClassificationHead : ses poids ne sont donc jamais entraînés).
# ============================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIEWID_MEAN = (0.5,0.5,0.5)
MIEWID_STD = (0.5,0.5,0.5)

ARCHITECTURES: dict[str, dict] = {
    "ResNet152": {
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "default_last_layer": "avgpool",
        "default_last_frozen_layer": "layer3",
        "default_batch_norm_only": False
    },
    "MegaDescriptor": {
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "default_last_layer": "norm",
        "default_last_frozen_layer": "layers",
        "default_batch_norm_only": False
    },
    "MiewID": {
        "input_size": 224,
        "padding": 108,
        "mean": MIEWID_MEAN,
        "std": MIEWID_STD,
        "default_last_layer": "bn",
        "default_last_frozen_layer": "backbone",
        "default_batch_norm_only": False
    },
}

FEATURE_EXTRACTORS = list(ARCHITECTURES.keys())


# ============================================================
# OUTILS GÉNÉRIQUES
# ============================================================
def set_seed(seed: int) -> None:
    """Fixe toutes les graines (torch, numpy, random)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ============================================================
# PRETRAITEMENT / AUGMENTATION
# ============================================================
LUMINANCE = np.array([0.299, 0.587, 0.114])


def transform_brightness_normalized(image: Image.Image, target_mean: float = 0.5):
    """Normalise la luminosité moyenne de l'image (Rec. 601)."""
    array = np.asarray(image).astype(np.float32) / 255.0

    luminance = array @ LUMINANCE
    current_mean = luminance.mean()

    if current_mean == 0:
        return image

    factor = target_mean / current_mean
    normalized = np.clip(array * factor, 0, 1)

    return Image.fromarray((normalized * 255).astype(np.uint8))


# Prétraitements disponibles (nom -> fonction). `None` = pas de prétraitement.
PREPROCESSING = {
    "brightness_normalized": transform_brightness_normalized,
    "none": None,
    None: None,
}


def _parse_sequence(value) -> list | None:
    """Normalise un paramètre YAML en liste de nombres.

    Le YAML ne reconnaît pas les tuples : `(0.1, 0, 1)` est chargé comme
    une CHAINE. On accepte donc indifféremment une liste, un tuple, un
    nombre ou une chaîne du type "(0.1, 0, 1)" / "[0.9, 1.10]".
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = [
                part for part in text.replace(",", " ").split() if part
            ]
        if isinstance(parsed, (int, float)):
            return [parsed]
        try:
            return [float(part) for part in parsed]
        except (TypeError, ValueError):
            return None
    try:
        return list(value)
    except TypeError:
        return None


def _normalize_translate(translate) -> tuple | None:
    """Normalise le paramètre `translate` de RandomAffine.

    Accepte :
        - None                      -> None
        - (tx, ty)                 -> (tx, ty)
        - (tx_min, tx_max, ty_min, ty_max) -> tel quel
    Un triplet est interprété comme (tx, ty) (la 3e valeur est ignorée, un
    avertissement est affiché).
    """
    values = _parse_sequence(translate)
    if values is None:
        return None
    if len(values) == 2:
        return (values[0], values[1])
    if len(values) == 4:
        return tuple(values)
    raise ValueError(f"translate invalide : {translate}")


def _normalize_scale(scale) -> tuple | float | None:
    """Normalise le paramètre `scale` de RandomAffine."""
    values = _parse_sequence(scale)
    if values is None:
        return None
    if len(values) == 2:
        return (values[0], values[1])
    if len(values) == 1:
        return float(values[0])
    raise ValueError(f"scale invalide : {scale}")


def _normalize_jitter_value(value):
    """Normalise un paramètre de ColorJitter (nombre ou intervalle)."""
    values = _parse_sequence(value)
    if values is None:
        return None
    if len(values) == 1:
        return float(values[0])
    if len(values) == 2:
        return (values[0], values[1])
    raise ValueError(f"paramètre ColorJitter invalide : {value}")


def build_augmentation(params: dict | None) -> transforms.Compose | None:
    """Construit l'augmentation RandomAffine + ColorJitter depuis le YAML.

    `params` = contenu de `data_augmentation.data_transformation.params`.
    """
    if not params:
        return None

    operations = []

    affine_params = params.get("affine") or {}
    if affine_params:
        kwargs = {}
        if affine_params.get("degrees") is not None:
            kwargs["degrees"] = affine_params["degrees"]
        translate = _normalize_translate(affine_params.get("translate"))
        if translate is not None:
            kwargs["translate"] = translate
        scale = _normalize_scale(affine_params.get("scale"))
        if scale is not None:
            kwargs["scale"] = scale
        if affine_params.get("shear") is not None:
            kwargs["shear"] = affine_params["shear"]
        operations.append(transforms.RandomAffine(**kwargs))

    jitter_params = (
        params.get("colorjittering")
        or params.get("color_jitter")
        or {}
    )
    if jitter_params:
        kwargs = {}
        for key in ("brightness", "contrast", "saturation", "hue"):
            if jitter_params.get(key) is None:
                continue
            normalized = _normalize_jitter_value(jitter_params[key])
            if normalized is not None:
                kwargs[key] = normalized
        if kwargs:
            operations.append(transforms.ColorJitter(**kwargs))

    if not operations:
        return None

    return transforms.Compose(operations)


def build_input_transform(
    preprocessing_name: str | None,
    preprocessing_params: dict | None,
    augmentation: transforms.Compose | None,
    input_size: int,
    mean: tuple,
    std: tuple,
    padding: int | None,
) -> transforms.Compose:
    """Pipeline image : prétraitement -> augmentation -> resize -> tensor.

    L'augmentation est appliquée AVANT le redimensionnement (sur l'image PIL).
    """
    operations = []

    preprocessing = PREPROCESSING.get(preprocessing_name, None)
    if preprocessing_name is not None and preprocessing is None:
        warnings.warn(
            f"Prétraitement inconnu : {preprocessing_name} (ignoré)."
        )
    if preprocessing is not None:
        params = preprocessing_params or {}
        operations.append(
            lambda img, fn=preprocessing, kw=params: fn(img, **kw)
        )

    if augmentation is not None:
        operations.append(augmentation)

    operations.append(
        transforms.Resize([input_size, input_size]))
    if padding is not None:
        operations.append(transforms.Pad(padding,fill=0))
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])

    return transforms.Compose(operations)


# ============================================================
# DATASETS
# ============================================================
class ImageCropDataset(Dataset):
    """Dataset d'entraînement sur IMAGES BRUTES.

    La transform (qui inclut l'augmentation aléatoire du groupe) est
    appliquée à CHAQUE appel de ``__getitem__`` : chaque image est donc
    re-transformée à chaque epoch, exactement comme dans un entraînement
    classique PyTorch. Aucune image transformée n'est enregistrée.

    ``needs_backbone = True`` indique à la boucle d'entraînement que les
    éléments produits sont des images : il faut un forward de l'extracteur
    de features avant la tête de classification.

    Si ``metadata`` est fourni (liste de dicts), l'item retourné est
    ``(image, label, metadata)`` : c'est utilisé pour le jeu de validation,
    afin de pouvoir écrire `val_pred.csv`.
    """

    needs_backbone = True

    def __init__(
        self,
        crop_path: Path,
        crop_ids: list,
        labels,
        transform,
        metadata: list | None = None,
    ):
        self.crop_path = Path(crop_path)
        self.crop_ids = crop_ids
        self.labels = labels
        self.transform = transform
        self.metadata = metadata

    def __len__(self):
        return len(self.crop_ids)

    def __getitem__(self, idx):
        image = Image.open(
            self.crop_path / self.crop_ids[idx]
        ).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.metadata is not None:
            return image, self.labels[idx], self.metadata[idx]
        return image, self.labels[idx]


# ============================================================
# BACKBONE : construction, extraction des features, gel/dégel
# ============================================================
@dataclass
class Backbone:
    """Extracteur de features + métadonnées d'architecture."""

    name: str
    model: nn.Module
    input_size: int
    padding: int
    mean: tuple
    std: tuple
    classification_layer: str
    children: list


def build_backbone(name: str, pretrained: bool = True) -> Backbone:
    """Construit l'extracteur de features demandé."""
    if name not in ARCHITECTURES:
        raise ValueError(
            f"Extracteur inconnu : {name} "
            f"(choix : {', '.join(FEATURE_EXTRACTORS)})"
        )

    spec = ARCHITECTURES[name]

    if name == "ResNet152":
        from torchvision import models
        weights = "DEFAULT" if pretrained else None
        model = models.resnet152(weights=weights)
    elif name == "ResNet50":
        from torchvision import models
        weights = "DEFAULT" if pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "MegaDescriptor":
        import timm
        model = timm.create_model(
            "hf-hub:BVRA/MegaDescriptor-T-224",
            pretrained=pretrained,
        )

    elif name == "MiewID":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "conservationxlabs/miewid-msv3",
            trust_remote_code=True,
        )
        if not pretrained:
            warnings.warn(
                "MiewID : le code distant charge toujours le backbone "
                "pré-entraîné (pretrained=False n'est pas supporté)."
            )

    else:  # pragma: no cover - Impossible d'atteindre ce cas
        raise ValueError(name)

    model.eval()

    return Backbone(
        name=name,
        model=model,
        input_size=spec["input_size"],
        padding=spec.get("padding", None),
        mean=spec["mean"],
        std=spec["std"],
        classification_layer=spec["classification_layer"],
        children=list(model.named_children()),
    )


def _is_after(key: tuple, frozen_key: tuple) -> bool:
    """Vrai si le module de rang `key` est situé APRÈS le module gelé.

    Les descendants du module gelé (par exemple les bottlenecks de "layer3")
    ne sont PAS considérés comme postérieurs : ils restent gelés.
    """
    if key <= frozen_key:
        return False
    return key[: len(frozen_key)] != frozen_key


def resolve_module(model: nn.Module, path: str) -> nn.Module:
    """Résout un chemin de module ("head.global_pool" -> model.head.global_pool)."""
    module = model
    for part in path.split("."):
        if not part:
            continue
        if not hasattr(module, part):
            raise AttributeError(
                f"Le module '{part}' n'existe pas dans "
                f"{type(model).__name__} (chemin : {path})."
            )
        module = getattr(module, part)
    return module


def extract_features(
    model: nn.Module,
    x: torch.Tensor,
    module_path: str,
) -> torch.Tensor:
    """Retourne la sortie d'un module donné du modèle.

    Un hook est posé sur le module pour récupérer sa sortie
    pendant le forward complet du modèle. Le tenseur retourné fait partie du
    graphe : les gradients peuvent donc remonter jusqu'aux couches
    fine-tunées.
    """
    module = resolve_module(model, module_path)
    captured: dict = {}
    def _hook(_module, _args, output):
        captured["value"] = output

    handle = module.register_forward_hook(_hook)

    try:
        model(x)
    finally:
        handle.remove()

    if "value" not in captured or captured["value"] is None:
        raise RuntimeError(
            f"Le module '{module_path}' n'a pas été appelé pendant le "
            f"forward du modèle : impossible d'en extraire les features "
        )

    return captured["value"]


class FeatureModel(nn.Module):
    """Enveloppe l'extracteur de features.

    * `forward(x)` renvoie le vecteur de features (sortie de `last_layer`).
    * `set_trainable(flag)` active / désactive le fine-tuning des couches
      comprises entre `last_frozen_layer` et `last_layer`.
    * `set_mode(training)` met le modèle en mode train (BatchNorm) pour les
      seules couches fine-tunées, et en mode eval pour les autres.
    """

    def __init__(
        self,
        backbone: Backbone,
        last_layer: str,
        last_frozen_layer: str | None = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.model = backbone.model
        self.last_layer = last_layer
        self.last_frozen_layer = last_frozen_layer
        self.batch_norm_only = True
        valid_layers = [n for n,_ in self.model.named_parameters()]
        if last_layer not in valid_layers:
            raise KeyError(
                f"last_layer '{last_layer}' inconnu pour "
            )
        if last_frozen_layer is not None and last_frozen_layer not in valid_layers:
            raise KeyError(
                f"last_frozen_layer '{last_frozen_layer}' inconnu pour "
            )

        self.module_path = self.last_layer

        # --------------------------------------------------
        # Couches fine-tunables : entre last_frozen_layer et
        # last_layer (bornes comprises), la couche de classification
        # d'origine n'est jamais entraînée (remplacée par la tête).
        # --------------------------------------------------
        self._trainable_modules = self._compute_trainable_modules()
        self._candidate_parameters = []

        for name, module in self.model.named_modules():
            if name == "":
                continue

            for parameter in module.parameters(recurse=False):
                self._candidate_parameters.append(parameter)

            if name == self.last_layer:
                break

        # Tout est gelé par défaut : le dégel est piloté par `set_trainable`
        # (epoch `defreeze_epoch`).
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._feature_dim = self._compute_feature_dim()

    # ------------------------------------------------------------
    def _module_order_keys(self) -> dict:
        """Associe à chaque module son "rang" dans l'ordre d'exécution.

        Le rang est le tuple des INDICES d'enfants depuis la racine :
        ("conv1",) -> (0,), ("layer4",) -> (7,), ("layer4.2.conv1",) -> (7, 2, 0).
        La comparaison de deux rangs donne l'ordre d'exécution.
        """
        keys = {}

        def walk(name_prefix, index_prefix, module):
            for index, (name, child) in enumerate(module.named_children()):
                path = name_prefix + (name,)
                keys[".".join(path)] = index_prefix + (index,)
                walk(path, index_prefix + (index,), child)

        walk((), (), self.model)
        return keys

    def _order_key(self, layer_name: str) -> tuple:
        keys = self._module_order_keys()
        return keys[layer_name]

    def _compute_trainable_modules(self) -> list:
        """Modules à fine-tuner (last_frozen_layer exclus, last_layer_inclus).

        Le fine-tuning porte sur tous les modules situés APRÈS
        `last_frozen_layer` et JUSQU'À `last_layer` (inclus).
        """
        keys = self._module_order_keys()
        key_last = self._order_key(self.last_layer)

        if self.last_frozen_layer is None:
            key_frozen = (-1,)
        else:
            key_frozen = self._order_key(self.last_frozen_layer)
            if key_last <= key_frozen:
                raise ValueError(
                    f"last_layer ('{self.last_layer}') doit être situé APRÈS "
                    f"last_frozen_layer ('{self.last_frozen_layer}')."
                )

        in_range = {
            path: key
            for path, key in keys.items()
            if _is_after(key, key_frozen) and key <= key_last
        }

        selected = []
        for path, key in in_range.items():
            parent = ".".join(path.split(".")[:-1]) #layer parent de path
            if parent and parent in in_range: #si parent est dans in_range pas besoin de l'ajouter car le parent couvre les enfants
                continue
            selected.append((path, key))

        selected.sort(key=lambda item: item[1]) #trie par ordre croissant

        return [
            resolve_module(self.model, path) for path, _ in selected
        ]

    def _compute_feature_dim(self) -> int:
        dummy = torch.zeros(
            1, 3, self.backbone.input_size, self.backbone.input_size
        )
        with torch.no_grad():
            features = extract_features(
                self.model, dummy, self.module_path
            )
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return int(features.shape[1])

    # ------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def trainable_parameters(self) -> list:
        """Paramètres potentiellement fine-tunés (selon l'epoch)."""
        return self._candidate_parameters

    def set_trainable(self, trainable: bool) -> None:
        """Active / désactive le fine-tuning du backbone."""
        for parameter in self._candidate_parameters:
            parameter.requires_grad_(trainable)

    def set_mode(self, training: bool) -> None:
        """Mode train/eval : seules les couches fine-tunées passent en
        mode train (BatchNorm), les autres restent en mode eval.

        Tant que le backbone est gelé (avant `defreeze_epoch`), les
        statistiques des BatchNorm restent figées : les features ne
        changent pas d'une epoch à l'autre.
        """
        self.model.eval()
        if training:
            for module in self._trainable_modules:
                if any(p.requires_grad for p in module.parameters()):
                    module.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #check si besoin de grad er flatten pour avant la classification Head -> avgpool?
        if any(p.requires_grad for p in self._candidate_parameters):
            features = extract_features(
                self.model, x, self.module_path
            )
        else:
            with torch.no_grad():
                features = extract_features(
                    self.model, x, self.module_path
                )
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return features


# ============================================================
# ENTRAÎNEMENT
# ============================================================
def find_early_stop_epoch(
    train_accuracies: list,
    min_acc_improvement: float,
) -> int:
    """Indice de l'epoch (0-based) où l'early stopping se serait déclenché.

    On se serait arrêté à la première epoch `e` (e >= 1) dont l'amélioration
    d'accuracy train par rapport à l'epoch précédente est strictement
    inférieure à `min_acc_improvement` %. Si aucun seuil ne déclenche
    l'arrêt, on retourne l'indice de la dernière epoch.
    Pour éviter le surentrainement sur la validation en l'absence de jeu de test
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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


def compute_features(
    backbone: FeatureModel,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> torch.Tensor:
    """Calcule les features d'un dataset (images -> vecteurs).

    Le backbone est mis en mode eval et aucun gradient n'est calculé :
    les features sont donc déterministes et réutilisables.
    """
    backbone.set_mode(False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    features = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device, non_blocking=True)
            features.append(backbone(batch).cpu())

    if not features:
        return torch.zeros(0)

    return torch.cat(features, dim=0)


def train_one_run(
    seed: int,
    make_train_loader: Callable[[int, bool], DataLoader],
    backbone: FeatureModel,
    val_dataset,
    class_names: list,
    num_classes: int,
    dim_in: int,
    device: torch.device,
    num_epoch: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    defreeze_epoch: int | None = None,
):
    """Entraîne la tête de classification (et éventuellement le backbone).

    `make_train_loader(epoch, frozen)` (epoch 1-based) renvoie un DataLoader
    pour l'epoch donnée. Son dataset produit des couples (tensor, label) :
        - features pré-calculées (TensorDataset) quand le backbone est gelé
          ET l'augmentation n'est pas encore active : elles sont calculées
          une seule fois (le backbone étant figé, elles restent valides) ;
        - images (ImageCropDataset, ``needs_backbone=True``) sinon : le
          backbone calcule les features de chaque batch à la volée, ce qui est
          indispensable dès que l'augmentation est active ou que le backbone
          est fine-tuné (sinon les features seraient obsolètes).

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

    # Le backbone est toujours dans l'optimizer : les couches fine-tunées
    # sont simplement inactives (requires_grad=False) tant que l'epoch de
    # dégel n'est pas atteinte.
    parameters = list(head.parameters()) + list(backbone.trainable_parameters)

    optimizer = torch.optim.Adam(
        parameters,
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    num_val = len(val_dataset)

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

        # Dégel des couches fine-tunées à partir de defreeze_epoch
        # (1-based). `None` => fine-tuning dès la première epoch.
        if defreeze_epoch is None:
            backbone.set_trainable(True)
        else:
            backbone.set_trainable((epoch + 1) >= defreeze_epoch)

        # Backbone figé => les features ne changent pas d'une epoch à l'autre.
        frozen = defreeze_epoch is not None and (epoch + 1) < defreeze_epoch

        train_loader = make_train_loader(epoch + 1, frozen)
        needs_backbone = getattr(
            train_loader.dataset, "needs_backbone", False
        )

        num_train = len(train_loader.dataset)

        backbone.set_mode(True)
        head.train()

        error = 0.0
        corrects = 0

        for x, label in train_loader:

            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if needs_backbone:
                # Images augmentées à la volée : features calculées sur le
                # batch courant, rien n'est pré-calculé ni enregistré.
                x = backbone(x)

            output = head(x)

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

        error /= max(len(train_loader), 1)

        train_error.append(error)

        train_accuracy.append(
            corrects * 100 / num_train
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        backbone.set_mode(False)
        head.eval()

        error = 0.0
        corrects = 0

        val_prediction_rows = []

        val_needs_backbone = getattr(
            val_dataset, "needs_backbone", False
        )

        with torch.no_grad():

            for feat, label, metadata in val_loader:

                feat = feat.to(device)
                label = label.to(device)

                if val_needs_backbone:
                    feat = backbone(feat)

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

        error /= max(len(val_loader), 1)

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
        "model_dict": head.state_dict(),
        "backbone_state_dict": backbone.model.state_dict(),
    }


# ============================================================
# PRÉPARATION DES DONNÉES (runs)
# ============================================================
def load_run_tables(
    crop_file: Path,
    split_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Charge les tables de crops et de splits.

    Retourne (crops_df, splits_df, runs).
    """
    crops_df = pd.read_csv(crop_file)
    splits_df = pd.read_csv(split_file)

    for column in ("image", "label"):
        if column not in crops_df.columns:
            raise ValueError(
                f"La table des crops ({crop_file}) ne contient pas de "
                f"colonne '{column}'."
            )

    crops_df = crops_df.dropna(subset=["image", "label"])
    crops_df["image"] = crops_df["image"].astype(str)

    runs = sorted(splits_df["run"].dropna().unique().tolist())
    if not runs:
        raise ValueError(
            f"Aucun run trouvé dans {split_file}."
        )

    return crops_df, splits_df, runs


def build_run_dataframe(
    crops_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    run,
) -> pd.DataFrame:
    """Assemble les crops d'un run avec leur label train/val."""
    run_splits = splits_df[splits_df["run"] == run]

    merged = crops_df.merge(
                run_splits[["event", "split"]],
                on=["event"],
                how="inner",
            )

    merged = merged[merged["split"].isin(["train", "val"])]

    if merged.empty:
        raise ValueError(f"Aucun crop pour le run {run}.")

    return merged.reset_index(drop=True)


# ============================================================
# BOUCLE PRINCIPALE SUR LES RUNS
# ============================================================
def run_training(args: argparse.Namespace) -> pd.DataFrame:
    """Entraîne la tête pour tous les runs du fichier de splits.

    Retourne le tableau récapitulatif des runs (une ligne par run).
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir) if args.model_dir else out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    set_seed(args.seed)

    # ------------------------------------------------------------
    # TABLES
    # ------------------------------------------------------------
    crops_df, splits_df, runs = load_run_tables(
        Path(args.crop_file), Path(args.split_file)
    )

    if args.limit_runs is not None:
        runs = runs[: args.limit_runs]

    print(f"Extracteur de features : {args.feature_extractor}")
    print(f"last_layer            : {args.last_layer}")
    print(f"last_frozen_layer     : {args.last_frozen_layer}")
    print(f"Run(s) dans le split  : {len(runs)}")
    print(f"Device                : {device}")

    # ------------------------------------------------------------
    # BACKBONE
    # ------------------------------------------------------------
    backbone = build_backbone(
        args.feature_extractor, pretrained=args.pretrained
    )

    feature_model = FeatureModel(
        backbone,
        last_layer=args.last_layer,
        last_frozen_layer=args.last_frozen_layer,
        batch_norm_only=args.batch_norm_only
    ).to(device)

    dim_in = feature_model.feature_dim
    print(f"Dimension des features: {dim_in}")

    # ------------------------------------------------------------
    # TRANSFORMATIONS
    # ------------------------------------------------------------
    augmentation = build_augmentation(
        args.augmentation_params
    )

    base_transform = build_input_transform(
        args.transform_function,
        args.transform_params,
        None,
        backbone.input_size,
        backbone.mean,
        backbone.std,
        backbone.padding
    )

    augmented_transform = build_input_transform(
        args.transform_function,
        args.transform_params,
        augmentation,
        backbone.input_size,
        backbone.mean,
        backbone.std,
        backbone.padding
    )

    if augmentation is None:
        print("Aucune augmentation configurée.")
    else:
        print(f"Augmentation (epoch >= {args.aug_epoch}) : {augmentation}")

    crop_path = Path(args.crop_path)

    # ------------------------------------------------------------
    # RUNS
    # ------------------------------------------------------------
    all_rows_last_epoch: list[dict] = []
    all_rows_early_stop: list[dict] = []
    summary_rows: list[dict] = []

    for run in runs:
        print(f"\n===== Run {run} / {len(runs)} =====")

        started = time.time()

        df = build_run_dataframe(crops_df, splits_df, run)

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)

        if train_df.empty or val_df.empty:
            raise ValueError(
                f"Run {run} : split train/val vide "
                f"(train={len(train_df)}, val={len(val_df)})."
            )

        class_names = sorted(df["label"].unique().tolist())
        label_map = {name: idx for idx, name in enumerate(class_names)}
        num_classes = len(class_names)

        train_crop_ids = list(train_df["image"])
        val_crop_ids = list(val_df["image"])

        train_labels = torch.tensor(
            [label_map[name] for name in train_df["label"]]
        )
        val_labels = torch.tensor(
            [label_map[name] for name in val_df["label"]]
        )

        # Métadonnées de validation (image, video, label, ...)
        metadata_columns = [
            c for c in ("image", "video", "label", "event", "split")
            if c in val_df.columns
        ]
        val_metadata = val_df[metadata_columns].to_dict("records")

        print(
            f"  Classes: {num_classes}, Train: {len(train_df)}, "
            f"Val: {len(val_df)}"
        )

        # --------------------------------------------------------
        # Loader d'entraînement PAR EPOCH :
        #   - epochs < aug_epoch : images sans augmentation ;
        #   - epochs >= aug_epoch : images RE-TRANSFORMÉES à chaque
        #     epoch par le DataLoader (augmentation à la volée).
        # --------------------------------------------------------
        shuffle_gen = torch.Generator().manual_seed(args.seed)

        # Cache des features "de base" : calculé UNE seule fois, uniquement
        # utilisé tant que le backbone est gelé (features figées) et que
        # l'augmentation n'est pas active.
        base_features_cache: dict = {}

        def get_base_features() -> torch.Tensor:
            if "features" not in base_features_cache:
                dataset = ImageCropDataset(
                    crop_path,
                    train_crop_ids,
                    train_labels,
                    base_transform,
                )
                base_features_cache["features"] = compute_features(
                    feature_model,
                    dataset,
                    device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            return base_features_cache["features"]

        def make_train_loader(epoch_1based: int, frozen: bool) -> DataLoader:
            use_augmentation = (
                augmentation is not None
                and epoch_1based >= args.aug_epoch
            )

            if not use_augmentation and frozen:
                # Features pré-calculées (le backbone est figé).
                dataset = TensorDataset(
                    get_base_features(),
                    train_labels,
                )
            else:
                transform = (
                    augmented_transform
                    if use_augmentation
                    else base_transform
                )
                dataset = ImageCropDataset(
                    crop_path,
                    train_crop_ids,
                    train_labels,
                    transform,
                )

            return DataLoader(
                dataset,
                shuffle=True,
                generator=shuffle_gen,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )

        val_dataset = ImageCropDataset(
            crop_path,
            val_crop_ids,
            val_labels,
            base_transform,
            metadata=val_metadata,
        )

        result = train_one_run(
            args.seed,
            make_train_loader,
            feature_model,
            val_dataset,
            class_names,
            num_classes,
            dim_in,
            device,
            num_epoch=args.num_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            defreeze_epoch=args.defreeze_epoch,
        )

        train_error = result["train_error"]
        val_error = result["val_error"]
        train_accuracy = result["train_accuracy"]
        val_accuracy = result["val_accuracy"]
        epoch_val_predictions = result["epoch_val_predictions"]

        # --------------------------------------------------------
        # COURBES D'ENTRAÎNEMENT
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # POIDS DES MODÈLES
        # --------------------------------------------------------
        model_path = model_dir / f"model_run_{run}.pth"
        torch.save(
            {
                "backbone_state_dict": result["backbone_state_dict"],
                "head_state_dict": result["model_dict"],
                "class_names": class_names,
                "feature_extractor": args.feature_extractor,
                "last_layer": args.last_layer,
                "last_frozen_layer": args.last_frozen_layer,
                "num_epoch": args.num_epoch,
                "val_accuracy": val_accuracy[-1],
                "train_accuracy": train_accuracy[-1],
            },
            model_path,
        )
        print(f"  Poids sauvegardés : {model_path}")

        # --------------------------------------------------------
        # PRÉDICTIONS DE VALIDATION
        # --------------------------------------------------------
        last_epoch = len(epoch_val_predictions) - 1
        last_rows = epoch_val_predictions[last_epoch]

        stop_epoch = find_early_stop_epoch(
            train_accuracy, args.early_stopping
        )
        stop_rows = epoch_val_predictions[stop_epoch]

        for row in last_rows:
            row["run"] = run
        for row in stop_rows:
            row["run"] = run

        all_rows_last_epoch.extend(last_rows)
        all_rows_early_stop.extend(stop_rows)

        # Fichier par run
        run_pred_df = build_val_pred_df(last_rows)
        run_pred_path = out_dir / f"val_pred_run_{run}.csv"
        run_pred_df.to_csv(run_pred_path, index=False)

        best_epoch = int(
            np.argmax(val_accuracy)
        )

        summary_rows.append(
            {
                "run": run,
                "num_classes": num_classes,
                "num_train": len(train_df),
                "num_val": len(val_df),
                "train_acc": round(train_accuracy[stop_epoch], 2),
                "val_acc": round(val_accuracy[stop_epoch], 2),
                "best_val_acc": round(val_accuracy[best_epoch], 2),
                "best_epoch": best_epoch + 1,
                "train_loss": round(train_error[stop_epoch], 4),
                "val_loss": round(val_error[stop_epoch], 4),
                "early_stop_epoch": stop_epoch + 1,
                "duration_s": round(time.time() - started, 1),
            }
        )

        print(
            f"  Train acc : {train_accuracy[-1]:.2f}% | "
            f"Val acc : {val_accuracy[-1]:.2f}% | "
            f"arrêt simulé epoch {stop_epoch + 1}"
        )

    # ============================================================
    # SAUVEGARDE FINALE
    # ============================================================
    val_pred_df = build_val_pred_df(all_rows_last_epoch)
    val_pred_path = out_dir / "val_pred.csv"
    val_pred_df.to_csv(val_pred_path, index=False)
    print(f"\nPrédictions de validation : {val_pred_path}")
    print(f"  {len(val_pred_df)} lignes")

    early_stop_df = build_val_pred_df(all_rows_early_stop)
    early_stop_path = out_dir / "val_pred_early_stop.csv"
    early_stop_df.to_csv(early_stop_path, index=False)
    print(f"Prédictions (arrêt simulé) : {early_stop_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "runs_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Résumé des runs : {summary_path}")

    return summary_df


# Colonnes de val_pred.csv (dans l'ordre demandé).
VAL_PRED_COLUMNS = [
    "run",
    "image",
    "video",
    "label",
    "pred_label",
    "margin",
    "logits",
    "conf",
    "indiv_name",
]


def build_val_pred_df(rows: list[dict]) -> pd.DataFrame:
    """Construit le DataFrame val_pred à partir des lignes de prédiction.

    Colonnes : run, image, video, label, pred_label, margin, logits, conf
    (+ `indiv_name`, alias de `label`, conservé pour la compatibilité avec
    les scripts d'analyse existants).
    """
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=VAL_PRED_COLUMNS)

    if "conf" not in df.columns and "conf_pred_label" in df.columns:
        df["conf"] = df["conf_pred_label"]

    if "indiv_name" not in df.columns and "label" in df.columns:
        df["indiv_name"] = df["label"]

    for column in VAL_PRED_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[VAL_PRED_COLUMNS].copy()

    for column in ("margin", "conf"):
        df[column] = pd.to_numeric(df[column], errors="coerce").round(4)

    return df


# ============================================================
# CONFIGURATION
# ============================================================
def load_pipeline_config(path: Path | None) -> dict:
    """Charge le fichier de configuration du pipeline (YAML)."""
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


# ============================================================
# CHEMINS DE LA CONFIGURATION
# ============================================================
def format_template(template: str, values: dict) -> str:
    """Remplace les {placeholders} d'un chemin de la configuration."""
    return template.format_map(defaultdict(str, values))


def build_path_values(config: dict) -> dict:
    """Valeurs utilisables dans les chemins de la configuration.

    La date est "assainie" (les "/" sont remplacés par des "_") afin de
    pouvoir servir de nom de fichier.
    """
    split = config.get("split", {}) or {}
    return {
        "dataset_date": str(config.get("dataset_date", "")).replace("/", "_"),
        "detection_model": str(config.get("detection_model", "")),
        "split_method": str(split.get("split_method", "")).replace("/", "_"),
        "tracker": str(config.get("tracker", "")),
        "home_path": str(config.get("home_path", "")),
    }


def resolve_path(home_path: Path, template: str, values: dict) -> Path:
    """Résout un chemin de la configuration par rapport à `home_path`."""
    rendered = format_template(template, values)
    path = Path(rendered)
    if path.is_absolute():
        return path
    return home_path / path


def resolve_config_paths(config: dict) -> dict:
    """Résout les chemins `paths.*` de la configuration.

    Retourne un dict avec les clés crop_path, crop_file, split_file et
    model_dir.
    """
    paths = config.get("paths", {}) or {}
    values = build_path_values(config)
    home_path = Path(config.get("home_path", "..")).expanduser()

    return {
        "crop_path": resolve_path(
            home_path, paths.get("crop_path", "data/crops"), values
        ),
        "crop_file": resolve_path(
            home_path, paths.get("crop_file", "data/crops.csv"), values
        ),
        "split_file": resolve_path(
            home_path, paths.get("split_file", "data/splits.csv"), values
        ),
        "model_dir": resolve_path(
            home_path, paths.get("model_dir", "models"), values
        ),
    }


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Complète les arguments non fournis par la ligne de commande avec
    les valeurs du fichier de configuration du pipeline."""
    config = load_pipeline_config(
        Path(args.config) if args.config else None
    )
    args.config_dict = config

    training = config.get("training_parameters", {}) or {}
    augmentation = config.get("data_augmentation", {}) or {}
    transformation = (
        augmentation.get("data_transformation", {}) or {}
    )
    extractor_params = config.get("feature_extractor_params", {}) or {}

    if args.feature_extractor is None:
        args.feature_extractor = config.get(
            "feature_extractor", "ResNet152"
        )
    if args.last_layer is None:
        args.last_layer = extractor_params.get("last_layer")
    if args.last_frozen_layer is None:
        args.last_frozen_layer = extractor_params.get("last_frozen_layer")
    if args.batch_size is None:
        args.batch_size = training.get("batch_size", 64)
    if args.num_epoch is None:
        args.num_epoch = training.get("num_epoch", 30)
    if args.lr is None:
        args.lr = training.get("lr", 1e-3)
    if args.aug_epoch is None:
        args.aug_epoch = training.get("aug_epoch", 1)
    if args.defreeze_epoch is None:
        args.defreeze_epoch = training.get("defreeze_epoch")
    if args.early_stopping is None:
        args.early_stopping = training.get("early_stopping", 1.0)
    if args.num_workers is None:
        args.num_workers = training.get("num_workers", 4)
    if args.seed is None:
        args.seed = training.get("seed", 42)
    if args.pretrained is None:
        args.pretrained = training.get("pretrained", True)
    if args.device is None:
        args.device = training.get("device")

    if getattr(args, "augmentation_params", None) is None:
        args.augmentation_params = transformation.get("params", {}) or {}
    if getattr(args, "transform_function", None) is None:
        args.transform_function = config.get("transform_function")
    if getattr(args, "transform_params", None) is None:
        args.transform_params = config.get("params", {}) or {}

    # Chemins : la ligne de commande prime, sinon la configuration.
    config_paths = resolve_config_paths(config)
    for name in ("crop_path", "crop_file", "split_file", "model_dir"):
        if getattr(args, name, None) is None and name in config_paths:
            setattr(args, name, str(config_paths[name]))

    # Valeurs par défaut de l'architecture si le YAML ne précise rien
    architecture = ARCHITECTURES.get(args.feature_extractor, {})
    if args.last_layer is None:
        args.last_layer = architecture.get("default_last_layer")
    if args.last_frozen_layer is None:
        args.last_frozen_layer = architecture.get(
            "default_last_frozen_layer"
        )
    if args.batch_norm_only is None:
        args.last_frozen_layer = architecture.get(
            "default_batch_norm_only"
        )

    if isinstance(args.last_frozen_layer, str):
        if args.last_frozen_layer.lower() in ("none", "null", ""):
            args.last_frozen_layer = None

    if args.feature_extractor not in ARCHITECTURES:
        raise ValueError(
            f"Extracteur inconnu : {args.feature_extractor} "
            f"(choix : {', '.join(FEATURE_EXTRACTORS)})"
        )

    return args


# ============================================================
# ARGPARSE
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entraînement modulaire d'une tête de classification sur un "
            "extracteur de features (MiewID, MegaDescriptor, ResNet152)."
        )
    )

    parser.add_argument(
        "--config", "-c", default=None,
        help="fichier de configuration du pipeline (YAML)",
    )
    parser.add_argument(
        "--feature-extractor", "-f", default=None,
        choices=FEATURE_EXTRACTORS,
        help="extracteur de features à utiliser",
    )
    parser.add_argument(
        "--last-layer", default=None,
        help=(
            "nom logique de la dernière couche avant la classification "
            "(voir LAYER_MAPPING dans ce fichier)"
        ),
    )
    parser.add_argument(
        "--last-frozen-layer", default=None,
        help=(
            "nom logique de la dernière couche gelée : les couches "
            "suivantes (jusqu'à last_layer) sont fine-tunées. "
            "'none' = fine-tuning complet"
        ),
    )

    parser.add_argument("--crop-path", default=None, help="dossier des crops")
    parser.add_argument(
        "--crop-file", default=None,
        help="table des crops (csv/ods) : image, video, label",
    )
    parser.add_argument(
        "--split-file", default=None,
        help="table des splits (csv/ods) : run, split",
    )
    parser.add_argument(
        "--out-dir", default=None, help="dossier de résultats"
    )
    parser.add_argument(
        "--model-dir", default=None, help="dossier des poids des modèles"
    )

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--aug-epoch", type=int, default=None,
        help="epoch (1-based) à partir de laquelle l'augmentation est appliquée",
    )
    parser.add_argument(
        "--defreeze-epoch", type=int, default=None,
        help="epoch (1-based) à partir de laquelle le backbone est fine-tuné",
    )
    parser.add_argument(
        "--early-stopping", type=float, default=None,
        help="seuil d'amélioration d'accuracy train (early stopping simulé)",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false",
        default=None, help="ne pas charger les poids pré-entraînés",
    )
    parser.add_argument(
        "--device", default=None, help="device (cpu, cuda, ...)"
    )
    parser.add_argument(
        "--limit-runs", type=int, default=None,
        help="limiter le nombre de runs (debug)",
    )
    parser.add_argument(
        "--transform-function", default=None,
        help=(
            "prétraitement d'image (brightness_normalized) ; "
            "par défaut : valeur du YAML"
        ),
    )
    parser.add_argument(
        "--augmentation-params", default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--transform-params", default=None,
        help=argparse.SUPPRESS,
    )

    return parser


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = resolve_args(args)

    required = {
        "crop_path": args.crop_path,
        "crop_file": args.crop_file,
        "split_file": args.split_file,
        "out_dir": args.out_dir,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(
            "Paramètres manquants : "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )

    return run_training(args)


if __name__ == "__main__":
    main()
