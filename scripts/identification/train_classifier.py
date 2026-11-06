"""
Modular training of a classification head (ClassificationHead) on top of a
feature extractor (MiewID / MegaDescriptor / ResNet152).

PRINCIPLES
----------
* The classification head is the `ClassificationHead` class already defined
  in `scripts/network.py`: it is imported, not rewritten.
* The loss is a CrossEntropy.
* The feature extractor can be:
    - fully frozen (on-the-fly feature extraction);
    - partially fine-tuned: the layers between `last_frozen_layer` and
      `last_layer` are trained from `defreeze_epoch`.
* Data augmentation (RandomAffine + ColorJitter) is applied ON THE FLY by the
  DataLoader from `aug_epoch`: each image is re-transformed at every epoch
  (standard PyTorch behaviour, no pre-computation, nothing is saved to disk).
* The number of runs is the one present in the splits file.

USAGE
-----
    python -m scripts.train_classifier -c config/pipeline.yaml \
        --feature-extractor ResNet152 --last-layer fc --last-frozen-layer layer3

or, from `pipeline.py`, via `run_training(args)`.

OUTPUTS (in `--out-dir`)
------------------------
    train_curv_run_<run>.png   loss / accuracy curves for the run
    model_run_<run>.pth        model weights at the stop epoch (early stopping)
    val_pred_early_stop.csv    predictions at the stop epoch,
                               appended after each run
    runs_summary.csv           per-run metrics (train_acc, val_acc, ...)
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
# BOOTSTRAP: the script must be importable as
# `scripts.train_classifier` (repository root in sys.path)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = Path('/pbs/home/a/ameurisse')

from network import ClassificationHead  # noqa: E402


# ============================================================
# FEATURE EXTRACTOR REGISTRY
# ------------------------------------------------------------
# input_size / mean / std: preprocessing expected by each model.
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
    "ResNet50": {
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
        "default_last_frozen_layer": "layers.3",
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
# GENERIC UTILITIES
# ============================================================
def set_seed(seed: int) -> None:
    """Set all seeds (torch, numpy, random)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_preprocessing(name) -> Callable | None:
    """Return function from `image_preprocessing.py` with the same name
    """
    if name is None:
        return None

    key = str(name).strip()

    # Lazy import: `image_preprocessing` is only loaded if needed.
    try:
        from scripts.identification import image_preprocessing
    except ModuleNotFoundError:
        import image_preprocessing  # direct script execution

    function = getattr(image_preprocessing, key, None)

    if function:
        return function

    available = sorted(
        attr
        for attr in dir(image_preprocessing)
        if callable(getattr(image_preprocessing, attr))
    )
    raise ValueError(
        f"Prétraitement inconnu : {name} "
        f"(choix : {', '.join(available)})"
    )


def _parse_sequence(value) -> list | None:
    """Normalize a YAML parameter into a list of numbers.

    YAML does not recognize tuples: `(0.1, 0, 1)` is loaded as a STRING.
    We therefore accept either a list, a tuple, a number or a string such as
    "(0.1, 0, 1)" / "[0.9, 1.10]".
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
    """Normalize the `translate` parameter of RandomAffine.

    Accepts:
        - None                      -> None
        - (tx, ty)                 -> (tx, ty)
        - (tx_min, tx_max, ty_min, ty_max) -> as is
    A triplet is interpreted as (tx, ty) (the 3rd value is ignored, a
    warning is displayed).
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
    """Normalize the `scale` parameter of RandomAffine."""
    values = _parse_sequence(scale)
    if values is None:
        return None
    if len(values) == 2:
        return (values[0], values[1])
    if len(values) == 1:
        return float(values[0])
    raise ValueError(f"scale invalide : {scale}")


def _normalize_jitter_value(value):
    """Normalize a ColorJitter parameter (number or interval)."""
    values = _parse_sequence(value)
    if values is None:
        return None
    if len(values) == 1:
        return float(values[0])
    if len(values) == 2:
        return (values[0], values[1])
    raise ValueError(f"paramètre ColorJitter invalide : {value}")


def build_augmentation(
    params: dict | None,
    aug_flags: dict | None = None,
) -> transforms.Compose | None:
    """Build the RandomAffine + ColorJitter augmentation from the YAML.

    `params` = contents of `data_augmentation.torch_transformation`
    (or `data_transformation.params` for compatibility).
    `aug_flags` = dict from `pipeline.AUG_PARAMS` (keys
    ``rotation``, ``translation``, ``scaling``, ``saturation``, ``hue``).
    Only transformations whose flag is ``True`` are applied.
    """
    if not params:
        return None

    flags = aug_flags or {}
    operations = []

    affine_params = params.get("affine") or {}
    if affine_params:
        kwargs = {}
        if flags.get("rotation", True) and affine_params.get("degrees") is not None:
            kwargs["degrees"] = affine_params["degrees"]
        if flags.get("translation", True):
            translate = _normalize_translate(affine_params.get("translate"))
            if translate is not None:
                kwargs["translate"] = translate
        if flags.get("scaling", True):
            scale = _normalize_scale(affine_params.get("scale"))
            if scale is not None:
                kwargs["scale"] = scale
        if affine_params.get("shear") is not None:
            kwargs["shear"] = affine_params["shear"]
        if kwargs:
            operations.append(transforms.RandomAffine(**kwargs))

    jitter_params = (
        params.get("colorjitter")
        or params.get("colorjittering")
        or params.get("color_jitter")
        or {}
    )
    if jitter_params:
        kwargs = {}
        for key in ("brightness", "contrast", "saturation", "hue"):
            if not flags.get(key, True):
                continue
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
    """Image pipeline: preprocessing -> augmentation -> resize -> tensor.

    Augmentation is applied BEFORE resizing (on the PIL image).
    """
    operations = []

    preprocessing = None
    if preprocessing_name is not None:
        try:
            preprocessing = resolve_preprocessing(preprocessing_name)
        except ValueError:
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
    """Dataset over raw crop images.

    Images are stored as ``crop_dir / model_name / detection / image_name``:

    * ``crop_dir`` is the root crop directory;
    * ``model_name`` is the detection model name, resolved from the parent
      directory of the detection folder;
    * ``detection`` is the value of the ``detection`` column of
      ``images_df`` (for example ``"left_ear_seg"``).

    The transform (including random augmentation) is applied on every
    ``__getitem__`` call, so each image is re-transformed at every epoch.
    Nothing is cached on disk.

    ``needs_backbone = True`` tells the training loop that items are images,
    so a forward pass through the feature extractor is required before the
    classification head.

    When ``metadata`` is provided (list of dicts), the returned item is
    ``(image, label, metadata)``. This is used for the validation set to
    write ``val_pred.csv``.
    """

    needs_backbone = True

    def __init__(
        self,
        crop_dir: Path,
        crop_ids: list,
        labels,
        transform,
        model_names: list | None = None,
        detections: list | None = None,
        metadata: list | None = None,
    ):
        self.crop_dir = Path(crop_dir)
        self.crop_ids = crop_ids
        self.labels = labels
        self.transform = transform
        self.model_names = model_names
        self.detections = detections
        self.metadata = metadata

    def _image_path(self, idx: int) -> Path:
        """Return ``crop_dir / model_name / detection / image_name``."""
        path = self.crop_dir
        if self.model_names is not None:
            path = path / str(self.model_names[idx])
        if self.detections is not None:
            path = path / str(self.detections[idx])
        return path / str(self.crop_ids[idx])

    def __len__(self):
        return len(self.crop_ids)

    def __getitem__(self, idx):
        image = Image.open(self._image_path(idx)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.metadata is not None:
            return image, self.labels[idx], self.metadata[idx]
        return image, self.labels[idx]


# ============================================================
# BACKBONE: construction, feature extraction, freeze/unfreeze
# ============================================================
@dataclass
class Backbone:
    """Feature extractor + architecture metadata."""

    name: str
    model: nn.Module
    input_size: int
    padding: int
    mean: tuple
    std: tuple
    children: list


def build_backbone(name: str, pretrained: bool = True) -> Backbone:
    """Build the requested feature extractor."""
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

    else:      # pragma: no cover - unreachable case
        raise ValueError(name)

    model.eval()

    return Backbone(
        name=name,
        model=model,
        input_size=spec["input_size"],
        padding=spec.get("padding", None),
        mean=spec["mean"],
        std=spec["std"],
        children=list(model.named_children()),
    )


def _is_after(key: tuple, frozen_key: tuple) -> bool:
    """True if the module with rank `key` is located AFTER the frozen module.

    Descendants of the frozen module (for example the bottlenecks of
    "layer3") are NOT considered as after it: they remain frozen.
    """
    if key <= frozen_key:
        return False
    return key[: len(frozen_key)] != frozen_key


def resolve_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve a module path ("head.global_pool" -> model.head.global_pool)."""
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
    """Return the output of a given module of the model.

    A hook is placed on the module to retrieve its output during the full
    model forward pass. The returned tensor is part of the graph: gradients
    can therefore flow back to the fine-tuned layers.
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
    """Wrap the feature extractor.

    * `forward(x)` returns the feature vector (output of `last_layer`).
    * `set_trainable(flag)` enables / disables fine-tuning of the layers
      between `last_frozen_layer` and `last_layer`.
    * `set_mode(training)` puts the model in train mode (BatchNorm) for the
      fine-tuned layers only, and in eval mode for the others.
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
        valid_layers = [n for n,_ in self.model.named_modules()]
        if last_layer not in valid_layers:
            raise KeyError(
                f"last_layer '{last_layer}' inconnu pour "
            )
        if last_frozen_layer is not None and last_frozen_layer not in valid_layers:
            raise KeyError(
                f"last_frozen_layer '{last_frozen_layer}' inconnu pour "
            )

        # --------------------------------------------------
        # Fine-tunable layers: between last_frozen_layer and
        # last_layer (bounds included); the original classification
        # layer is never trained (replaced by the head).
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

        # Everything is frozen by default: unfreezing is driven by
        # `set_trainable` (epoch `defreeze_epoch`).
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._feature_dim = self._compute_feature_dim()

    # ------------------------------------------------------------
    def _module_order_keys(self) -> dict:
        """Associate each module with its "rank" in execution order.

        The rank is the tuple of CHILD INDICES from the root:
        ("conv1",) -> (0,), ("layer4",) -> (7,), ("layer4.2.conv1",) -> (7, 2, 0).
        Comparing two ranks gives the execution order.
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
        """Modules to fine-tune (last_frozen_layer excluded, last_layer included).

        Fine-tuning applies to all modules located AFTER `last_frozen_layer`
        and UP TO `last_layer` (included).
        """
        keys = self._module_order_keys()
        key_last = self._order_key(self.last_layer)
        if self.last_frozen_layer is None:
            key_frozen = (-1,)
        else:
            key_frozen = self._order_key(self.last_frozen_layer)
            if key_last <= key_frozen:
                print(key_last, key_frozen)
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
            parent = ".".join(path.split(".")[:-1]) # parent layer of path
            if parent and parent in in_range: # if the parent is in in_range, no need to add it since the parent covers its children
                continue
            selected.append((path, key))

        selected.sort(key=lambda item: item[1]) # sort in ascending order

        return [
            resolve_module(self.model, path) for path, _ in selected
        ]

    def _compute_feature_dim(self) -> int:
        dummy = torch.zeros(
            1, 3, self.backbone.input_size, self.backbone.input_size
        )
        with torch.no_grad():
            features = extract_features(
                self.model, dummy, self.last_layer
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
        """Parameters potentially fine-tuned (depending on the epoch)."""
        return self._candidate_parameters

    def set_trainable(self, trainable: bool) -> None:
        """Enable / disable fine-tuning of the backbone."""
        for parameter in self._candidate_parameters:
            parameter.requires_grad_(trainable)

    def set_mode(self, training: bool) -> None:
        """Train/eval mode: only the fine-tuned layers are switched to
        train mode (BatchNorm), the others stay in eval mode.

        As long as the backbone is frozen (before `defreeze_epoch`), the
        BatchNorm statistics remain fixed: features do not change from one
        epoch to another.
        """
        self.model.eval()
        if training:
            for module in self._trainable_modules:
                if any(p.requires_grad for p in module.parameters()):
                    module.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # check whether grad is needed and flatten before the classification head -> avgpool?
        if any(p.requires_grad for p in self._candidate_parameters):
            features = extract_features(
                self.model, x, self.last_layer
            )
        else:
            with torch.no_grad():
                features = extract_features(
                    self.model, x, self.last_layer
                )
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return features


# ============================================================
# TRAINING
# ============================================================

def save_train_curves(
    out_dir: Path,
    run: int,
    train_error: list,
    val_error: list,
    train_accuracy: list,
    val_accuracy: list,
    num_classes: int,
) -> Path:
    """Save train/validation curves for one run.

    Loss is plotted on top and accuracy at the bottom, with the random
    baseline ``1 / num_classes``.
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
    """Extract features from a dataset without accumulating batch tensors."""

    backbone.set_mode(False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if len(dataset) == 0:
        return torch.empty((0, backbone.feature_dim))

    features = None
    offset = 0

    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device, non_blocking=True)
            batch_features = backbone(batch).cpu()

            if features is None:
                features = torch.empty(
                    (len(dataset), *batch_features.shape[1:]),
                    dtype=batch_features.dtype,
                )

            end = offset + len(batch_features)
            features[offset:end] = batch_features
            offset = end

            del batch, batch_features

    return features


def train_one_run(
    seed: int,
    make_train_loader: Callable[[int, bool], DataLoader],
    backbone: FeatureModel,
    val_dataset,
    class_names: list,
    num_classes: int,
    device: torch.device,
    num_epoch: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    defreeze_epoch: int | None = None,
    min_acc_improvement: float | None = 1.0,
    aug_epoch: int | None = None,
):
    """Train the classification head (and optionally the backbone).

    `make_train_loader(epoch, frozen)` (1-based epoch) returns a DataLoader
    for the given epoch. Its dataset yields (tensor, label) pairs:
        - pre-computed features (TensorDataset) when the backbone is frozen
          AND augmentation is not yet active: they are computed once (the
          backbone being frozen, they remain valid);
        - images (ImageCropDataset, ``needs_backbone=True``) otherwise: the
          backbone computes the features of each batch on the fly, which is
          required as soon as augmentation is active or the backbone is
          fine-tuned (otherwise features would be stale).

    Training goes through ALL epochs (num_epoch): curves are collected to
    decide the stop epoch afterwards. If `min_acc_improvement` is set, the
    stop corresponds to the first epoch whose train accuracy improvement
    falls below the threshold. If it is `None`, no early stopping is applied
    and the stop corresponds to the last epoch.

    Returns
    -------
    dict
        - train_error / val_error : list[float] (loss per epoch)
        - train_accuracy / val_accuracy : list[float] (accuracy % per epoch)
        - early_stop_predictions : list[dict]
            Validation predictions at the stop epoch (or the last epoch if
            no stop is triggered).
        - stop_epoch : int
            Index (0-based) of the stop epoch.
        - is_finetuned : bool
            True if the backbone was actually fine-tuned (at least one
            trainable parameter) at the stop epoch.
        - model_dict / backbone_state_dict
            Weights saved AT THE STOP EPOCH.
    """

    set_seed(seed)
    stop_epoch = None
    stop_head_state = None
    stop_backbone_state = None
    # True if the backbone was actually fine-tuned (at least one trainable
    # parameter) at the stop epoch. Used before saving the weights.
    stop_is_finetuned = False
    dim_in = backbone._feature_dim
    head = ClassificationHead(
        dim_out=num_classes,
        dim_in=dim_in,
    ).to(device)

    if head.out_features != num_classes:
        raise AssertionError(
            "Classification head output size mismatch."
        )

    criterion = nn.CrossEntropyLoss()

    # The backbone is always in the optimizer: the fine-tuned layers are
    # simply inactive (requires_grad=False) until the unfreeze epoch is
    # reached.
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

    # Only the predictions of the stop epoch and, by default, of the last
    # epoch are kept (the others are discarded after validation).
    #if stop epoch happens before fine-tuning launch defreezing and waits at least 5 epochs before defreezing
    early_stop_predictions = None
    last_epoch_predictions = None

    # Fine-tuning state at the last processed epoch (see loop below).
    is_finetuned = False

    def snapshot_state(module: nn.Module) -> dict:
        """Copy the weights to CPU to freeze the state of an epoch."""
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }

    for epoch in range(num_epoch):

        # ====================================================
        # TRAINING
        # ====================================================

        # Unfreeze the fine-tuned layers from defreeze_epoch (1-based).
        # `None` => no fine-tuning
        if defreeze_epoch is None:
            backbone.set_trainable(False)
        else:
            backbone.set_trainable((epoch + 1) >= defreeze_epoch)

        # Indicates whether the backbone is actually fine-tuned at this epoch
        # (at least one backbone parameter is trainable).
        is_finetuned = any(
            parameter.requires_grad
            for parameter in backbone.trainable_parameters
        )

        # Frozen backbone => features do not change from one epoch to another.
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
                # Images augmented on the fly: features computed on the
                # current batch, nothing is pre-computed or saved.
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

        # Decide whether the predictions of THIS epoch should be kept.
        # Only the simulated stop epoch and the last epoch are kept: rows
        # from other epochs are useless and memory-expensive.
        is_stop_epoch = False
        if (
            min_acc_improvement is not None
            and early_stop_predictions is None
            and len(train_accuracy) >= 2
        ):
            previous_acc = train_accuracy[-2]
            current_acc = train_accuracy[-1]

            if previous_acc > 0:
                improvement = (
                    (current_acc - previous_acc) / previous_acc * 100
                )
                is_stop_epoch = improvement < min_acc_improvement
                if frozen:
                    defreeze_epoch = epoch + 1
                    is_stop_epoch = None
                elif (epoch + 1 - defreeze_epoch)<5:
                    is_stop_epoch = None
                elif (epoch + 1 - aug_epoch)<5:
                    is_stop_epoch = None



        is_last_epoch = epoch == num_epoch - 1
        # If no early stopping has been triggered yet, keep the predictions
        # of the last epoch: they will serve as the stop if
        # `min_acc_improvement` is None.
        keep_predictions = is_stop_epoch or (
            is_last_epoch and stop_epoch is None
        )

        val_prediction_rows = [] if keep_predictions else None

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

                predicted = torch.argmax(
                    output,
                    dim=1,
                )

                corrects += (
                    predicted == label
                ).sum().item()

                if not keep_predictions:
                    continue

                # --------------------------------------------
                # Save predictions for this epoch
                # --------------------------------------------

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )

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

        if is_last_epoch and stop_epoch is None:
            last_epoch_predictions = val_prediction_rows

        # Freeze the weights of the stop epoch: THEY (and not those of the
        # last epoch) will be saved.
        if is_stop_epoch:
            early_stop_predictions = val_prediction_rows
            stop_epoch = epoch
            stop_is_finetuned = is_finetuned
            stop_head_state = snapshot_state(head)
            stop_backbone_state = snapshot_state(backbone.model)

    # If no stop was triggered (early stopping disabled or never reached),
    # the stop corresponds to the last epoch.
    if stop_epoch is None:
        stop_epoch = len(train_accuracy) - 1
        stop_is_finetuned = is_finetuned
        early_stop_predictions = last_epoch_predictions
        stop_head_state = snapshot_state(head)
        stop_backbone_state = snapshot_state(backbone.model)

    return {
        "train_error": train_error,
        "val_error": val_error,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "early_stop_predictions": early_stop_predictions,
        "stop_epoch": stop_epoch,
        "is_finetuned": stop_is_finetuned,
        "model_dict": stop_head_state,
        "backbone_state_dict": stop_backbone_state,
    }


# ============================================================
# DATA PREPARATION (runs)
# ============================================================
def build_detection_model_map(crop_dir: Path) -> dict[str, str]:
    """Map each detection folder to the name of its parent model folder.

    Crop images are stored as ``crop_dir / model_name / detection / image``.
    The model name is therefore the name of the parent directory of every
    folder whose name is a ``detection`` value from ``images_df``.

    Parameters
    ----------
    crop_dir:
        Root crop directory.

    Returns
    -------
    dict[str, str]
        Mapping ``detection -> model_name``.
    """
    crop_dir = Path(crop_dir)
    if not crop_dir.exists():
        raise FileNotFoundError(f"Crop directory not found: {crop_dir}")

    mapping: dict[str, str] = {}
    for model_dir in crop_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for detection_dir in model_dir.iterdir():
            if not detection_dir.is_dir():
                continue
            detection = detection_dir.name
            if detection in mapping and mapping[detection] != model_dir.name:
                warnings.warn(
                    f"Detection {detection!r} found under several model "
                    f"directories; keeping {mapping[detection]!r}."
                )
                continue
            mapping[detection] = model_dir.name

    return mapping


def build_run_dataframe(
    images_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    run,
) -> pd.DataFrame:
    """Assemble the images of one run.

    Parameters
    ----------
    images_df:
        Images table. It must contain ``image``, ``indiv``, ``event``,
        ``label`` and ``detection`` columns.
    splits_df:
        Split table (``event``, ``run``, ``split``).
    run:
        Run identifier to keep.

    Returns
    -------
    pandas.DataFrame
        Images of the run with the additional ``split`` column.
    """
    required = {"image", "indiv", "event", "label", "detection"}
    missing = required - set(images_df.columns)
    if missing:
        raise ValueError(
            f"images_df is missing columns: {', '.join(sorted(missing))}."
        )

    run_splits = splits_df[splits_df["run"] == run]
    if run_splits.empty:
        raise ValueError(f"No split found for run {run!r}.")

    images = images_df.dropna(
        subset=["image", "indiv", "event", "label", "detection"]
    ).copy()
    images["image"] = images["image"].astype(str)
    images["label"] = images["label"].astype(str)
    images["detection"] = images["detection"].astype(str)

    merged = images.merge(
        run_splits[["event", "split"]],
        on="event",
        how="inner",
    )
    merged = merged[merged["split"].isin(["train", "val"])]
    merged = merged.dropna(subset=["split"])

    if merged.empty:
        raise ValueError(f"No image found for run {run!r}.")

    return merged.reset_index(drop=True)


# ============================================================
# MAIN LOOP OVER RUNS
# ============================================================
def run_training(args: argparse.Namespace) -> pd.DataFrame:
    """Train the head for all runs of the splits file.

    Returns the run summary table (one row per run).
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Models are saved directly in the results folder.
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    set_seed(args.seed)

    # ------------------------------------------------------------
    # TABLES (single DataFrames)
    # ------------------------------------------------------------
    splits_df = getattr(args, "splits_df", None)
    images_df = getattr(args, "images_df", None)

    # Fallback for command-line usage: read the tables from CSV files.
    if splits_df is None and getattr(args, "split_file", None):
        splits_df = pd.read_csv(args.split_file)
    if images_df is None and getattr(args, "crop_file", None):
        images_df = pd.read_csv(args.crop_file)

    if splits_df is None or images_df is None:
        raise ValueError(
            "run_training expects 'splits_df' and 'images_df' DataFrames."
        )
    if splits_df.empty or images_df.empty:
        raise ValueError("'splits_df' and 'images_df' must not be empty.")

    print(f"Feature extractor : {args.feature_extractor}")
    print(f"last_layer            : {args.last_layer}")
    print(f"last_frozen_layer     : {args.last_frozen_layer}")
    print(f"Images                : {len(images_df)}")
    print(f"Device                : {device}")

    # ------------------------------------------------------------
    # BACKBONE
    # ------------------------------------------------------------
    architecture = ARCHITECTURES[args.feature_extractor]
    # ------------------------------------------------------------
    # TRANSFORMATIONS
    # ------------------------------------------------------------
    augmentation = build_augmentation(
        args.augmentation_params,
        getattr(args, "aug_params", None),
    )

    base_transform = build_input_transform(
        args.preprocessing_function,
        args.transform_params,
        None,
        architecture["input_size"],
        architecture["mean"],
        architecture["std"],
        architecture.get("padding"),
    )

    augmented_transform = build_input_transform(
        args.preprocessing_function,
        args.transform_params,
        augmentation,
        architecture["input_size"],
        architecture["mean"],
        architecture["std"],
        architecture.get("padding"),
    )

    if augmentation is None:
        print("Aucune augmentation configurée.")
    else:
        print(f"Augmentation (epoch >= {args.aug_epoch}) : {augmentation}")

    crop_dir = Path(args.crop_path)
    detection_model_map = build_detection_model_map(crop_dir)

    # ------------------------------------------------------------
    # RUNS
    # ------------------------------------------------------------
    early_stop_path = out_dir / "val_pred_early_stop.csv"

    # Fresh file for this execution: predictions are appended after each
    # run so that already-finished runs are kept if a later run crashes.
    early_stop_path.unlink(missing_ok=True)

    summary_rows: list[dict] = []
    run_val_accuracies: dict = {}
    num_stop_rows = 0

    runs = sorted(splits_df["run"].dropna().unique().tolist())
    if args.limit_runs is not None:
        runs = runs[: args.limit_runs]
    if not runs:
        raise ValueError("No run found in 'splits_df'.")

    print(f"\nRuns : {len(runs)}")

    for run in runs:
        print(f"\n----- Run {run} -----")

        started = time.time()

        df = build_run_dataframe(images_df, splits_df, run)

        # Resolve the model name from the parent of every detection folder.
        missing_detections = sorted(
            set(df["detection"]) - set(detection_model_map)
        )
        if missing_detections:
            raise FileNotFoundError(
                f"No model directory found under {crop_dir} for detection(s): "
                f"{', '.join(missing_detections)}."
            )
        df["model_name"] = [
            detection_model_map[detection] for detection in df["detection"]
        ]

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)

        if train_df.empty or val_df.empty:
            raise ValueError(
                f"Run {run!r}: empty train/val split "
                f"(train={len(train_df)}, val={len(val_df)})."
            )

        # The classification target is the value of the `label` column of
        # images_df.
        class_names = sorted(df["label"].unique().tolist())
        label_map = {name: idx for idx, name in enumerate(class_names)}
        num_classes = len(class_names)

        train_crop_ids = list(train_df["image"])
        val_crop_ids = list(val_df["image"])
        train_model_names = list(train_df["model_name"])
        val_model_names = list(val_df["model_name"])
        train_detections = list(train_df["detection"])
        val_detections = list(val_df["detection"])

        train_labels = torch.tensor(
            [label_map[name] for name in train_df["label"]]
        )
        val_labels = torch.tensor(
            [label_map[name] for name in val_df["label"]]
        )

        # Validation metadata (image, individual, model, detection, ...).
        metadata_columns = [
            c
            for c in (
                "image",
                "indiv",
                "model_name",
                "detection",
                "label",
                "event",
                "split",
            )
            if c in val_df.columns
        ]
        val_metadata = val_df[metadata_columns].to_dict("records")

        print(
            f"  Classes: {num_classes}, Train: {len(train_df)}, "
            f"Val: {len(val_df)}"
        )

        # ----------------------------------------------------
        # Per-epoch training loader:
        #   - epochs < aug_epoch: images without augmentation;
        #   - epochs >= aug_epoch: images re-transformed at every
        #     epoch by the DataLoader (on-the-fly augmentation).
        # ----------------------------------------------------
        shuffle_gen = torch.Generator().manual_seed(args.seed)

        # Base-feature cache: computed ONLY once and used only while
        # the backbone is frozen and augmentation is not active.
        base_features_cache: dict = {}

        def get_base_features() -> torch.Tensor:
            if "features" not in base_features_cache:
                dataset = ImageCropDataset(
                    crop_dir,
                    train_crop_ids,
                    train_labels,
                    base_transform,
                    model_names=train_model_names,
                    detections=train_detections,
                )
                base_features_cache["features"] = compute_features(
                    feature_model,
                    dataset,
                    device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            return base_features_cache["features"]

        def make_train_loader(
            epoch_1based: int, frozen: bool
        ) -> DataLoader:
            use_augmentation = (
                augmentation is not None
                and epoch_1based >= args.aug_epoch
            )

            if not use_augmentation and frozen:
                # Pre-computed features (backbone is frozen).
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
                    crop_dir,
                    train_crop_ids,
                    train_labels,
                    transform,
                    model_names=train_model_names,
                    detections=train_detections,
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
            crop_dir,
            val_crop_ids,
            val_labels,
            base_transform,
            model_names=val_model_names,
            detections=val_detections,
            metadata=val_metadata,
        )
        backbone = build_backbone(
            args.feature_extractor, pretrained=args.pretrained
        )

        feature_model = FeatureModel(
            backbone,
            last_layer=args.last_layer,
            last_frozen_layer=args.last_frozen_layer,
        ).to(device)
        result = train_one_run(
            args.seed,
            make_train_loader,
            feature_model,
            val_dataset,
            class_names,
            num_classes,
            device,
            num_epoch=args.num_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            defreeze_epoch=args.defreeze_epoch,
            min_acc_improvement=args.early_stopping,
            aug_epoch=args.aug_epoch
        )

        train_error = result["train_error"]
        val_error = result["val_error"]
        train_accuracy = result["train_accuracy"]
        val_accuracy = result["val_accuracy"]
        stop_epoch = result["stop_epoch"]
        stop_val_accuracy = val_accuracy[stop_epoch]
        run_val_accuracies[run] = stop_val_accuracy

        # ----------------------------------------------------
        # TRAINING CURVES
        # ----------------------------------------------------
        curv_path = save_train_curves(
            out_dir,
            run,
            train_error,
            val_error,
            train_accuracy,
            val_accuracy,
            num_classes,
        )
        print(f"  Curves saved: {curv_path}")

        # ----------------------------------------------------
        # MODEL WEIGHTS (early stopping only)
        # ----------------------------------------------------
        model_path = model_dir / f"model_run_{run}.pth"
        detections = sorted(df["detection"].unique().tolist())
        is_finetuned = result["is_finetuned"]

        if not is_finetuned:
            print(
                f"  Modèle non fine-tuné à l'epoch {stop_epoch + 1} : "
                f"poids non enregistrés ({model_path})."
            )
        else:
            torch.save(
                {
                    "backbone_state_dict": result["backbone_state_dict"],
                    "head_state_dict": result["model_dict"],
                    "run": run,
                    "class_names": class_names,
                    "detections": detections,
                    "feature_extractor": args.feature_extractor,
                    "last_layer": args.last_layer,
                    "last_frozen_layer": args.last_frozen_layer,
                    "num_epoch": args.num_epoch,
                    "stop_epoch": stop_epoch + 1,
                    "early_stopping": args.early_stopping,
                    "is_finetuned": is_finetuned,
                    "val_accuracy": stop_val_accuracy,
                    "train_accuracy": train_accuracy[stop_epoch],
                },
                model_path,
            )
            print(f"  Weights saved (epoch {stop_epoch + 1}): {model_path}")

        # ----------------------------------------------------
        # VALIDATION PREDICTIONS
        # ----------------------------------------------------
        stop_rows = result["early_stop_predictions"] or []

        for row in stop_rows:
            row["run"] = run

        # Append this run's predictions immediately (header only on first
        # write) so a crash does not lose the runs already completed.

        run_stop_pred_df = build_val_pred_df(stop_rows)
        run_stop_pred_df.to_csv(
            early_stop_path,
            mode="a",
            header=not early_stop_path.exists(),
            index=False,
        )
        num_stop_rows += len(run_stop_pred_df)

        best_epoch = int(np.argmax(val_accuracy))

        summary_rows.append(
            {
                "run": run,
                "detections": ", ".join(detections),
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
            f"  Train acc : {train_accuracy[stop_epoch]:.2f}% | "
            f"Val acc : {val_accuracy[stop_epoch]:.2f}% | "
            f"stop epoch {stop_epoch + 1}"
        )

    # ============================================================
    # FINAL SAVE
    # ============================================================

    print(f"Predictions (stop epoch): {early_stop_path}")
    print(f"  {num_stop_rows} rows")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "runs_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Runs summary: {summary_path}")

    """# ------------------------------------------------------------
    # Keep only the model whose val accuracy is closest to the
    # median val accuracy of all runs.
    # ------------------------------------------------------------
    if run_val_accuracies:
        median_val_accuracy = float(
            np.median(list(run_val_accuracies.values()))
        )
        median_run = min(
            run_val_accuracies,
            key=lambda run: abs(
                run_val_accuracies[run] - median_val_accuracy
            ),
        )

        selected_path = model_dir / f"model_run_{median_run}.pth"
        median_model_path = model_dir / "median_model.pth"

        if selected_path.exists():
            for run in run_val_accuracies:
                candidate_path = model_dir / f"model_run_{run}.pth"
                if candidate_path != selected_path and candidate_path.exists():
                    candidate_path.unlink()

            median_model_path.unlink(missing_ok=True)
            selected_path.rename(median_model_path)

            print(
                f"Median model: run {median_run} "
                f"(val acc {run_val_accuracies[median_run]:.2f}%, "
                f"median {median_val_accuracy:.2f}%) -> {median_model_path}"
            )
        else:
            print(f"Median model not found: {selected_path}")"""

    return summary_df


# Columns of val_pred.csv, in the requested order.
VAL_PRED_COLUMNS = [
    "run",
    "model_name",
    "detection",
    "image",
    "video",
    "indiv",
    "label",
    "pred_label",
    "margin",
    "logits",
    "conf",
]


def build_val_pred_df(rows: list[dict]) -> pd.DataFrame:
    """Build the validation-prediction DataFrame.

    Columns: run, model_name, detection, image, video, indiv, label,
    pred_label, margin, logits, conf. ``indiv_name`` is kept as an alias
    for ``indiv`` (or ``label`` when ``indiv`` is absent) for compatibility
    with existing analysis scripts.
    """
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=VAL_PRED_COLUMNS)

    if "conf" not in df.columns and "conf_pred_label" in df.columns:
        df["conf"] = df["conf_pred_label"]

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
    """Load the pipeline configuration file (YAML)."""
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


# ============================================================
# CONFIGURATION PATHS
# ============================================================
def format_template(template: str, values: dict) -> str:
    """Replace the {placeholders} of a configuration path."""
    return template.format_map(defaultdict(str, values))


def build_path_values(config: dict) -> dict:
    """Values usable in configuration paths.

    The date is "sanitized" ("/" are replaced by "_") so that it can be
    used as a file name.
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
    """Resolve a configuration path relative to `home_path`."""
    rendered = format_template(template, values)
    path = Path(rendered)
    if path.is_absolute():
        return path
    return home_path / path


def resolve_config_paths(config: dict) -> dict:
    """Resolve the `paths.*` entries of the configuration.

    Returns a dict with the keys crop_path, crop_file, split_file and
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
    """Fill the arguments not provided on the command line with
    values from the pipeline configuration file."""
    config = load_pipeline_config(
        Path(args.config) if args.config else None
    )
    args.config_dict = config

    training = config.get("training_parameters", {}) or {}
    augmentation = config.get("data_augmentation", {}) or {}
    transformation = augmentation.get("torch_transformation")
    
    extractor_config = config.get("feature_extractor", {}) or {}
    extractor_params = config.get("feature_extractor_params", {}) or {}

    # `feature_extractor` is a nested block in pipeline.yaml
    # or chosen (overrun) in pipeline.py if default finetuning parameters are choosen
    if isinstance(extractor_config, dict):
        extractor_name = extractor_config.get("name")
        extractor_params = {**extractor_config, **extractor_params}
    else:
        extractor_name = extractor_config

    if args.feature_extractor is None:
        args.feature_extractor = extractor_name
        if args.last_layer is None:
            args.last_layer = extractor_params.get("last_layer")
        if args.last_frozen_layer is None:
            args.last_frozen_layer = extractor_params.get("last_frozen_layer")
    else:
        # Architecture defaults if the YAML does not specify anything
        architecture = ARCHITECTURES.get(args.feature_extractor, {})
        print("Default last and frozen layers choosen, if you want to choose them modidy pipeline.yaml \
        and put architecture to None in pipeline.py")
        if args.last_layer is None:
            args.last_layer = architecture.get("default_last_layer")
        if args.last_frozen_layer is None:
            args.last_frozen_layer = architecture.get(
                "default_last_frozen_layer"
            )
        if args.feature_extractor not in ARCHITECTURES:
            raise ValueError(
                f"Extracteur inconnu : {args.feature_extractor} "
                f"(choix : {', '.join(FEATURE_EXTRACTORS)})"
            )
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
    if args.batch_norm_only is None:
        args.batch_norm_only = training.get("batch_norm_only")

    if getattr(args, "augmentation_params", None) is None:
        args.augmentation_params = transformation
    if getattr(args, "aug_params", None) is None:
        args.aug_params = augmentation.get("aug_params") or {}
    if getattr(args, "preprocessing_function", None) is None:
        args.preprocessing_function = config.get("preprocessing_function")
    if getattr(args, "transform_params", None) is None:
        args.transform_params = config.get("params", {}) or {}

    # Paths: the command line takes precedence, otherwise the configuration.
    config_paths = resolve_config_paths(config)
    for name in ("crop_path", "crop_file", "split_file", "model_dir"):
        if getattr(args, name, None) is None and name in config_paths:
            setattr(args, name, str(config_paths[name]))


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
