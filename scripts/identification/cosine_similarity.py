"""
Recherche par similarité cosinus après l'entraînement.

Ce script est prévu pour être exécuté APRÈS `train_classifier.py`. Il :

1. charge le backbone des modèles fine-tunés sauvegardés dans
   `<out_dir>/models` ; si aucun modèle n'a été sauvegardé (aucun
   fine-tuning), utilise le backbone pré-entraîné avec les mêmes réglages
   que `train_classifier.py` ;
2. reconstruit le split du run correspondant ;
3. calcule les features du train et de la validation avec le même
   prétraitement que l'entraînement (sans augmentation) ;
4. pour chaque crop de validation, cherche les `num_points` features
   d'entraînement les plus proches (similarité cosinus) ;
5. prédit le label par vote majoritaire parmi ces voisins ;
6. écrit les prédictions dans `<out_dir>/val_pred.csv`.

Le nombre de voisins est lu dans `similarity_research.num_points` du fichier
`pipeline.yaml`. Il peut être surchargé par `--num-points`.

USAGE
-----
    python -m scripts.identification.cosine_similarity \
        -c config/pipeline.yaml --out-dir data/results/v1

ou, depuis `pipeline.py`, via `run_similarity(args)`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Le script doit pouvoir être importé comme
# `scripts.identification.cosine_similarity` ou exécuté directement.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_classifier  # noqa: E402

CHUNK_SIZE = 2048
OUTPUT_NAME = "val_cosine_similarity_pred.csv"

# ============================================================
# PRÉPARATION DES DONNÉES
# ============================================================
def load_run_tables(args: argparse.Namespace, run):
    """Charge les tables et reconstruit le split du run médian."""
    splits_df = getattr(args, "splits_df", None)
    images_df = getattr(args, "images_df", None)

    if splits_df is None:
        if not getattr(args, "split_file", None):
            raise ValueError("--split-file est requis.")
        splits_df = pd.read_csv(args.split_file)

    if images_df is None:
        if not getattr(args, "crop_file", None):
            raise ValueError("--crop-file est requis.")
        images_df = pd.read_csv(args.crop_file)

    df = train_classifier.build_run_dataframe(images_df, splits_df, run)
    return df


def build_datasets_and_metadata(
    args: argparse.Namespace,
    df: pd.DataFrame,
    transform,
    label_map: dict,
    run,
):
    """Construit les datasets train/val et les métadonnées de validation."""
    crop_dir = Path(args.crop_path)
    detection_model_map = train_classifier.build_detection_model_map(crop_dir)

    missing_detections = sorted(
        set(df["detection"]) - set(detection_model_map)
    )
    if missing_detections:
        raise FileNotFoundError(
            f"Aucun dossier modèle trouvé sous {crop_dir} pour les "
            f"détections : {', '.join(missing_detections)}."
        )

    df = df.copy()
    df["model_name"] = [
        detection_model_map[detection] for detection in df["detection"]
    ]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Run {run!r} : split vide "
            f"(train={len(train_df)}, val={len(val_df)})."
        )

    # On conserve toutes les métadonnées utiles du split (caméra, mode,
    # entry_day, etc.). Les colonnes réécrites par la prédiction sont
    # exclues pour éviter tout doublon.
    reserved_columns = {"run", "pred_label", "margin", "logits", "conf"}
    metadata_columns = [
        column
        for column in val_df.columns
        if column not in reserved_columns
    ]
    val_metadata = val_df[metadata_columns].to_dict("records")

    train_labels = torch.tensor(
        [label_map[name] for name in train_df["label"]]
    )
    val_labels = torch.tensor(
        [label_map[name] for name in val_df["label"]]
    )

    train_dataset = train_classifier.ImageCropDataset(
        crop_dir,
        list(train_df["image"]),
        train_labels,
        transform,
        model_names=list(train_df["model_name"]),
        detections=list(train_df["detection"]),
    )
    val_dataset = train_classifier.ImageCropDataset(
        crop_dir,
        list(val_df["image"]),
        val_labels,
        transform,
        model_names=list(val_df["model_name"]),
        detections=list(val_df["detection"]),
    )

    return train_df, val_df, train_dataset, val_dataset, val_metadata


# ============================================================
# MODÈLE / FEATURES
# ============================================================
def build_feature_model(
    checkpoint: dict,
    args: argparse.Namespace,
    device: torch.device,
):
    """Reconstruit le backbone du modèle médian et son prétraitement.

    Si le checkpoint ne contient pas de ``backbone_state_dict`` (cas d'un
    modèle non fine-tuné, donc non sauvegardé), le backbone est chargé avec
    ses poids pré-entraînés.
    """
    feature_extractor = checkpoint["feature_extractor"]
    if feature_extractor not in train_classifier.ARCHITECTURES:
        raise ValueError(
            f"Extracteur inconnu dans le checkpoint : {feature_extractor}"
        )

    architecture = train_classifier.ARCHITECTURES[feature_extractor]

    state_dict = checkpoint.get("backbone_state_dict")
    backbone = train_classifier.build_backbone(
        feature_extractor,
        pretrained=state_dict is None,
    )
    feature_model = train_classifier.FeatureModel(
        backbone,
        last_layer=checkpoint["last_layer"],
        last_frozen_layer=checkpoint.get("last_frozen_layer"),
    ).to(device)
    if state_dict is not None:
        feature_model.model.load_state_dict(state_dict)
    feature_model.model.eval()

    transform = train_classifier.build_input_transform(
        getattr(args, "preprocessing_function", None),
        getattr(args, "transform_params", None),
        None,
        architecture["input_size"],
        architecture["mean"],
        architecture["std"],
        architecture.get("padding"),
    )

    return feature_model, transform


def compute_features(
    feature_model,
    train_dataset,
    val_dataset,
    args: argparse.Namespace,
    device: torch.device,
):
    """Calcule les features train et validation."""
    train_features = train_classifier.compute_features(
        feature_model,
        train_dataset,
        device,
        batch_size=getattr(args, "batch_size", 64),
        num_workers=getattr(args, "num_workers", 0),
    )
    val_features = train_classifier.compute_features(
        feature_model,
        val_dataset,
        device,
        batch_size=getattr(args, "batch_size", 64),
        num_workers=getattr(args, "num_workers", 0),
    )
    return train_features, val_features


# ============================================================
# RECHERCHE DES VOISINS ET VOTE MAJORITAIRE
# ============================================================
def topk_majority_vote(
    val_features: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_points: int,
    num_classes: int,
    device: torch.device,
) -> dict:
    """Cherche les `num_points` voisins cosinus et vote par majorité.

    Returns
    -------
    dict
        - pred : np.ndarray (n_val,)
        - conf : np.ndarray (n_val,) fraction de votes de la classe gagnante
        - margin : np.ndarray (n_val,) écart entre les deux meilleurs votes
        - vote_fractions : np.ndarray (n_val, num_classes)
        - top_labels : np.ndarray (n_val, k) labels des voisins
        - top_similarities : np.ndarray (n_val, k)
    """
    n_train = train_features.shape[0]
    k = min(int(num_points), n_train)
    if k < 1:
        raise ValueError("Le nombre de voisins doit être >= 1.")

    val_normalized = F.normalize(val_features.to(device), dim=1)
    train_normalized = F.normalize(train_features.to(device), dim=1)
    train_labels = train_labels.to(device)

    predictions = []
    confidences = []
    margins = []
    vote_fractions_all = []
    top_labels_all = []
    top_similarities_all = []

    for start in range(0, val_normalized.shape[0], CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, val_normalized.shape[0])
        similarities = val_normalized[start:stop] @ train_normalized.T

        values, indices = torch.topk(similarities, k, dim=1)
        neighbour_labels = train_labels[indices]

        votes = torch.zeros(
            (neighbour_labels.shape[0], num_classes),
            dtype=torch.float32,
            device=device,
        )
        votes.scatter_add_(
            1,
            neighbour_labels,
            torch.ones_like(neighbour_labels, dtype=torch.float32),
        )

        predicted = votes.argmax(dim=1)
        max_votes = votes.gather(1, predicted[:, None]).squeeze(1)
        conf = max_votes / k

        if num_classes > 1:
            top_two = votes.topk(2, dim=1).values
            margin = (top_two[:, 0] - top_two[:, 1]) / k
        else:
            margin = torch.ones_like(conf)

        predictions.append(predicted.cpu())
        confidences.append(conf.cpu())
        margins.append(margin.cpu())
        vote_fractions_all.append((votes / k).cpu())
        top_labels_all.append(neighbour_labels.cpu())
        top_similarities_all.append(values.cpu())

        del similarities, votes

    return {
        "pred": torch.cat(predictions).numpy(),
        "conf": torch.cat(confidences).numpy(),
        "margin": torch.cat(margins).numpy(),
        "vote_fractions": torch.cat(vote_fractions_all).numpy(),
        "top_labels": torch.cat(top_labels_all).numpy(),
        "top_similarities": torch.cat(top_similarities_all).numpy(),
    }


# ============================================================
# CONSTRUCTION DU val_pred.csv
# ============================================================
def build_prediction_df(
    run,
    class_names: list,
    val_metadata: list,
    results: dict,
) -> pd.DataFrame:
    """Construit le DataFrame de prédictions de validation."""
    rows = []
    k = results["top_labels"].shape[1]

    for index, metadata in enumerate(val_metadata):
        predicted_class = int(results["pred"][index])

        row = {key: metadata[key] for key in metadata}
        row.update(
            {
                "run": run,
                "pred_label": class_names[predicted_class],
                "conf": round(results["conf"][index],2),
                "margin": round(results["margin"][index],2),
                "logits": [
                    round(value,2)
                    for value in results["vote_fractions"][index]
                ],
            }
        )

        for rank in range(k):
            neighbour_class = int(results["top_labels"][index, rank])
            row[f"label_{rank + 1}"] = class_names[neighbour_class]
            row[f"sim_{rank + 1}"] = round(
                results["top_similarities"][index, rank],2
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    for column in train_classifier.VAL_PRED_COLUMNS:
        if column not in df.columns:
            df[column] = None

    extra_columns = [
        column
        for column in df.columns
        if column not in train_classifier.VAL_PRED_COLUMNS
    ]
    ordered_columns = train_classifier.VAL_PRED_COLUMNS + extra_columns
    return df[ordered_columns]


# ============================================================
# ORCHESTRATION
# ============================================================
def get_num_points(args: argparse.Namespace) -> int:
    """Lit `similarity_research.num_points` (ou la surcharge CLI)."""
    override = getattr(args, "num_points", None)
    if override is not None:
        return max(1, int(override))

    config = getattr(args, "config_dict", None)
    if config is None:
        config = train_classifier.load_pipeline_config(
            Path(args.config) if getattr(args, "config", None) else None
        )

    similarity_config = config.get("similarity_research", {}) or {}
    return max(1, int(similarity_config.get("num_points", 1)))


def load_saved_checkpoints(model_dir: Path) -> list[dict]:
    """Charge les checkpoints fine-tunés de ``model_dir`` (hors médian)."""
    if not model_dir.exists():
        return []

    checkpoints = []
    for model_path in sorted(model_dir.iterdir()):
        if model_path.suffix != ".pth" or model_path.name == "median_model.pth":
            continue
        checkpoints.append(
            torch.load(model_path, map_location="cpu", weights_only=False)
        )
    return checkpoints


def get_runs(args: argparse.Namespace) -> list:
    """Retourne les runs du fichier de splits."""
    splits_df = getattr(args, "splits_df", None)
    if splits_df is None:
        if not getattr(args, "split_file", None):
            raise ValueError("--split-file est requis.")
        splits_df = pd.read_csv(args.split_file)
    return sorted(splits_df["run"].dropna().unique().tolist())


def build_pretrained_checkpoint(args: argparse.Namespace) -> dict:
    """Checkpoint factice décrivant le backbone pré-entraîné.

    Utilisé quand aucun poids n'a été sauvegardé (modèle non fine-tuné).
    """
    return {
        "feature_extractor": args.feature_extractor,
        "last_layer": args.last_layer,
        "last_frozen_layer": args.last_frozen_layer,
        # None => build_feature_model charge les poids pré-entraînés.
        "backbone_state_dict": None,
    }


def run_similarity(args: argparse.Namespace) -> pd.DataFrame:
    """Exécute la recherche cosinus sur les modèles fine-tunés.

    Si aucun modèle n'a été sauvegardé (aucun fine-tuning), le backbone
    pré-entraîné est utilisé pour tous les runs du fichier de splits.

    `args` doit contenir au minimum `out_dir`, `crop_path` et les tables
    `splits_df` / `images_df` (ou `split_file` / `crop_file`).
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(
        getattr(args, "output", None) or out_dir / OUTPUT_NAME
    )

    device = torch.device(
        getattr(args, "device", None)
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_dir = Path(args.out_dir) / "models"

    checkpoints = load_saved_checkpoints(model_dir)
    if checkpoints:
        jobs = [(checkpoint["run"], checkpoint) for checkpoint in checkpoints]
    else:
        print(
            "Aucun modèle fine-tuné trouvé dans "
            f"{model_dir} : utilisation du backbone pré-entraîné."
        )
        pretrained_checkpoint = build_pretrained_checkpoint(args)
        jobs = [
            (run, pretrained_checkpoint) for run in get_runs(args)
        ]

    cosine_results = []
    for run, checkpoint in jobs:
        num_points = get_num_points(args)

        df = load_run_tables(args, run)

        if checkpoint.get("backbone_state_dict") is None:
            # Modèle pré-entraîné : les classes proviennent du run.
            class_names = sorted(
                str(name) for name in df["label"].unique()
            )
        else:
            class_names = [str(name) for name in checkpoint["class_names"]]

        label_map = {name: index for index, name in enumerate(class_names)}

        missing_labels = set(df["label"]) - set(label_map)
        if missing_labels:
            raise KeyError(
                "Labels absents du checkpoint : "
                + ", ".join(sorted(missing_labels))
            )

        feature_model, transform = build_feature_model(checkpoint, args, device)
        (
            train_df,
            val_df,
            train_dataset,
            val_dataset,
            val_metadata,
        ) = build_datasets_and_metadata(args, df, transform, label_map, run)

        train_features, val_features = compute_features(
            feature_model,
            train_dataset,
            val_dataset,
            args,
            device,
        )

        results = topk_majority_vote(
            val_features,
            train_features,
            train_dataset.labels,
            num_points,
            len(class_names),
            device,
        )

        val_pred_df = build_prediction_df(
            run,
            class_names,
            val_metadata,
            results,
        )
        val_pred_df.to_csv(output_path,
        mode="a",
        header=not output_path.exists(),
        index=False)

        truth = np.array(
            [label_map[name] for name in val_df["label"]],
            dtype=np.int64,
        )
        accuracy = float((results["pred"] == truth).mean() * 100)
        cosine_acc = round(accuracy,0)
        cosine_results.append({"run": run, "cosine_acc": cosine_acc})

        print(f"  Validation accuracy : {accuracy:.2f}%")
        print(f"  Prédictions : {output_path}")

    cosine_df = pd.DataFrame(cosine_results)
    if not cosine_df.empty:
        cosine_path = out_dir / "cosine_accuracy_summary.csv"
        cosine_df.to_csv(cosine_path, index=False)
        print(f"Cosine accuracy summary: {cosine_path}")

    return cosine_df


# ============================================================
# CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    # On réutilise le parser de train_classifier pour que `resolve_args()`
    # dispose de tous les champs attendus (feature_extractor, transforms...).
    parser = train_classifier.build_parser()
    parser.description = (
        "Prédiction par similarité cosinus avec le backbone du modèle "
        "médian produit par train_classifier.py."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="checkpoint à utiliser",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV de sortie (défaut : <out-dir>/val_pred.csv)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="surcharge de similarity_research.num_points",
    )
    return parser


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = train_classifier.resolve_args(args)

    if not args.out_dir:
        parser.error("--out-dir est requis.")

    return run_similarity(args)


if __name__ == "__main__":
    main()
