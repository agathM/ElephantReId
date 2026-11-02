"""
PIPELINE D'ENTRAÎNEMENT

Lit un fichier de configuration YAML (par défaut `config/pipeline.yaml`) et :

1. crée un dossier de résultats versionné (`<results>/v1`, `v2`, ...) ;
2. y copie TOUS les fichiers `.yaml` : ceux référencés dans la configuration,
   ceux présents dans le dossier de la configuration, et le `pipeline.yaml`
   lui-même ;
3. exécute le script d'entraînement modulaire `scripts/train_classifier.py`
   (extraction des features, fine-tuning éventuel, tête ClassificationHead,
   augmentation RandomAffine + ColorJitter, un run par split) ;
4. complète le tableau récapitulatif des expériences (`all_experiments.csv`)
   avec une ligne par run (train_acc, val_acc, ...).

USAGE
-----
    python -m scripts.pipeline -c config/pipeline.yaml
    python -m scripts.pipeline -c config/pipeline.yaml --dry-run
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import importlib
import pandas as pd
import yaml
import train_classifier, detect_and_track
import os

from IPython.core.magics import script

#DETECTION CHOICES
DETECT_CHOICES = ['ear_seg','body_detect'] #'ear_seg','body_detect','ear_detect', 'head_detect'

#DEFAULT PARAMETERS FOR CHOICES
DETECT_PARAMS = {}
for choice in DETECT_CHOICES:
    if choice in ['ear_detect','head_detect']:
        DETECT_PARAMS.setdefault("head_ears_detect_09_08",{"conf":0.1,
                                                              'labels_to_keep':[choice]
                                                              })['labels_to_keep'].append(choice)
    elif choice =="body_detect":
        DETECT_PARAMS["MDV6-yolov10-e"] ={"conf":0.1,'labels_to_keep':[choice]}
    elif choice =="ear_seg":
        DETECT_PARAMS["seg_ears_lr_v2"] = {"conf":0.1,'labels_to_keep':[choice]}

# Data-augmentation flags: only transformations set to True are applied.
# The matching parameters are read from
# `data_augmentation.torch_transformation.params` in pipeline.yaml.
AUG_PARAMS = {
    "rotation": True,
    "translation": True,
    "scaling": True,
    "saturation": True,
    "hue": True,
}


EXPERIMENT_COLUMNS = [
    "version",
    "dataset_date",
    "detection_model",
    "split_method",
    "architecture",
    "data_augmentation",
    "classification_head",
    "train_acc",
    "val_acc",
]

EXPERIMENTS_CSV_NAME = "all_experiments.csv"



def load_config(config_dir: Path) -> dict:
    """Charge le fichier de configuration du pipeline."""
    config_path = config_dir / "pipeline.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


# ============================================================
# DOSSIER DE RÉSULTATS
# ============================================================
def next_results_dir(results_root: Path) -> tuple[str, Path]:
    """Détermine le prochain dossier versionné (`v1`, `v2`, ...).

    Retourne (version, dossier de résultats).
    """

    def contains_any_file(folder):
        """Return True if the folder or any subfolder contains at least one file."""
        return any((p.is_file() and p.suffix != ".yaml") for p in Path(folder).rglob("*"))
    versions = []
    if results_root.exists():
        for child in results_root.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                if not contains_any_file(child):
                    shutil.rmtree(child)
                else:
                    try:
                        versions.append(int(child.name[1:]))
                    except ValueError:
                        continue

    version = f"v{max(versions) + 1 if versions else 1}"
    return version, results_root / version


# ============================================================
# COPIE DES FICHIERS DE CONFIGURATION (.yaml)
# ============================================================
def collect_yaml_references(config: dict) -> list[str]:
    """Renvoie tous les chemins `.yaml` référencés dans la configuration."""
    references: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            if node.strip().lower().endswith((".yaml", ".yml")):
                references.append(node.strip())

    walk(config)
    return references


def _resolve_yaml_file(
    reference: str,
    config_dir: Path
) -> Path | None:
    """Cherche un fichier .yaml référencé par la configuration."""

    search_paths = [p for p in config_dir.glob('*') if p.is_dir()
    ]
    for path in search_paths:
        path = path / reference
        if path.exists():
            return path
    return None


def collect_config_files(
    config_dir: Path,
    config: dict,
) -> list[Path]:
    """Liste les `.yaml` de l'expérience, dans l'ordre de copie.

    Sont retenus :
        - le `pipeline.yaml` lui-même ;
        - tous les `.yaml` référencés dans la configuration ;
    """
    sources = [config_dir / "pipeline.yaml"]

    # 2) les .yaml référencés dans la configuration
    for reference in collect_yaml_references(config):
        source = _resolve_yaml_file(reference, config_dir)
        if source is not None:
            sources.append(source)
    return sources


def copy_config_files(
    config_dir: Path,
    config: dict,
    results_dir: Path,
) -> list[Path]:
    """Copie dans `results_dir` tous les `.yaml` de l'expérience.

    Retourne la liste des fichiers copiés.
    """

    copied: list[Path] = []
    used_names: set[str] = set()

    for source in collect_config_files(config_dir, config):
        name = source.name
        destination = results_dir / "input_metadata" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def get_existing_version(df: pd.DataFrame, row: dict) -> str | None:
    """Return the version if a matching row exists, otherwise None."""

    mask = df[list(row.keys())].eq(pd.Series(row)).all(axis=1)

    if mask.any():
        return df.loc[mask, "version"].iloc[0]

    return None


# ============================================================
# TABLEAU RÉCAPITULATIF DES EXPÉRIENCES
# ============================================================
def describe_augmentation(config: dict) -> str:
    """Décrit l'augmentation de données de l'expérience (colonne CSV)."""
    augmentation = config.get("data_augmentation", {}) or {}
    params = augmentation.get("torch_transformation")
    training = config.get("training_parameters", {}) or {}

    names = []
    affine = params.get("affine") or {}
    affine_enabled = any(
        AUG_PARAMS.get(flag, True)
        for key, flag in (
            ("degrees", "rotation"),
            ("translate", "translation"),
            ("scale", "scaling"),
        )
        if key in affine
    )
    if affine_enabled:
        names.append("RandomAffine")
    jitter = (
        params.get("colorjitter")
        or params.get("colorjittering")
        or params.get("color_jitter")
        or {}
    )
    jitter_enabled = any(
        AUG_PARAMS.get(flag, True)
        for key, flag in (("saturation", "saturation"), ("hue", "hue"))
        if key in jitter
    )
    if jitter_enabled:
        names.append("ColorJitter")

    if not names:
        return "none"

    return "+".join(names) + f"_from_epoch_{training.get('aug_epoch', 1)}"


def update_experiment_df(
    experiments_path: Path,
    rows: list[dict],
) -> pd.DataFrame:
    """Ajoute les lignes d'une expérience au tableau récapitulatif.
    CHANGER POUR EMPECHER UN AJOUT SI MEME LIGNE ?
    """
    if experiments_path.exists():
        df = pd.read_csv(experiments_path)
    else:
        df = pd.DataFrame(columns=EXPERIMENT_COLUMNS)

    for column in EXPERIMENT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[EXPERIMENT_COLUMNS + [c for c in df.columns if c not in EXPERIMENT_COLUMNS]]

    if rows:
        new_df = pd.DataFrame(rows)
        df = pd.concat([df, new_df], ignore_index=True)

    df = df.sort_values(
        by=["version"],
        na_position="last",
    ).reset_index(drop=True)

    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(experiments_path, index=False)
    return df


# ============================================================
# ENTRAÎNEMENT
# ============================================================
def build_training_args(
    config: dict,
    config_path: Path,
    results_dir: Path,
) -> argparse.Namespace:
    """Construit les arguments du script d'entraînement à partir du YAML."""
    paths = config.get("paths", {}) or {}
    values = train_classifier.build_path_values(config)
    home_path = Path(config.get("home_path", "..")).expanduser()

    crop_path = train_classifier.resolve_path(
        home_path, paths.get("crop_path", "data/crops"), values
    )
    crop_file = train_classifier.resolve_path(
        home_path, paths.get("crop_file", "data/crops.csv"), values
    )
    split_file = train_classifier.resolve_path(
        home_path, paths.get("split_file", "data/splits.csv"), values
    )
    model_dir = train_classifier.resolve_path(
        home_path, paths.get("model_dir", "models"), values
    )

    args = argparse.Namespace(
        config=str(config_path),
        feature_extractor=None,
        last_layer=None,
        last_frozen_layer=None,
        crop_path=str(crop_path),
        crop_file=str(crop_file),
        split_file=str(split_file),
        out_dir=str(results_dir),
        model_dir=str(model_dir),
        batch_size=None,
        num_epoch=None,
        lr=None,
        aug_epoch=None,
        defreeze_epoch=None,
        early_stopping=None,
        num_workers=None,
        seed=None,
        pretrained=None,
        device=None,
        limit_runs=None,
        batch_norm_only=None,
        augmentation_params=None,
        aug_params=AUG_PARAMS,
        preprocessing_function=None,
        transform_params=None,
    )

    return train_classifier.resolve_args(args)


def check_inputs(args: argparse.Namespace) -> None:
    """Check that the crop directory and split/image DataFrames exist."""
    missing = []
    crop_path = getattr(args, "crop_path", None)
    if not crop_path or not Path(crop_path).exists():
        missing.append(f"crop_path: {crop_path}")

    for name in ("images_df", "splits_df"):
        value = getattr(args, name, None)
        if value is None or value.empty:
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            "Missing training inputs:\n  - " + "\n  - ".join(missing)
        )


# ============================================================
# MAIN
# ============================================================
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="pipeline d'entraînement (configuration YAML)"
    )
    parser.add_argument(
        "--config", "-c", default="config/pipeline.yaml",
        help="fichier de configuration du pipeline",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="prépare le dossier de résultats sans entraîner",
    )

    arg = parser.parse_args(argv)

    config_dir = Path(arg.config).resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"Configuration introuvable : {config_dir}")

    config = load_config(config_dir)

    home_dir = Path(config['home_dir'])
    split_dir = Path(config['home_dir']) / config['split_dir']
    split_versions = config["split_versions"]
    frame_database = config["frame_database"]
    dataset_date = config["dataset_date"]
    paths = config.get("paths", {}) or {}
    results_root = train_classifier.resolve_path(
        home_dir,
        paths.get("output", "results"),
        train_classifier.build_path_values(config),
    )

    version, results_dir = next_results_dir(results_root)

    print(f"Configuration    : {config_dir}")
    print(f"Dossier résultats: {results_dir}")

    # ------------------------------------------------------------
    # 1) COPIE DES FICHIERS .yaml
    # ------------------------------------------------------------
    print("\nCopie des fichiers de configuration :")
    if arg.dry_run:
        for source in collect_config_files(config_dir, config):
            print(f"  - {source.name}")
        print("\n[dry-run] aucun dossier créé, entraînement non lancé.")
        return

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(
            f"Impossible de créer le dossier de résultats '{results_dir}'.\n"
            f"Vérifiez 'home_dir' ({home_dir}) et son accessibilité "
            f"en écriture.\nDétail : {error}"
        ) from error

    copied = copy_config_files(config_dir, config, results_dir)
    for path in copied:
        print(f"  - {path.name}")
    crops_dir = Path(config["home_dir"]) / config["crops_dir"]
    # DETECTIONS
    detection = config["detection"]
    detection['crops_dir'] = config["crops_dir"]
    detection['home_dir'] = home_dir
    detection["split_dir"] = Path(config["home_dir"]) / config["split_dir"]
    detection['frame_database'] = frame_database
    detection['video_database'] = config["video_database"]
    print("Detections")
    import detect_and_track as detect
    for model in DETECT_PARAMS:
        detection['model_name'] = model
        detection['conf'] = DETECT_PARAMS[model]["conf"]
        detection['labels_to_keep'] = DETECT_PARAMS[model]["labels_to_keep"]
        print(f"Model : {model}")
        print(f"Labels : {detection['labels_to_keep']}")
        #detect.main(detection)

    # DETECTIONS FILTERING AND SPLIT
    # FIRST CHECK THAT NO SPLIT WITH SAME FILTERING AND SPLIT PARAMETERS WAS DONE
    filters = config['filter']
    split_config = config["split"]
    all_detections = []
    for model in DETECT_PARAMS:
        for label in DETECT_PARAMS[model]["labels_to_keep"]:
            all_detections.append(label)

    filters_metadata = {
        f"{key1}_{key2}": value if key1 in all_detections else None
        for key1, sub_dict in filters.items()
        for key2, value in sub_dict.items()
    }
    split_metadata = split_config | filters_metadata
    if os.path.exists(split_dir / split_versions):
        split_metadata_df = pd.read_csv(split_dir / split_versions)
        split_version = get_existing_version(split_metadata_df, split_metadata)
    else:
        split_version = None

    #SPLIT DETECTIONS
    if split_version:
        splits_df = pd.read_csv(split_dir  / f"{split_version}_splits.csv")
        images_df = pd.read_csv(split_dir / f"{split_version}_selected_images.csv")

    else:
        # FIRST CONFIDENCE SCORE AND AREA, THEN SCRIPTS FOR COMPLEX FILTERS
        selected_dfs = []
        for model in DETECT_PARAMS:
            for label in DETECT_PARAMS[model]["labels_to_keep"]:
                filter = filters[label]
                conf = filter["conf"]
                min_area = filter["min_area"]
                min_ratio = filter["min_ratio"]
                max_ratio = filter["max_ratio"]
                crops_file = crops_dir / f"{model.split(".")[0]}.csv"
                df = pd.read_csv(crops_file)
                df = df[~df.indiv.isna()]
                df = df[(df.conf >= conf) & (df.area >= min_area)]
                if min_ratio:
                    df = df[df.ratio >= min_ratio]
                if max_ratio:
                    df = df[df.ratio <= max_ratio]
                script_names = filter["script"]
                if script_names:
                    for script_name in script_names:
                        if script_name:
                            filter_script = importlib.import_module(
                                f"detection_filters.{script_name}"
                            )
                            df = filter_script.main(df, crops_dir, model)
                df['label'] = df["indiv"] + "_" + label  # vrai séparation gauche droite pour oreille?
                selected_dfs.append(df)
        selected_df = pd.concat(selected_dfs)
        #GENERATE SPLIT OVER ALL DETECTIONS
        split_config["video_database"] = Path(config["home_dir"]) / config["video_database"]
        import split as split
        split_config['images_df'] = selected_df
        split_config["metadata"] = split_metadata
        split_config["split_versions"] = split_dir / split_versions
        splits_df, images_df = split.main(split_config)
        print(splits_df.head())
        print(images_df.head())
    splits_df.to_csv(results_dir / "input_metadata" / f"{split_version}_splits.csv")
    images_df.to_csv(results_dir / "input_metadata" / f"{split_version}_selected_images.csv")
    """
    # ------------------------------------------------------------
    # 2) TRAINING
    # ------------------------------------------------------------
    args = build_training_args(config, config_dir, results_dir)

    # The training script consumes the split/image DataFrames directly.
    args.crop_path = str(crops_dir)
    args.splits_df = splits_df
    args.images_df = images_df

    check_inputs(args)

    summary_df = train_classifier.run_training(args)
    if summary_df.empty:
        return
    summary_df['train_weights'] = (
        summary_df.num_train / summary_df.num_train.sum()
    )
    summary_df['val_weights'] = summary_df.num_val / summary_df.num_val.sum()
    train_std = summary_df.train_acc.std(axis=0)
    val_std = summary_df.val_acc.std(axis=0)
    mean_train_acc_over_runs = (
        (summary_df['train_acc'] * summary_df['train_acc']).mean()
    )
    mean_val_acc_over_runs = (
        (summary_df['val_acc'] * summary_df['val_acc']).mean()
    )

    # ------------------------------------------------------------
    # 3) TABLEAU RÉCAPITULATIF DES EXPÉRIENCES
    # ------------------------------------------------------------
    row ={
            "version": version,
            "dataset_date": config.get("dataset_date", ""),
            "detection_model": config.get("detection_model", ""),
            "split_method": (
                (config.get("split", {}) or {}).get("split_method", "")
            ),
            "architecture": args.feature_extractor,
            "data_augmentation": describe_augmentation(config),
            "classification_head": "ClassificationHead",
            "mean_train_acc_over_runs": mean_train_acc_over_runs,
            "mean_val_acc_over_run": mean_val_acc_over_runs,
            "train_acc_std": train_std,
            "val_acc_std": val_std,
        }

    experiments_path = results_root / EXPERIMENTS_CSV_NAME
    df = update_experiment_df(experiments_path, [row])
    print(f"\nTableau des expériences : {experiments_path}")
    print(df.to_string(index=False))"""


if __name__ == "__main__":
    main()
