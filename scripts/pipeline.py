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
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

# ============================================================
# BOOTSTRAP : le pipeline doit pouvoir être importé comme
# `scripts.pipeline` (racine du dépôt dans sys.path)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path('/pbs/home/a/ameurisse')

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.identification import train_classifier

# ============================================================
# COLONNES DU TABLEAU RÉCAPITULATIF DES EXPÉRIENCES
# ============================================================
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


# ============================================================
# CONFIGURATION
# ============================================================
def load_config(config_path: Path) -> dict:
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
        return any(p.is_file() for p in Path(folder).rglob("*"))
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
    results_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    used_names: set[str] = set()

    for source in collect_config_files(config_dir, config):
        name = source.name
        destination = results_dir / name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


# ============================================================
# TABLEAU RÉCAPITULATIF DES EXPÉRIENCES
# ============================================================
def describe_augmentation(config: dict) -> str:
    """Décrit l'augmentation de données de l'expérience (colonne CSV)."""
    augmentation = config.get("data_augmentation", {}) or {}
    params = (
        (augmentation.get("data_transformation", {}) or {}).get("params", {})
        or {}
    )
    training = config.get("training_parameters", {}) or {}

    names = []
    if params.get("affine"):
        names.append("RandomAffine")
    if params.get("colorjitter"):
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
    home_path = Path(config.get("home_path", ".")).expanduser()

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
        augmentation_params=None,
        transform_function=None,
        transform_params=None,
    )

    return train_classifier.resolve_args(args)


def check_inputs(args: argparse.Namespace) -> None:
    """Vérifie que les fichiers d'entrée existent."""
    missing = []
    for name in ("crop_path", "crop_file", "split_file"):
        path = Path(getattr(args, name))
        if not path.exists():
            missing.append(f"{name} : {path}")
    if missing:
        raise FileNotFoundError(
            "Fichiers d'entrée introuvables :\n  - "
            + "\n  - ".join(missing)
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

    home_dir = config['home_dir']
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

    # DETECTION
    detection = config['detection']
    detection['home_dir'] = home_dir
    detection['frame_database'] = frame_database
    detection['video_database'] = config["video_database"]
    import scripts.identification.detect as detect
    detect.main(detection)

    if not os.path.exists(split_file):
        import scripts.create_split as create_split
        create_split.main(split_metadata)


    # ------------------------------------------------------------
    # 2) ENTRAÎNEMENT
    # ------------------------------------------------------------
    args = build_training_args(config, config_path, results_dir)

    check_inputs(args)

    summary_df = train_classifier.run_training(args)
    summary_df['train_weights'] = summary_df.num_train / summary_df.num_train.sum()
    summary_df['val_weights'] = summary_df.num_val / summary_df.num_val.sum()
    train_std = summary_df.train_acc.std(axis=0)
    val_std = summary_df.val_acc.std(axis=0)
    mean_train_acc_over_runs = (summary_df['train_acc']*summary_df['train_acc']).mean()
    mean_val_acc_over_runs = (summary_df['val_acc'] * summary_df['val_acc']).mean()

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
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
