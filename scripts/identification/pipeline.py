"""
TRAINING PIPELINE

Reads a YAML configuration file (default: `config/pipeline.yaml`) and:

1. creates a versioned results folder (`<results>/v1`, `v2`, ...);
2. copies ALL `.yaml` files into it: those referenced in the configuration,
   those present in the configuration folder, and `pipeline.yaml` itself;
3. runs the modular training script `scripts/train_classifier.py`
   (feature extraction, optional fine-tuning, ClassificationHead head,
   RandomAffine + ColorJitter augmentation, one run per split);
4. runs `cosine_similarity.py` on the backbone of the median model to
   produce `val_pred.csv` (cosine neighbours + majority vote);
5. fills in the experiment summary table (`all_experiments.csv`) with one
   row per run (train_acc, val_acc, ...).

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
import train_classifier, detect_and_track, cosine_similarity
import os
import numpy as np
import pdb

#DETECTION CHOICES
DETECT_CHOICES = ['ear_seg'] #'ear_seg','body_detect','ear_detect', 'head_detect'
MODE = ["Day"]
RERUN_SOME_DETECT = False
TEST = True
TRAIN_CLASSIFIER = True
FINETUNE = False
ASSYMETRY = False #if True different labels for right and left ear

#ARCHITECTURE CHOICE
ARCHITECTURE = "ResNet50" # MiewID | MegaDescriptor | ResNet152 | ResNet50 | None if you choose last and frozen layer in pipeline.yaml

#DEFAULT PARAMETERS FOR CHOICES
REPO_ROOT = Path.cwd()
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
    "detections",
    "mode",
    "architecture",
    "data_augmentation",
]

EXPERIMENTS_CSV_NAME = "all_experiments.csv"

CROPS_DTYPES = {
    "video": str,
    "image": str,
    "track_id": int,
    "detection": str,
    "area": int,
    "ratio": float,
    "indiv": str,
    "frame": int,
    "conf": float,
    "x1": int,
    "x2": int,
    "y1": int,
    "y2": int,
}

def load_config(config_path: Path) -> dict:
    """Load the pipeline configuration file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


# ============================================================
# RESULTS FOLDER
# ============================================================
def next_results_dir(results_root: Path) -> tuple[str, Path]:
    """Determine the next versioned folder (`v1`, `v2`, ...).

    Returns (version, results folder).
    """

    def contains_any_file(folder):
        """Return True if the folder or any subfolder contains at least one file."""
        return any((p.is_file() and p.suffix != ".yaml") for p in Path(folder).glob("*"))
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


def get_existing_version(df: pd.DataFrame, row: dict) -> str | None:
    """Return the version if a matching row exists, otherwise None."""

    def normalize(value):
        if isinstance(value, list):
            return str(value)
        if value is None or (np.isscalar(value) and pd.isna(value)):
            return None
        return value

    columns = list(row.keys())

    row_values = {key: normalize(value) for key, value in row.items()}

    for idx, df_row in df.iterrows():
        differences = {}

        for col in columns:
            df_value = normalize(df_row[col])
            row_value = row_values[col]

            if df_value != row_value:
                differences[col] = {
                    "df": repr(df_value),
                    "row": repr(row_value),
                    "df_type": type(df_value).__name__,
                    "row_type": type(row_value).__name__,
                }

        if not differences:
            return df_row["version"]

    return None


# ============================================================
# EXPERIMENT SUMMARY TABLE
# ============================================================
def describe_augmentation(config: dict) -> str:
    """Describe the experiment data augmentation (CSV column)."""
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
    """Add an experiment's rows to the summary table.
    CHANGE TO PREVENT ADDING A DUPLICATE ROW?
    """
    if experiments_path.exists():
        df = pd.read_csv(experiments_path)
    else:
        print(f"First experiments saved in {experiments_path.name}")
        df = pd.DataFrame(columns=EXPERIMENT_COLUMNS)

    for column in EXPERIMENT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[EXPERIMENT_COLUMNS + [c for c in df.columns if c not in EXPERIMENT_COLUMNS]]

    if rows:
        new_df = pd.DataFrame(rows)
        valid_columns = new_df.columns[~new_df.isna().all()]
        df = pd.concat([df, new_df[valid_columns]], ignore_index=True)
        
    df = df.sort_values(
        by=["version"],
        na_position="last",
    ).reset_index(drop=True)

    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(experiments_path, index=False)
    return df


# ============================================================
# TRAINING
# ============================================================
def build_training_args(
    config: dict,
    config_path: Path,
    results_dir: Path,
) -> argparse.Namespace:
    home_dir = Path(config["home_dir"])
    crop_path = home_dir / config["crops_dir"]

    args = argparse.Namespace(
        config=str(config_path),
        feature_extractor=ARCHITECTURE,
        finetune=FINETUNE,
        last_layer=None,
        last_frozen_layer=None,
        crop_path=str(crop_path),
        out_dir=str(results_dir),
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
        transform_params=None
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


def build_untrained_summary(args: argparse.Namespace) -> pd.DataFrame:
    """Build a per-run summary without any training metric.

    Used when ``TRAIN_CLASSIFIER`` is false: one row per run with split sizes and
    detections, but columns coming from training (accuracy, loss, stop
    epoch, duration, ...) are left empty.
    """
    splits_df = getattr(args, "splits_df", None)
    images_df = getattr(args, "images_df", None)

    if splits_df is None or images_df is None:
        raise ValueError(
            "'splits_df' et 'images_df' sont requis pour construire le "
            "résumé sans entraînement."
        )

    runs = sorted(splits_df["run"].dropna().unique().tolist())
    if getattr(args, "limit_runs", None) is not None:
        runs = runs[: args.limit_runs]

    rows = []
    for run in runs:
        df = train_classifier.build_run_dataframe(images_df, splits_df, run)
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]

        rows.append(
            {
                "run": run,
                "detections": ", ".join(
                    sorted(df["detection"].unique().tolist())
                ),
                "num_classes": int(df["label"].nunique()),
                "num_train": len(train_df),
                "num_val": len(val_df),
                # Training metrics: empty (no training).
                "train_acc": float("nan"),
                "val_acc": float("nan"),
                "best_val_acc": float("nan"),
                "best_epoch": float("nan"),
                "train_loss": float("nan"),
                "val_loss": float("nan"),
                "early_stop_epoch": float("nan"),
                "duration_s": float("nan"),
            }
        )

    return pd.DataFrame(rows)


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

    config_path = Path(arg.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {config_path}")

    config = load_config(config_path)

    home_dir = Path(config['home_dir'])
    split_dir = Path(config['home_dir']) / config['split_dir']
    split_versions = config["split_versions"]
    frame_database = config["frame_database"]
    dataset_date = config["dataset_date"]
    results_root = home_dir / "results"

    version, results_dir = next_results_dir(results_root)

    print(f"Configuration    : {config_path}")
    print(f"Dossier résultats: {results_dir}")

    # ------------------------------------------------------------
    # 1) COPYING .yaml FILES
    # ------------------------------------------------------------
    print("\nCopie du fichier de configuration :")

    try:
        (results_dir / "input_metadata").mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, results_dir / "input_metadata" / "pipeline.yaml")
    except OSError as error:
        raise OSError(
            f"Impossible de créer le dossier de résultats '{results_dir}'.\n"
            f"Vérifiez 'home_dir' ({home_dir}) et son accessibilité "
            f"en écriture.\nDétail : {error}"
        ) from error

    crops_dir = Path(config["home_dir"]) / config["crops_dir"]
    video_database_df = pd.read_csv(home_dir / config["video_database"]) #REPO_ROOT = Path.cwd() / "database"
    video_database_df = video_database_df[video_database_df["mode"].isin(MODE)]
    # DETECTIONS
    detection = config["detection"]
    detection['crops_dir'] = config["crops_dir"]
    detection['home_dir'] = home_dir
    detection["split_dir"] = Path(config["home_dir"]) / config["split_dir"]
    detection['frame_database'] = frame_database
    detection['video_database'] = video_database_df
    detection['rerun'] = RERUN_SOME_DETECT
    detection['test'] = TEST
    print("Detections")
    import detect_and_track as detect
    for model in DETECT_PARAMS:
        detection['model_name'] = model
        detection['conf'] = DETECT_PARAMS[model]["conf"]
        detection['labels_to_keep'] = DETECT_PARAMS[model]["labels_to_keep"]
        print(f"Model : {model}")
        print(f"Labels : {detection['labels_to_keep']}")
        detect.main(detection)

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
                crops_file = crops_dir / f"{model.split('.')[0]}.csv"
                df = pd.read_csv(crops_file,dtype=CROPS_DTYPES)
                df = df[~df.indiv.isna()]
                if conf:
                    df = df[(df.conf >= conf)]
                if min_area:
                    df = df[(df.area >= min_area)]
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
                if ASSYMETRY:
                    df['label'] = df["indiv"] + "_" + df["detection"]
                else:
                    df['label'] = df["indiv"] + "_" + label  #same label for right and left ear
                selected_dfs.append(df)
        selected_df = pd.concat(selected_dfs)
        print(len(selected_df),flush=True)
        #GENERATE SPLIT OVER ALL DETECTIONS
        print("GENERATE NEW SPLIT")
        split_config["video_database"] = video_database_df
        import split as split
        split_config['images_df'] = selected_df
        split_config["metadata"] = split_metadata
        split_config["split_versions"] = split_dir / split_versions
        splits_df, images_df, split_version = split.main(split_config)
        print(splits_df.head())
        print(images_df.head())
    if TEST:
        images_df = images_df.drop_duplicates(
            subset=['event', 'label'],
            keep='last'
        ).reset_index(drop=True)
        images_df = images_df[images_df.event.isin([i for i in range(10)])]
        splits_df = splits_df[splits_df.event.isin([i for i in range(10)])]
        label_to_split = images_df.merge(splits_df, how='inner', on='event').groupby('label').agg(set).to_dict()
        for label in label_to_split:
            if label_to_split[label]==set(['val']):
                images_df = images_df[~images_df.label.isin([label])]
                

    splits_df.to_csv(results_dir / "input_metadata" / f"{split_version}_splits.csv")
    images_df.to_csv(results_dir / "input_metadata" / f"{split_version}_selected_images.csv")


    # ------------------------------------------------------------
    # 2) TRAINING
    # ------------------------------------------------------------
    args = build_training_args(config, config_path, results_dir)

    # The training script consumes the split/image DataFrames directly.
    args.crop_path = str(crops_dir)
    args.splits_df = splits_df
    args.images_df = images_df
    
    check_inputs(args)
    if TRAIN_CLASSIFIER:
        summary_df = train_classifier.run_training(args)
        if summary_df.empty:
            return
    else:
        summary_df = build_untrained_summary(args)
            
    cosine_df = cosine_similarity.run_similarity(args)
    if isinstance(cosine_df, pd.DataFrame) and not cosine_df.empty:
        if "run" in cosine_df.columns and "cosine_acc" in cosine_df.columns:
            summary_df = summary_df.merge(
                cosine_df[["run", "cosine_acc"]], on="run", how="left"
            )
            cosine_path = Path(args.out_dir) / "runs_summary.csv"
            summary_df.to_csv(cosine_path, index=False)
            print(f"Runs summary updated with cosine_acc: {cosine_path}")

    summary_df['train_weights'] = (
        summary_df.num_train / summary_df.num_train.sum()
    )
    summary_df['val_weights'] = summary_df.num_val / summary_df.num_val.sum()
    cos_val_std = (
        summary_df.cosine_acc.std(axis=0)
        if "cosine_acc" in summary_df.columns
        else None
    )

    if TRAIN_CLASSIFIER:
        train_std = summary_df.train_acc.std(axis=0)
        val_std = summary_df.val_acc.std(axis=0)
        mean_train_acc_over_runs = (
            (summary_df['train_acc'] * summary_df['train_weights']).mean()
        )
        mean_val_acc_over_runs = (
            (summary_df['val_acc'] * summary_df['val_weights']).mean()
        )
    else:
        # No training: experiment columns left empty.
        train_std = None
        val_std = None
        mean_train_acc_over_runs = None
        mean_val_acc_over_runs = None

    mean_cosine_acc_over_run = None
    if "cosine_acc" in summary_df.columns and summary_df["cosine_acc"].notna().any():
        mean_cosine_acc_over_run = float(
            (summary_df["cosine_acc"].astype(float) * summary_df["val_weights"]).mean()
        )

    # ------------------------------------------------------------
    # 3) EXPERIMENT SUMMARY TABLE
    # ------------------------------------------------------------
    row ={
            "version": version,
            "dataset_date": config.get("dataset_date", ""),
            "detections": DETECT_CHOICES,
            "mode": MODE,
            "split_event": config['split']['event'],
            "data_augmentation": describe_augmentation(config),
            "architecture": args.feature_extractor,
            "finetune": FINETUNE,
            "last_fozen_layer": config["feature_extractor"]["last_frozen_layer"],
            "last_layer": config["feature_extractor"]["last_layer"],
            "mean_train_acc_over_runs": mean_train_acc_over_runs,
            "train_acc_std": train_std,
            "mean_val_acc_over_run": mean_val_acc_over_runs,
            "val_acc_std": val_std,
            "mean_cosine_acc_over_run": mean_cosine_acc_over_run,
            "val_cosine_acc_std": cos_val_std,

        }

    experiments_path = results_root / EXPERIMENTS_CSV_NAME
    df = update_experiment_df(experiments_path, [row])
    print(f"\nTableau des expériences : {experiments_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
