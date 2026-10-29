from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
def load_config(config_dir: Path) -> dict:
    """Charge le fichier de configuration du pipeline."""
    config_path = config_dir / "pipeline.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
from scripts.identification import train_classifier, detect_and_track


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
    results_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    used_names: set[str] = set()

    for source in collect_config_files(config_dir, config):
        name = source.name
        destination = results_dir / name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied

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
    import scripts.identification.detect_and_track as detect
    detect.main(detection)

if __name__ == "__main__":
    main()