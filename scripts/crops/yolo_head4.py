"""
YOLO CLI - Détection/segmentation par dossier d'individus
Usage: python yolo_cli.py --config config.yaml

Structure d'entrée attendue :
  input/
  ├── individu_01/
  │   ├── video1.mp4
  │   └── video2.mp4
  └── individu_02/
      └── video1.mp4

Structure de sortie générée :
  output/
  ├── individu_01/
  │   ├── videos/
  │   │   ├── video1.mp4
  │   │   └── video2.mp4
  │   ├── crops/
  │   │   ├── id_0001/
  │   │   │   ├── frame_000000_id_0001.jpg
  │   │   │   └── ...
  │   │   └── id_0002/
  │   └── metadata.csv
  └── individu_02/
      └── ...
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Valeurs par défaut
    defaults = {
        "confidence":    0.80,
        "fps":           2.0,
        "tracking":      True,
        "tracker":       "bytetrack.yaml",
        "save_crops":    True,
        "save_video":    True,
        "save_bounding_box": True,
        "show":          False,
        "min_crop_width":  30,
        "min_crop_height": 50,
        "model":         "./models/head_detection.pt",
        "input":         "../data/beach_individuals",
        "output":        "./resultats",
    }
    for key, val in defaults.items():
        cfg.setdefault(key, val)

    return cfg


def validate_config(cfg: dict) -> None:
    errors = []

    if not cfg.get("model"):
        errors.append("'model' est requis")
    elif not cfg["model"].lower().endswith(".pt"):
        errors.append("'model' doit être un fichier .pt")
    elif not os.path.exists(cfg["model"]):
        errors.append(f"Modèle introuvable : {cfg['model']}")

    if not cfg.get("input"):
        errors.append("'input' est requis")
    elif not os.path.isdir(cfg["input"]):
        errors.append(f"Dossier d'entrée introuvable : {cfg['input']}")

    if not cfg.get("output"):
        errors.append("'output' est requis")

    if not 0.0 <= float(cfg["confidence"]) <= 1.0:
        errors.append("'confidence' doit être entre 0.0 et 1.0")

    if errors:
        for e in errors:
            log(f"Erreur de configuration : {e}")
        sys.exit(1)


def get_vid_stride(video_path: str, target_fps: float) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        log(f"Impossible de lire le FPS de {video_path}, stride = 1")
        return 1
    stride = max(1, round(fps / target_fps))
    log(f"  FPS source : {fps:.1f} → stride : {stride} (~{fps/stride:.1f} fps traités)")
    return stride


def collect_individual_videos(input_dir: str) -> dict[str, list[str]]:
    """
    Retourne un dict { nom_individu: [liste de chemins vidéo] }
    en parcourant les sous-dossiers du dossier d'entrée.
    """
    individuals = {}
    input_path = Path(input_dir)

    for subdir in sorted(input_path.iterdir()):
        if not subdir.is_dir():
            continue
        individuals[subdir.name] = {}
        for mode in sorted(subdir.iterdir()):
            if mode.name in ['Night','Day']:
                videos = sorted([
                    str(f) for f in mode.iterdir()
                    if f.suffix.lower() in VIDEO_EXTENSIONS
                ])
                if videos:
                    individuals[subdir.name][mode.name] = videos
                else:
                    log(f"Aucune vidéo trouvée dans : {subdir.name}, ignoré.")

    return individuals

# ---------------------------------------------------------------------------
# VideoWriter H.264 (sans audio)
# ---------------------------------------------------------------------------

def open_video_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """
    Ouvre un VideoWriter H.264 sans audio.
    Essaie successivement avc1, H264, mp4v selon la disponibilité OpenCV.
    """
    for fourcc_str in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            log(f"  Codec vidéo : {fourcc_str}")
            return writer
    raise RuntimeError("Aucun codec H.264/mp4v disponible dans OpenCV")


# ---------------------------------------------------------------------------
# Inférence sur une vidéo
# ---------------------------------------------------------------------------

def get_source_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25.0


def process_video(
    model: YOLO,
    video_path: str,
    individual_name: str,
    mode: str,
    cfg: dict
) -> None:
    video_stem = Path(video_path).stem
    video_name = Path(video_path).name
    stride = get_vid_stride(video_path, cfg["fps"])
    source_fps = get_source_fps(video_path)
    # FPS réel de la vidéo de sortie = fps traités (pas le fps source)
    output_fps = source_fps / stride
    common_kwargs = dict(
        source=video_path,
        conf=cfg["confidence"],
        show=cfg["show"],
        verbose=False,
        stream=True,
        vid_stride=stride,
        save_crop=cfg["save_crops"],
        save=cfg["save_video"],
        save_txt=cfg["save_bounding_box"],
        project=Path(cfg["output"]).resolve(),
        name=str(Path(individual_name) / Path(mode)),
        exist_ok=True
    )
    if cfg["tracking"]:
        results = model.track(**common_kwargs, persist=True, tracker=cfg["tracker"])
    else:
        results = model(**common_kwargs)
    for _ in results:
        pass

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_all(model: YOLO, individuals: dict[str, list[str]], cfg: dict) -> None:
    output_root = Path(cfg["output"])
    total_individuals = len(individuals)

    for idx, (individual_name, modes) in enumerate(individuals.items(), start=1):
        for mode, videos in modes.items():
            log(f"[{idx}] Individu : {individual_name} Mode : {mode} ({len(videos)} vidéo(s))")
            for v_idx, video_path in enumerate(videos, start=1):
                log(f"  [{v_idx}/{len(videos)}] {Path(video_path).name}")
                try:
                    process_video(model, video_path, individual_name, mode, cfg)
                except Exception as e:
                    log(f"  Erreur sur {Path(video_path).name} : {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO CLI – Traitement par individu")
    parser.add_argument("--config", "-c", required=True, help="Chemin vers le fichier config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Fichier de configuration introuvable : {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)
    validate_config(cfg)

    log(f"Configuration chargée : {args.config}")
    log(f"Modèle         : {cfg['model']}")
    log(f"Entrée         : {cfg['input']}")
    log(f"Sortie         : {cfg['output']}")
    log(f"Confiance      : {cfg['confidence']} | FPS cible : {cfg['fps']}")
    log(f"Tracking       : {'Oui' if cfg['tracking'] else 'Non'} ({cfg['tracker']})")
    log(f"Save vidéo     : {'Oui' if cfg['save_video'] else 'Non'}")
    log(f"Save crops     : {'Oui' if cfg['save_crops'] else 'Non'} "
        f"(min {cfg['min_crop_width']}x{cfg['min_crop_height']}px)")

    individuals = collect_individual_videos(cfg["input"])
    if not individuals:
        log("Aucun individu trouvé dans le dossier d'entrée. Abandon.")
        sys.exit(1)
    log(f"Individus trouvés : {len(individuals)}")

    log(f"Chargement du modèle : {cfg['model']}")
    try:
        model = YOLO(cfg["model"])
        log("Modèle chargé avec succès.")
    except Exception as e:
        log(f"Impossible de charger le modèle : {e}")
        sys.exit(1)

    process_all(model, individuals, cfg)
    log("Traitement terminé.")


if __name__ == "__main__":
    main()
