"""
Récapitule mean_accuracy / std_accuracy par (transform, stop) pour les
résultats de 100_runs_splits_transforms.

Pour chaque transform du répertoire :
  - stop "no stop" : calculé depuis val_pred.csv
  - stop "X%"      : lu depuis X%/summary_metrics.csv s'il existe,
                     sinon calculé depuis val_pred_min_acc_X.0.csv

La formule de calcul reproduit exactement analyze_splits.py :
    correct = (pred_label == indiv_name)
    per_run_acc = groupby("run")["correct"].mean() * 100
    mean_accuracy = per_run_acc.mean()
    std_accuracy  = per_run_acc.std()   # ddof=1 (pandas par défaut)

Sortie : un CSV avec les colonnes
    transform, stop, mean_accuracy, std_accuracy
"""

import pandas as pd
from pathlib import Path

# ============================================================
# PARAMETRES
# ============================================================
VERSION = "v9"
CROP_TYPE = "Ear"
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers/")
TRANSFORMS_DIR = RESULTS_PATH / VERSION / CROP_TYPE / "100_runs_splits_aug_from_epoch5"

STOPS = ["no stop", "2%", "5%", "10%"]


def compute_from_val_pred(val_pred_path: Path) -> tuple[float, float]:
    """Calcule (mean_accuracy, std_accuracy) depuis un fichier val_pred."""
    vp = pd.read_csv(val_pred_path)
    vp["correct"] = vp["pred_label"] == vp["indiv_name"]
    per_run_acc = vp.groupby("run")["correct"].mean() * 100
    return float(per_run_acc.mean()), float(per_run_acc.std())


def read_summary(summary_path: Path) -> tuple[float, float]:
    """Lit (mean_accuracy, std_accuracy) depuis summary_metrics.csv."""
    df = pd.read_csv(summary_path).set_index("metric")["value"]
    return float(df["mean_accuracy"]), float(df["std_accuracy"])


def main() -> None:
    rows = []
    transform_dirs = sorted(p for p in TRANSFORMS_DIR.iterdir() if p.is_dir())

    for transform in transform_dirs:
        for stop in STOPS:
            if stop == "no stop":
                summary_path = None
                val_pred_path = transform / "val_pred.csv"
            else:
                summary_path = transform / stop / "summary_metrics.csv"
                stop_num = stop.rstrip("%")
                val_pred_path = transform / f"val_pred_min_acc_{stop_num}.0.csv"

            if summary_path is not None and summary_path.exists():
                mean_acc, std_acc = read_summary(summary_path)
            elif val_pred_path.exists():
                mean_acc, std_acc = compute_from_val_pred(val_pred_path)
            else:
                print(
                    f"[warning] {transform.name}/{stop} : ni summary_metrics.csv "
                    f"ni {val_pred_path.name} — ligne ignorée"
                )
                continue

            rows.append(
                {
                    "transform": transform.name,
                    "stop": stop,
                    "mean_accuracy": round(mean_acc, 2),
                    "std_accuracy": round(std_acc, 2),
                }
            )

    out_df = pd.DataFrame(rows, columns=["transform", "stop", "mean_accuracy", "std_accuracy"])
    out_path = TRANSFORMS_DIR / "summary_transform_stop.csv"
    out_df.to_csv(out_path, index=False, float_format="%.2f")

    print(out_df.to_string(index=False))
    print(f"\nCSV écrit dans : {out_path}")


if __name__ == "__main__":
    main()