"""
Visualisations et statistiques globales sur tous les runs / splits à
partir de val_pred.csv (produit par scripts/train_10_runs_splits.py).

Le fichier val_pred.csv contient une colonne `run` qui identifie le numéro
du run, plus toutes les prédictions de validation de chaque run
(indiv_name, pred_label, camera, mode, video, entry, split, crop_id,
logits, pred, conf_pred_label, margin, entropy).

Outputs écrits dans le même dossier que val_pred.csv :
  - accuracies_per_split.csv              : accuracy de validation par run,
                                            avec une colonne par couple
                                            (entry, indiv_name)
  - summary_metrics.csv                   : accuracy moyenne / écart-type
  - per_individual_recall.csv             : recall moyen / écart-type par individu
  - per_indiv_entry_accuracy_splits.csv   : accuracy par (individu, entry) et par run
  - confusion_mean.csv / confusion_std.csv           (comptes)
  - confusion_mean_normalized.csv / confusion_std_normalized.csv
  - confusion_mean.png / confusion_std.png
  - confusion_mean_normalized.png / confusion_std_normalized.png
  - accuracy_distribution.png
  - per_individual_recall.png
  - per_indiv_entry_accuracy.png
  - confusion_matrices.npz                : matrices (moyenne/écart-type) pour réutilisation
"""

import numpy as np
import pandas as pd
import matplotlib
import pdb
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

# ============================================================
# PARAMETRES
# ============================================================
OUT_DIR_NAME = "100_runs_splits_aug_from_epoch5"
version = "v9"
crop_type = "Ear"
mode = "Day"
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers/")

OUT_DIR = RESULTS_PATH / version / crop_type / OUT_DIR_NAME
VAL_PRED_PATH = OUT_DIR.parent / "val_pred.csv"


# ============================================================
# HELPERS (couleurs par caméra, plots de matrice de confusion)
# ============================================================

def load_camera_of(vp: pd.DataFrame, class_names: list) -> dict:
    """Map each individual to its camera from the val_pred metadata."""
    camera_of = vp.groupby("indiv_name")["camera"].first()
    return {name: camera_of.get(name, "") for name in class_names}


def get_camera_colors(camera_names: list) -> dict:
    """Assign one color per camera."""
    cmap = plt.get_cmap("tab10")
    return {cam: cmap(i % 10) for i, cam in enumerate(camera_names)}


def color_tick_labels(ax, x_names: list, y_names: list, camera_of: dict, camera_colors: dict) -> None:
    for lbl, name in zip(ax.get_xticklabels(), x_names):
        lbl.set_rotation(45)
        lbl.set_ha("right")
        lbl.set_color(camera_colors.get(camera_of.get(name, ""), "black"))
    for lbl, name in zip(ax.get_yticklabels(), y_names):
        lbl.set_color(camera_colors.get(camera_of.get(name, ""), "black"))


def plot_confusion_matrix(
    cm,
    row_names,
    col_names,
    title,
    out_path,
    camera_of=None,
    camera_colors=None,
    fmt="d",
):
    # Figure plus large : la largeur suit le nombre de colonnes.
    n_rows, n_cols = np.asarray(cm).shape
    fig_w = max(20, n_cols * 0.85)
    fig_h = max(14, n_rows * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=col_names,
        yticklabels=row_names,
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(title)
    if camera_of is not None and camera_colors is not None:
        color_tick_labels(ax, col_names, row_names, camera_of, camera_colors)
    # Encadrer en noir les cases où l'individu réel (ligne) et l'individu
    # prédit (colonne) sont le même individu.
    for i, name in enumerate(row_names):
        if name in col_names:
            j = col_names.index(name)
            rect = patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)
    plt.subplots_adjust(left=0.20, right=0.85, bottom=0.20, top=0.90)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_normalized(
    cm,
    row_names,
    col_names,
    title,
    out_path,
    camera_of=None,
    camera_colors=None,
    police_size=8,
    already_normalized=False,
    show_counts=True,
):
    """Row-normalized confusion matrix with crop counts per individual and
    individual names colored by camera."""
    cm = np.asarray(cm, dtype=float)

    if already_normalized:
        cm_norm = cm
    else:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        valid = cm_sum.squeeze(axis=1) != 0
        cm_norm[valid] = cm[valid] / cm_sum[valid]

    crop_counts = cm.sum(axis=1).astype(int) #nombre de crops

    n_rows, n_cols = np.asarray(cm).shape
    fig_w = max(20, n_cols * 0.85)
    fig_h = max(14, n_rows * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=col_names,
        yticklabels=row_names,
        annot_kws={"size": police_size},
        ax=ax,
        cbar_kws={"pad": 0.15},
    )

    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(np.arange(len(row_names)) + 0.5)
    if show_counts:
        ax_right.set_yticklabels(
            [f"{name} ({count})" for name, count in zip(row_names, crop_counts)],
            rotation=0,
            fontsize=police_size,
        )
        ax_right.set_ylabel("Individu (#crops moyen)", rotation=270, labelpad=20)
    else:
        ax_right.set_yticklabels(
            row_names,
            rotation=0,
            fontsize=police_size,
        )

    if camera_of is not None and camera_colors is not None:
        color_tick_labels(ax, col_names, row_names, camera_of, camera_colors)
        for lbl, name in zip(ax_right.get_yticklabels(), row_names):
            lbl.set_color(camera_colors.get(camera_of.get(name, ""), "black"))

    for i, name in enumerate(row_names):
        if name in col_names:
            j = col_names.index(name)
            rect = patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)

    ax.set_title(title)
    plt.subplots_adjust(left=0.20, right=0.85, bottom=0.20, top=0.90)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not VAL_PRED_PATH.exists():
        raise FileNotFoundError(
            f"{VAL_PRED_PATH} introuvable. Lancer d'abord scripts/train_10_runs_splits.py"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vp = pd.read_csv(VAL_PRED_PATH)
    if "run" not in vp.columns:
        raise ValueError(
            "Colonne 'run' absente de val_pred.csv — relancer "
            "scripts/train_10_runs_splits.py"
        )

    vp["correct"] = vp["pred_label"] == vp["indiv_name"]

    runs = sorted(vp["run"].unique())
    NUM_RUNS = len(runs)
    print(f"\n{len(vp)} prédictions de validation, {NUM_RUNS} runs")

    # ============================================================
    # ACCURACY PAR RUN
    # ============================================================
    per_run_acc = vp.groupby("run")["correct"].mean() * 100
    mean_acc = vp['correct'].mean()*100 #accuracy moyenne sur tous les runs, par moyenne des accuracy des runs qui ne prends pas en compte le fait que certains runs ont plus de données de validation
    mean_run_acc = per_run_acc.mean()
    std_acc = per_run_acc.std()

    runs_df = pd.DataFrame({
        "run": per_run_acc.index,
        "val_accuracy": per_run_acc.values.round(2),
    }).sort_values("val_accuracy", ascending=False).reset_index(drop=True)

    # Accuracy par (run, entry, indiv_name) : une colonne par couple
    # (entrée, individu), ajoutée à accuracies_per_split.csv.
    pair_acc = (
        vp.groupby(["run", "entry", "indiv_name"])["correct"].mean() * 100
    )
    nb_crops_val = vp.groupby("run")["crop_id"].count()
    pair_df = pair_acc.unstack(level=["entry", "indiv_name"])
    pair_df = pair_df.round(2)
    pair_df.columns = [
        f"val_acc_{entry}_{indiv}"
        for entry, indiv in pair_df.columns
    ]
    pair_df = pair_df.reindex(columns=sorted(pair_df.columns))
    runs_df_full = runs_df.merge(
        pair_df, left_on="run", right_index=True, how="left"
    )
    runs_df_full = runs_df_full.merge(nb_crops_val,on="run")


    print("\n=================== ACCURACY PAR RUN (triés par accuracy) ===================")
    print(runs_df.to_string(index=False))
    print("\n=================== PERFORMANCE MOYENNE ===================")
    print(f"Accuracy moyenne sur {NUM_RUNS} runs : {mean_acc:.2f} %")
    print(f"Écart-type de l'accuracy            : {std_acc:.2f} %")

    # ============================================================
    # MATRICES DE CONFUSION PAR RUN (à partir des noms de classes)
    # ============================================================
    all_classes = sorted(set(vp["indiv_name"]) | set(vp["pred_label"]))

    label_to_idx = {name: idx for idx, name in enumerate(all_classes)}
    n_classes = len(all_classes)

    confusion_matrices = []
    for run in runs:
        sub = vp[vp["run"] == run]
        actual = sub["indiv_name"].map(label_to_idx).values
        predicted = sub["pred_label"].map(label_to_idx).values
        cm = confusion_matrix(
            actual,
            predicted,
            labels=np.arange(n_classes),
        )
        confusion_matrices.append(cm)

    # Individus présents dans la validation d'au moins un run : ce sont les
    # seules lignes gardées dans les matrices moyennes / écart-types.
    val_indivs_union = sorted(set(vp["indiv_name"].unique()))
    never_val = sorted(set(all_classes) - set(val_indivs_union))
    print(f"\nIndividus présents en validation dans au moins un run ({len(val_indivs_union)}):")
    print(f"  {val_indivs_union}")
    if never_val:
        print(f"Individus absents de toutes les validations ({len(never_val)}): {never_val}")
    val_union_idx = [all_classes.index(n) for n in val_indivs_union]
    n_val = len(val_indivs_union)

    stacked = np.stack(confusion_matrices)        # (N, C, C)
    stacked = stacked[:, val_union_idx, :]        # (N, n_val, C)

    # Pour chaque individu, on ne moyenne que sur les runs où il apparaît
    # en validation (sinon les runs sans lui diluent la matrice avec des 0).
    row_total = stacked.sum(axis=2)               # (N, n_val) crops par (run, individu)
    valid = row_total > 0                         # masque (N, n_val)

    # Matrice de confusion : pour chaque case (indiv_A réel -> indiv_B prédit),
    # moyenne/écart-type sur les runs où indiv_A est en validation.
    confusions = (
        vp.groupby(['indiv_name', 'pred_label', 'crop_id'])['run']
        .count()
        .reset_index()
    )

    vp_crop_nb_run = (
        vp.groupby('crop_id')['run']
        .count()
    )

    confusions['run'] = (
            confusions['run'] / confusions['crop_id'].map(vp_crop_nb_run)
    )
    confusions = confusions.groupby(['indiv_name', 'pred_label'])['run'].sum().reset_index()
    cm_pivot = confusions.pivot(index='indiv_name', columns='pred_label', values='run')
    mean_cm = cm_pivot.reindex(
        index=val_indivs_union, columns=all_classes, fill_value=0
    ).values

    # std_cm : même méthode que confusions — chaque crop vote pour
    # 1/nb_runs où il apparaît, mais on garde la dimension run pour
    # mesurer la variation entre runs.
    vp_frac = vp.assign(frac=1.0 / vp['crop_id'].map(vp_crop_nb_run))
    run_frac = (
        vp_frac.groupby(['run', 'indiv_name', 'pred_label'])['frac']
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=all_classes, fill_value=0)
    )
    frac_matrices = np.zeros((NUM_RUNS, n_val, n_classes))
    for k, run in enumerate(runs):
        block = run_frac.xs(run, level='run', drop_level=True)
        frac_matrices[k] = block.reindex(index=val_indivs_union, fill_value=0).values
    std_conf_normalized = (
        vp.groupby(['indiv_name', 'pred_label', 'run'])['crop_id']
        .count()
        .reset_index()
    )

    vp_run_nb_crop = (
        vp.groupby('run')['crop_id']
        .count()
    )
    #confusion normalisée par run -> ecart-type sur les runs
    std_conf_normalized['crop_id'] = (
            std_conf_normalized['crop_id'] / std_conf_normalized['run'].map(vp_run_nb_crop)
    )
    std_conf_normalized = std_conf_normalized.groupby(['indiv_name', 'pred_label'])['crop_id'].std().reset_index()
    cm_std_normalized_pivot = std_conf_normalized.pivot(index='indiv_name', columns='pred_label', values='crop_id')
    std_normalized_cm = cm_std_normalized_pivot.reindex(
        index=val_indivs_union, columns=all_classes, fill_value=0
    ).values
    # Recall par individu : diagonale de la matrice normalisée par ligne,
    # moyennée sur les runs où l'individu est en validation.
    rows_sum = np.where(row_total > 0, row_total, 1)
    norm_stacked = stacked / rows_sum[:, :, None]       # (N, n_val, C)
    diag_idx = [all_classes.index(n) for n in val_indivs_union]
    diag = norm_stacked[:, np.arange(n_val), diag_idx]  # (N, n_val)
    mean_per_class = np.array([
        diag[valid[:, i], i].mean() for i in range(n_val)
    ])
    std_per_class = np.array([
        diag[valid[:, i], i].std() for i in range(n_val)
    ])

    # ============================================================
    # TRI PAR CAMÉRA PUIS ORDRE ALPHABÉTIQUE
    # ============================================================
    camera_of = load_camera_of(vp, all_classes)
    camera_names = sorted({cam for cam in camera_of.values() if cam})
    camera_colors = get_camera_colors(camera_names)
    print("\nCameras:", {cam: camera_colors[cam] for cam in camera_names})

    display_order = sorted(all_classes, key=lambda n: (camera_of.get(n, ""), n))
    display_order_val = sorted(val_indivs_union, key=lambda n: (camera_of.get(n, ""), n))
    col_idx = [all_classes.index(n) for n in display_order]
    row_idx_val = [val_indivs_union.index(n) for n in display_order_val]

    mean_cm_int = np.rint(mean_cm[np.ix_(row_idx_val, col_idx)]).astype(int)
    mean_cm_int[mean_cm_int < 0] = 0
    mean_cm_norm = np.zeros_like(mean_cm_int, dtype=float)
    cm_sum = mean_cm_int.sum(axis=1, keepdims=True)
    mask = cm_sum.squeeze(axis=1) != 0
    mean_cm_norm[mask] = mean_cm_int[mask] / cm_sum[mask]

    # Écart-type normalisé par ligne (à partir des matrices normalisées par run,
    # restreintes aux runs où l'individu est en validation)
    std_cm_norm = np.round(std_normalized_cm[np.ix_(row_idx_val, col_idx)],2)

    # ============================================================
    # RECALL MOYEN PAR INDIVIDU
    # ============================================================
    per_class_df = pd.DataFrame({
        "camera": [camera_of.get(n, "") for n in display_order_val],
        "indiv_name": display_order_val,
        "recall_mean": mean_per_class[row_idx_val].round(3),
        "recall_std": std_per_class[row_idx_val].round(3),
    })
    print("\n=================== RECALL MOYEN PAR INDIVIDU ===================")
    print(per_class_df.to_string(index=False))

    # ============================================================
    # ACCURACY PAR (INDIVIDU, ENTRY) — colonne par run
    # ============================================================
    acc = vp.groupby(["run", "indiv_name", "entry"])["correct"].mean() * 100
    pivot = acc.unstack(level=0)
    run_cols = [f"run_{int(r)}" for r in runs]
    pivot.columns = run_cols
    pivot = pivot.reset_index().sort_values(["indiv_name", "entry"])

    # Nombre de crops (uniques) en validation pour (individu, entry)
    nb_crops = (
        vp.groupby(["indiv_name", "entry"])["crop_id"]
        .nunique()
        .rename("nb_crops")
    )
    pivot = pivot.merge(nb_crops, on=["indiv_name", "entry"], how="left")

    # mean_acc et std_acc placés AVANT les colonnes des runs
    pivot["mean_acc"] = pivot[run_cols].mean(axis=1, skipna=True)
    pivot["std_acc"] = pivot[run_cols].std(axis=1, skipna=True, ddof=1)
    pivot = pivot.round(2)
    col_order = ["indiv_name", "entry", "nb_crops", "mean_acc", "std_acc"] + run_cols
    pivot = pivot[col_order]

    iv_path = OUT_DIR / "per_indiv_entry_accuracy_splits.csv"
    pivot.to_csv(iv_path, index=False, na_rep="")
    print(f"\nAccuracy par (individu, entry) sauvegardée dans : {iv_path}")

    # ============================================================
    # SAUVEGARDE DES CSV
    # ============================================================
    runs_df_full.to_csv(OUT_DIR / "accuracies_per_split.csv", index=False)
    pd.DataFrame({
        "metric": ["mean_accuracy", "mean_run_accuracy","std_run_accuracy"],
        "value": [round(mean_acc, 2), round(mean_run_acc,2),round(std_acc, 2)],
    }).to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    per_class_df.to_csv(OUT_DIR / "per_individual_recall.csv", index=False)

    pd.DataFrame(mean_cm_int, index=display_order_val, columns=display_order).to_csv(
        OUT_DIR / "confusion_mean.csv"
    )

    pd.DataFrame(mean_cm_norm, index=display_order_val, columns=display_order).to_csv(
        OUT_DIR / "confusion_mean_normalized.csv",
        float_format="%.3f",
    )
    pd.DataFrame(std_cm_norm, index=display_order_val, columns=display_order).to_csv(
        OUT_DIR / "confusion_std_normalized.csv",
        float_format="%.3f",
    )
    np.savez(
        OUT_DIR / "confusion_matrices.npz",
        mean_cm=mean_cm[np.ix_(row_idx_val, col_idx)],
        mean_cm_int=mean_cm_int,
        mean_cm_norm=mean_cm_norm,
        std_cm_norm=std_cm_norm,
        accuracies=per_run_acc.values,
        display_order=np.array(display_order),
        display_order_val=np.array(display_order_val),
        camera_of=np.array([camera_of[n] for n in display_order]),
        run_names=np.array(runs),
    )

    # ============================================================
    # PLOTS
    # ============================================================
    plot_confusion_matrix(
        mean_cm_int,
        display_order_val,
        display_order,
        f"Mean confusion matrix over {NUM_RUNS} runs (accuracy {mean_acc:.2f} +- {std_acc:.2f}%)",
        OUT_DIR / "confusion_mean.png",
        camera_of=camera_of,
        camera_colors=camera_colors,
        fmt="d",
    )

    plot_confusion_matrix_normalized(
        mean_cm_int,
        display_order_val,
        display_order,
        f"Matrice de confusion normalisée sur {NUM_RUNS} expériences (précision {mean_acc:.2f} +- {std_acc:.2f}%)",
        OUT_DIR / "confusion_mean_normalized.png",
        camera_of=camera_of,
        camera_colors=camera_colors,
    )
    plot_confusion_matrix_normalized(
        std_cm_norm,
        display_order_val,
        display_order,
        f"Normalized confusion matrix standard deviation over {NUM_RUNS} runs",
        OUT_DIR / "confusion_std_normalized.png",
        camera_of=camera_of,
        camera_colors=camera_colors,
        already_normalized=True,
        show_counts=False,
    )

    # Distribution des accuracies par run
    sorted_acc = np.sort(per_run_acc.values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(range(1, NUM_RUNS + 1), sorted_acc, color="steelblue", alpha=0.8)
    ax1.axhline(mean_acc, color="red", linestyle="--", linewidth=1.5,
                label=f"moyenne = {mean_acc:.2f}%")
    ax1.axhline(mean_acc - std_acc, color="orange", linestyle=":", linewidth=1,
                label=f"+- 1 écart-type ({std_acc:.2f}%)")
    ax1.axhline(mean_acc + std_acc, color="orange", linestyle=":", linewidth=1)
    ax1.set_xlabel("Run (trié par accuracy)")
    ax1.set_ylabel("Accuracy de validation (%)")
    ax1.set_title(f"Accuracy par run (min {sorted_acc[0]:.2f}%, max {sorted_acc[-1]:.2f}%)")
    ax1.legend()
    ax2.boxplot(per_run_acc.values, vert=False)
    ax2.set_yticks([])
    ax2.set_xlabel("Accuracy de validation (%)")
    ax2.set_title(f"Moyenne {mean_acc:.2f}% +- {std_acc:.2f}%")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "accuracy_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Recall moyen par individu (barres horizontales, couleurs par caméra)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * n_val)))
    means = mean_per_class[row_idx_val] * 100
    stds = std_per_class[row_idx_val] * 100
    colors = [camera_colors.get(camera_of.get(n, ""), "grey") for n in display_order_val]
    ax.barh(range(n_val), means, xerr=stds, color=colors, alpha=0.85, capsize=3)
    ax.set_yticks(range(n_val))
    ax.set_yticklabels(display_order_val)
    ax.set_xlabel("Recall moyen (%)")
    ax.set_title(f"Recall moyen par individu (validation, {NUM_RUNS} runs)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_individual_recall.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Heatmap accuracy moyenne par (individu, entry)
    hm = pivot.pivot(index="indiv_name", columns="entry", values="mean_acc")
    if len(hm) > 0:
        fig_h = max(8, len(hm) * 0.35)
        fig, ax = plt.subplots(figsize=(18, fig_h))
        sns.heatmap(
            hm,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            ax=ax,
            annot_kws={"size": 7},
            cbar_kws={"label": "Accuracy moyenne (%)"},
        )
        ax.set_xlabel("Entry")
        ax.set_ylabel("Individu")
        ax.set_title(f"Accuracy moyenne par (individu, entry) sur {NUM_RUNS} runs")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "per_indiv_entry_accuracy.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nRésultats et figures sauvegardés dans : {OUT_DIR}")


if __name__ == "__main__":
    main()