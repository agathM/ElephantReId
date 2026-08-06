"""À partir du fichier des vidéos, crée un split train/val par vidéo
(pour chaque individu et mode), copie les crops et génère train_data.csv
et val_data.csv.
"""

import shutil, os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


crops_type = "body"
crop_selection = "entry_07_15"

CROPS_SOURCE_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/") / crops_type / "all"

CROPS_SPLIT_PATH = (
        Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/")
    / crops_type
    / crop_selection
)

CROPS_METADATA = (
    Path("~/Documents/MNHN/database_info/processed/crops").expanduser()
    / crops_type
    / "crops_annotations.csv"
)

CROPS_SPLIT_METADATA = (
    Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/") / crops_type
    / crops_type
    / "crops_split.ods"
)

ENTRIES_FILE = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw") / file_selection


def copy_video(src: Path, dst: Path):
    """Copie un dossier vidéo s'il n'existe pas déjà."""
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def build_summary(df, value_col):

    tmp = (
        df.groupby(
            ["indiv_name", "camera", "mode", "split"],
            observed=True
        )[value_col]
        .nunique()
        .rename("count")
        .reset_index()
    )

    pivot = tmp.pivot_table(
        index="indiv_name",
        columns=["camera", "mode", "split"],
        values="count",
        fill_value=0,
    )

    pivot.columns = [
        f"{camera}_{mode}_{split}"
        for camera, mode, split in pivot.columns
    ]

    return pivot.reset_index()


def main():

    df = pd.read_excel(ENTRIES_FILE, sheet_name="clips",engine="odf")
    df = df.sort_values(["indiv_name", "mode", "entry", "video"])

    train_folder = CROPS_SPLIT_PATH / "train"
    val_folder = CROPS_SPLIT_PATH / "val"

    os.makedirs(train_folder,exist_ok=True)
    os.makedirs(val_folder,exist_ok=True)

    train_samples = []
    val_samples = []

    for (indiv, mode), group in df.groupby(["indiv_name", "mode"]):

        entries = sorted(group["entry"].unique())
        if len(entries) < 2:
            train_entries = entries
            val_entries = []
        else:
            train_entries, val_entries = train_test_split(
                entries,
                train_size = 0.75,
                random_state=42,
            )
        # ---------------- TRAIN ---------------- #

        train_group = group[group["entry"].isin(train_entries)]

        for _, row in train_group.iterrows():

            video = row["video"]

            src = CROPS_SOURCE_PATH / video
            dst = train_folder / indiv / mode / video

            copy_video(src, dst)
            if 

            for img_path in src.iterdir():
                if img_path.is_file():
                    train_samples.append(
                        {
                            "crop": img_path.name,
                            **row.to_dict(),
                        }
                    )

        # ---------------- VAL ---------------- #

        val_group = group[group["entry"].isin(val_entries)]

        for _, row in val_group.iterrows():

            video = row["video"]

            src = CROPS_SOURCE_PATH / video
            dst = val_folder / indiv / mode / video

            copy_video(src, dst)

            for img_path in src.iterdir():
                if img_path.is_file():
                    val_samples.append(
                        {
                            "crop": img_path.name,
                            **row.to_dict(),
                        }
                    )
    val_samples = pd.DataFrame(val_samples)
    val_samples["split"] = "val"
    train_samples = pd.DataFrame(train_samples)
    train_samples["split"] = "train"
    samples = pd.concat([train_samples, val_samples])
    samples['crop_id'] = samples['video'] + '/' + samples['crop']
    entry_summary = build_summary(samples, "entry")
    video_summary = build_summary(samples, "video")
    crop_summary = build_summary(samples, "crop_id")
    with pd.ExcelWriter(CROPS_SPLIT_METADATA,engine='odf') as writer:
        samples.to_excel(writer, sheet_name="samples", index=False)
        entry_summary.to_excel(writer, sheet_name="nb_entry", index=False)
        video_summary.to_excel(writer, sheet_name="nb_video", index=False)
        crop_summary.to_excel(writer, sheet_name="nb_crop", index=False)

if __name__ == "__main__":
    main()