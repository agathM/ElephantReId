""" Videos selections for body crops with (MegaDetector -> for manual crops can keep some videos with several indivs):
Only individuals appearing in at least 2 entries (1 entry = successive videos overlaps less than 2 hours)
Check that at least two individuals appear in each mode (night/day) and camera (beach/cave) pair
"""

import argparse
import os
import pandas as pd

SOURCE_PATH = "/sps/mnhn/ameurisse/ElephantReId/data/raw/"
VIDEOS_METADATA_FILE = "/sps/mnhn/ameurisse/ElephantReId/data/raw/summary_videos.csv"

def main() -> None:
    # parser = argparse.ArgumentParser(description="Dataset selection to avoid biases")
    # parser.add_argument("--path", "-p", help=".csv containing clips information", required=True)
    # arg = parser.parse_args()
    
    database = pd.read_csv(VIDEOS_METADATA_FILE,header=0)
    database['several_indivs'] = database['several_indivs'].astype(int)
    database = database[database['several_indivs']==0]
    database.drop(columns=['several_indivs'],inplace=True)
    database.sort_values(['indiv_name','video'],inplace=True)

    #indivs appearing in at least 2 videos
    several_videos_indivs = (database.groupby("indiv_name")["video"].nunique().loc[lambda s: s > 1].index)
    database = database[database['indiv_name'].isin(several_videos_indivs)].copy()

    #keep only camera, mode pair with several indivs
    nb_indivs_by_camera_mode = (database.groupby(['camera','mode'])['indiv_name'].nunique().reset_index(name='n_indivs'))
    single_indiv_camera_modes = nb_indivs_by_camera_mode[nb_indivs_by_camera_mode["n_indivs"] <= 1][["camera", "mode"]]
    indivs_to_remove = (database.merge(single_indiv_camera_modes,on=["camera", "mode"],how="inner")["indiv_name"].unique())
    print(indivs_to_remove)
    filtered_database = database[~database["indiv_name"].isin(indivs_to_remove)].copy()
    output_path = os.path.join(SOURCE_PATH, "video_selection2.ods")

    with pd.ExcelWriter(output_path, engine="odf") as writer:
        filtered_database.to_excel(
            writer,
            sheet_name="filtered_database",
            index=False
        )
    
        # ------------------------------------------------------------------
        # nb vidéos / individu
        # ------------------------------------------------------------------
    
        videos_per_indiv = (
            filtered_database
            .groupby(["indiv_name", "camera", "mode"])["video"]
            .nunique()
            .reset_index(name="n_videos")
            .sort_values("n_videos", ascending=False)
        )
    

        videos_per_indiv = videos_per_indiv.pivot(
            index="indiv_name",
            columns=["camera", "mode"],
            values="n_videos"
        ).fillna(0).astype(int)

        videos_per_indiv.columns = [
            f"{camera}_{mode}"
            for camera, mode in videos_per_indiv.columns
        ]

        total = (
            filtered_database
            .groupby("indiv_name")["video"]
            .nunique()
        )

        videos_per_indiv["Total"] = total

        videos_per_indiv = (
            videos_per_indiv
            .sort_values("Total", ascending=False)
            .reset_index()
        )
    
        # ------------------------------------------------------------------
        # distribution du nombre de vidéos
        # ------------------------------------------------------------------
    
        distribution = []
    
        # par camera/mode
        for (camera, mode), df in filtered_database.groupby(["camera", "mode"]):
    
            tmp = (
                df.groupby("indiv_name")["video"]
                .nunique()
                .value_counts()
                .sort_index()
                .rename_axis("n_videos")
                .reset_index(name="n_individuals")
            )
    
            tmp["camera"] = camera
            tmp["mode"] = mode
    
            distribution.append(tmp)
    
        distribution = pd.concat(distribution, ignore_index=True)
    
        distribution.to_excel(
            writer,
            sheet_name="distribution",
            index=False
        )
        return None

if __name__=="__main__":
    main()
    
        
    

