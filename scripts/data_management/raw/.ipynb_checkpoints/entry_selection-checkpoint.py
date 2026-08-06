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
    database.sort_values(['indiv_name','entry'],inplace=True)

    #indivs appearing in at least 2 entries
    several_entries_indivs = (database.loc[database['entry']>0,'indiv_name'].drop_duplicates().tolist())
    database = database[database['indiv_name'].isin(several_entries_indivs)].copy()

    #camera, mode pair with several indivs
    nb_indivs_by_camera_mode = (database.groupby(['camera','mode'])['indiv_name'].nunique().reset_index(name='n_indivs'))
    single_indiv_camera_modes = nb_indivs_by_camera_mode[nb_indivs_by_camera_mode["n_indivs"] <= 1][["camera", "mode"]]
    indivs_to_remove = (database.merge(single_indiv_camera_modes,on=["camera", "mode"],how="inner")["indiv_name"].unique())
    filtered_database = database[~database["indiv_name"].isin(indivs_to_remove)].copy()
    output_path = os.path.join(SOURCE_PATH,"entry_selection.csv")
    filtered_database.to_csv(output_path, index=False)
    return None

if __name__=="__main__":
    main()
    
        
    

