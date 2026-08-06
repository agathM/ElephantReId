#python ElephantReId/scripts/data_management/processed/summary_megadetector_crops.py -cf "entry_selection_17_06" -if "entry_selection.csv"

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

VIDEO_METADATA_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw")
MEGADETECTOR_CROP_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/body") #stock tous les fichiers informatifs dans le répertoire principal

def main()->None:
    """For extraction with Megadetector
    Read csv created with video selection containing metadata
    Create csv adding the crops to each video"""
    parser = argparse.ArgumentParser(description="Dataset Review")
    parser.add_argument('--crop_folder','-cf',required=True,help="folder containing crops")
    parser.add_argument('--input_file','-if',required=True,help="Metadata file to read")
    arg = parser.parse_args()
    crop_path = MEGADETECTOR_CROP_PATH / arg.crop_folder
    video_metadata_file = VIDEO_METADATA_PATH / arg.input_file
    database = pd.read_csv(video_metadata_file,header=0)
    database.dropna(subset=["video"],inplace=True)

    l = []
    for indiv in crop_path.iterdir():
        for mode in indiv.iterdir():
            for video in mode.iterdir():
                row = database.loc[database["video"]==video.name.split('.')[0]].iloc[0]
                row = [row['indiv_name'], row['camera'],row['mode'],row['entry']]
                l.extend([row + [crop_path] for crop_path in video.iterdir()])
                
    crop_metadata = pd.DataFrame(l,columns=['indiv_name','camera','mode','entry',"crop_path"])
    crop_metadata.to_csv(MEGADETECTOR_CROP_PATH/ f'crop_summary_{MEGADETECTOR_CROP_PATH.name}_{arg.crop_folder}.csv',index=False) #type de crop + version

if __name__=="__main__":
    main()