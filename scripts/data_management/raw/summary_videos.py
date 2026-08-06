import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

VIDEOS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras")
SOURCE_METADATA = "/sps/mnhn/ameurisse/ElephantReId/data/raw/DB_SEB_VD.ods"
SOURCE_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw/")
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}

def main()->None:
    """From a raw data directory of all individual videos sorted this way :
    camera > individual > mode > videos + excluded several indivs folder
    Summarize the dataset in a csv file with columns : indiv_name, camera, mode, time, entry, clip_path"""
    #add CAMERAS SELECTION ?
    # parser = argparse.ArgumentParser(description="Dataset Review")
    # parser.add_argument('--path','-p',required=True,help="Folder to treat")
    # arg = parser.parse_args()
    
    database = pd.read_excel(SOURCE_METADATA,sheet_name="DB_SEB_VD",header=0,engine='odf')
    database = database.filter(['clip','timestamp','mode'])
    database.dropna(subset=["clip"],inplace=True)
    
    database["time"] = database["timestamp"].apply(lambda x: pd.to_datetime(x,format='mixed',dayfirst=True))
    database = database.filter(['clip','time','mode'])
    database['clip'] = database['clip'].apply(lambda x: x.split('.')[0])
    
    rows = []
    for camera in VIDEOS_PATH.iterdir():
        if camera.is_dir():
            for indiv in camera.iterdir():
                #Add date to dataframe
                if indiv.is_dir():
                    for mode in indiv.iterdir():
                        if mode.name in ['Night','Day']:
                            videos_1_indiv,videos_several_indivs = [], []
                            for f in mode.iterdir():
                                if (not f.is_dir()) and (f.name.split('.')[1].lower() in VIDEO_EXTENSIONS):
                                    videos_1_indiv.append(f.name.split('.')[0])
                                elif f.is_dir() and f.name=="several_indivs":
                                    videos_several_indivs = [f2.name.split('.')[0] for f2 in f.iterdir()]
                            try:
                                rows.extend([(indiv.name,camera.name,mode.name,database[database['clip'].isin([v])]['time'].iloc[0],v,0) for v in videos_1_indiv])
                                rows.extend([(indiv.name,camera.name,mode.name,database[database['clip'].isin([v])]['time'].iloc[0],v,1) for v in videos_several_indivs])
                            except:
                                for v in videos_1_indiv:
                                    print(indiv.name,camera.name,mode.name,v)
                                    break
                                for v in videos_several_indivs:
                                    print(indiv.name,camera.name,mode.name,v)
                                    break
                                break
    clips_dataframe = pd.DataFrame(rows,columns=['indiv_name','camera','mode','time','video','several_indivs'])

    #ADD ENTRIES 
    clips_dataframe= clips_dataframe.sort_values(["indiv_name","time"]).reset_index(drop=True)
    times = clips_dataframe["time"].to_numpy()
    names = clips_dataframe["indiv_name"]
    entry = np.zeros(len(times), dtype=np.int32)
    group = 0
    start = times[0]
    for i in range(1, len(times)):
        if names[i-1]!=names[i]:
            group=0 ; start = times[i]
        elif times[i] - start >= np.timedelta64(2, 'h'):
            group += 1
            start = times[i]
        entry[i] = group
    clips_dataframe["entry"] = entry
    clips_dataframe.to_csv(SOURCE_PATH / 'summary_videos.csv',index=False)

if __name__=="__main__":
    main()