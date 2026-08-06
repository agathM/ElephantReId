import os
import pandas as pd
import numpy as np
import argparse


def main()->None:
    """For Extraction with yolo
    Read csv adding time from source database DB_SEB_VD
    Add entry to csv"""
    parser = argparse.ArgumentParser(description="Dataset Review")
    parser.add_argument('--path','-p',required=True,help="Folder to treat")
    arg = parser.parse_args()
    
    database = pd.read_excel("data/DB_SEB_VD.ods",sheet_name="DB_SEB_VD",header=0,engine='odf')
    database = database.filter(['clip','timestamp','mode'])
    database.dropna(subset=["clip"],inplace=True)
    
    database["time"] = database["timestamp"].apply(lambda x: pd.to_datetime(x,format='mixed',dayfirst=True))
    database = database.filter(['clip','time','mode'])
    
    
    data_path = arg.path
    l = []
    for indiv in os.listdir(data_path):
        try:
            #Add date to dataframe
            indiv_path = os.path.join(data_path,indiv,'metadata.csv')
            indiv_dataframe = pd.read_csv(indiv_path)
            indiv_dataframe['name'] = indiv
            indiv_dataframe = indiv_dataframe.filter(['name','video','crop_path'])
            indiv_dataframe = indiv_dataframe.set_index('video').join(database.set_index('clip'),how='left')
            #Add entry to dataframe (one entry = all videos taken at the same spot within a 2 hours range)
            # A revoir pour la nuit?
            indiv_dataframe['clip'] = indiv_dataframe.index
            indiv_dataframe= indiv_dataframe.sort_values("time").reset_index(drop=True)
            times = indiv_dataframe["time"].to_numpy()
            entry = np.zeros(len(times), dtype=np.int32)
            group = 0
            start = times[0]
            for i in range(1, len(times)):
                if times[i] - start >= np.timedelta64(2, 'h'):
                    group += 1
                    start = times[i]
                entry[i] = group
            indiv_dataframe["entry"] = entry
            l.append(indiv_dataframe)
            indiv_dataframe = indiv_dataframe.filter(['clip','entry']).drop_duplicates()
            metadata = pd.read_csv(indiv_path).set_index('video')
            metadata = metadata.join(indiv_dataframe.set_index('clip'),how='left')
            metadata['video'] = metadata.index
            metadata.to_csv(indiv_path,index=False)
        except:
            continue
    data = pd.concat(l)
    data.to_csv(os.path.join(data_path,'data_summary.csv'),index=False)
    
    data = data.filter(['name','entry','clip','crop_path','mode']).drop_duplicates()
    crop_per_entry =  data.groupby(['name','entry','mode','clip'])['crop_path'].count()
    crop_per_entry.to_csv(os.path.join(data_path,'crop_per_entry.csv'),index=True)
    
    data = data.filter(['name','entry','mode','clip']).drop_duplicates()
    clip_per_entry =  data.groupby(['name','entry','mode'])['clip'].count()
    clip_per_entry.to_csv(os.path.join(data_path,'clip_per_entry.csv'),index=True)

    data = data.filter(['name','mode','clip']).drop_duplicates()
    clip_per_mode =  data.groupby(['name','mode'])['clip'].count()
    clip_per_mode.to_csv(os.path.join(data_path,'clip_per_mode.csv'),index=True)

    
    data = data.filter(['name','entry']).drop_duplicates()
    data = data.groupby('name').count()
    data.rename(columns={'entry':'entries'})
    data.to_csv(os.path.join(data_path,'data_summary_entries.csv'),index=True) #garder index quand groupby

if __name__=="__main__":
    main()