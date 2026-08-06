import pandas as pd
import os,shutil,sys,argparse

def main() -> None:
    """ before folder > Indiv_name > Mode(Night&Day) 
        after  folder > Indiv_name > Mode(Night&Day) > clips/several_indivs """
    parser = argparse.ArgumentParser(description="Filter clip with only one individual")
    parser.add_argument("--folder", "-f",required=True,help="chemin vers le folder à traiter")
    arg = parser.parse_args()

    if not os.path.exists(arg.folder):
        print("Unfound path")
        sys.exit(1)
    data_info_path = os.path.join(os.path.dirname(arg.folder),"DB_SEB_VD.ods")
    database = pd.read_excel(data_info_path,sheet_name="DB_SEB_VD",header=0,engine='odf')
    database = database.filter(['clip','indiv_number','timestamp'])
    database = database[database['indiv_number'].astype(str).str.fullmatch(r'\d+', na=False)]
    database['indiv_number'] = database['indiv_number'].astype(int)
    
    database = database.filter(['clip','indiv_number','time'])
    clips_1_indiv = database[database['indiv_number']==1]['clip'].to_list()
    clips_1_indiv = [c.split(".")[0] for c in clips_1_indiv]
    print(clips_1_indiv[:10])
    
    for indiv in os.listdir(arg.folder):
        indiv_path = os.path.join(arg.folder,indiv)
        for mode in os.listdir(indiv_path):
            if mode in ['Night','Day']:
                mode_path = os.path.join(indiv_path,mode)
                for clip in os.listdir(mode_path):
                    several_indivs = os.path.join(mode_path,"several_indivs")
                    if clip not in clips_1_indiv:
                        src = os.path.join(mode_path,clip)
                        os.makedirs(several_indivs,exist_ok=True)
                        shutil.move(src=src,dst=os.path.join(several_indivs,clip))


if __name__ == "__main__":
    main()