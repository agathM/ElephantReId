import argparse
import os
import pandas as pd


def main()->None:
    parser = argparse.ArgumentParser(description="train and evaluate classifier")
    parser.add_argument("--config", "-c", required=True, help="pipeline file")
    arg = parser.parse_args()

    database_file = arg.database_file
    dataset_date = arg.dataset_date
    split_method = arg.split_method.replace("/","_")
    split_file = f"{dataset_date}_{model_name}_{split_method}.csv"
    model_name = arg.detection_model
    detection_part, detection_type, date = model_name.split('_')
    HOME_PATH = arg.home_path
    CROP_PATH = HOME_PATH / "data/processed/crops" / model_name
    os.makedirs(CROP_PATH, exist_ok=True)
    CROP_FILE_PATH = HOME_PATH / "data/processed/crops/crops_files" / f"{model_name}.csv" #un identifiant unique par détection : video, frame, coords boîte ou indiv_name?
    os.makedirs(CROP_FILE_PATH.parent, exist_ok=True)
    SPLIT_PATH = HOME_PATH / "data/processed/crops/split_files" / split_file
    os.makedirs(SPLIT_PATH.parent, exist_ok=True)
    MODEL_PATH = HOME_PATH / model_name
    RESULTS_PATH = HOME_PATH / "data/results"
    

    if not os.path.exists(CROP_PATH):
        import scripts.1_detect as detect
        detect.main() -model MODEL_PATH -database_file
    
    if not os.path.exists(SPLIT_PATH):
        import scripts.2_create_split as create_split
        create_split.main() -m {split_method} -train_ratio {train_ratio} -crop_file CROP_FILE_PATH -dataset_date dataset_date

    if not 
    df = pd.read_csv(SPLIT_PATH)
    images = df.image.unique()

    

    