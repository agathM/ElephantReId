import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import pickle
import urllib.request
from elephantreid.identification import *
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
import argparse, yaml
import pandas as pd

#PARAMETERS
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers")
FEATURES_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features")
CROP_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers")
MODEL_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/models/identification")
CONFIG_PATH = Path("/pbs/home/a/ameurisse/ElephantReId/config/train_classifiers")
model_name = "ResNet50_v4" #une couche linéar (2048,11)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) #trained on ImageNet
input_transform = transforms.Compose([
            transforms.Resize([224,224]),  #check resize 232 (doc)
            transforms.ToTensor(), # convert numpy.array to tensor
            normalize]) #normalize the tensor


def main()->None:
    parser = argparse.ArgumentParser(description="train classifier for individual identification")
    parser.add_argument("--config", "-c", required=True, help="crop type config file")
    arg = parser.parse_args()
    with open(CONFIG_PATH / arg.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    #where to save precomputed features
    # features_folder = FEATURES_PATH / model_name / cfg['features_folder']
    ## crop_folder = CROP_PATH / "body/entry_selection_17_06"
    selection_type = cfg['selection_type']
    crop_type = cfg['crop_type']
    date = cfg['date']
    split_version = cfg['split_version']
    crop_version = cfg['crop_version']
    mode = cfg['mode']
    version = cfg['version']
    # crop_path = CROP_PATH / crop_type / f"{selection_type}_{date}" / split_version
    crop_path = CROP_PATH / crop_type / crop_version
    results_folder = RESULTS_PATH / version / crop_type
    model_path = MODEL_PATH / version / crop_type
    split_file = CROP_PATH / "split" / cfg["split_file"]
    features_folder = FEATURES_PATH / version / crop_type
    os.makedirs(features_folder,exist_ok=True)
    os.makedirs(model_path,exist_ok=True)
    os.makedirs(results_folder,exist_ok=True)
    model_path = model_path / 'model.pth'
    
    #training hyperparameters
    batch_size = 64
    num_epoch = 15
    lr = 0.001
    
    #CHARGEMENT DU MODELE
    base_model = models.resnet50(weights='DEFAULT')
    #ajouter un block de 3 couches linéaires : tête de classification à entraîner
    classifier_name = "fc"
    dim_in = 2048
    #CHARGEMENT DES DONNEES
    label_map_path = os.path.join(features_folder, "label_map.pt")
    label_map = torch.load(
            label_map_path,
            weights_only=False
        )
    train_data = CustomImageDataset(
            Path(split_file),
            crop_path,
            crop_type,
            transform=input_transform,
            label_map=label_map,
            mode=mode
        )

    val_data = CustomImageDataset(
            Path(file),
            crop_path,
            crop_type,
            split='val',
            transform=input_transform,
            label_map=label_map,
            mode=mode
        )