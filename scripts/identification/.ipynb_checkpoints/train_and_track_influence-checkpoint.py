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
from torch.func import functional_call, grad, vmap

#PARAMETERS
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers")
FEATURES_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features")
CROP_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers")
MODEL_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/models/identification")
MODEL_CONFIG = Path("/pbs/home/a/ameurisse/ElephantReId/config/train_classifiers")
TRACK_CONFIG = Path("/pbs/home/a/ameurisse/ElephantReId/config/TrackIn")
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

    with open(TRACK_CONFIG / args.config, "r") as f:
        track_cfg = yaml.safe_load(f)
    CHECKPOINTS = track_cfg["checkpoints"]
    
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
    track_file = results_folder / "val_indices_for_tracking.pt"
    selected_val = torch.load(selected_val)
    print(crop_type, mode)
    #training hyperparameters
    batch_size = 64
    num_epoch = 30
    lr = 0.001
    
    #CHARGEMENT DU MODELE
    base_model = models.resnet50(weights='DEFAULT')
    #ajouter un block de 3 couches linéaires : tête de classification à entraîner
    classifier_name = "fc"
    dim_in = 2048
    #CHARGEMENT DES DONNEES
    train_features_dataset, val_features_dataset, train_data, val_data,_ = load_data(base_model,input_transform,crop_path, split_file ,crop_type,DEVICE,features_folder,mode)
    class_names = train_data.classes
    num_classes = train_data.classes_nb
    setattr(base_model,classifier_name,ClassificationHead(dim_out=num_classes,dim_in=dim_in))
    assert(getattr(base_model,classifier_name).out_features==num_classes)
    base_model.to(DEVICE)

    #ENTRAINEMENT
    g = torch.Generator().manual_seed(SEED)
    num_train = len(train_features_dataset)
    num_val = len(val_features_dataset)
    criterion = nn.CrossEntropyLoss()
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.fc.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(base_model.fc.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )
    train_error = [] ; val_error =[]
    train_accuracy = [] ; val_accuracy = []
    train_loader = DataLoader(train_features_dataset,shuffle=True,generator=g,batch_size=batch_size)
    val_loader = DataLoader(val_features_dataset,batch_size=batch_size,shuffle=False)
    all_predicted, all_labels, all_idx, all_outputs = [],[],[],[]

    # ============================================================
    # Functions for per-sample gradients
    # ============================================================

    def loss_per_sample(params, feature, target):
    
        logits = functional_call(
            base_model.fc,
            params,
            (feature.unsqueeze(0),)
        )
    
        return (
            F.cross_entropy(
                logits,
                target.unsqueeze(0)
            ),
            logits
        )
    
    
    per_sample_grad_fn = vmap(
        grad(
            loss_per_sample,
            has_aux=True
        ),
        in_dims=(None, 0, 0)
    )
    
    
    # ============================================================
    # Training
    # ============================================================
    
    training_history = []
    
    validation_history = {}
    
    
    for epoch in range(num_epoch):
    
    
        ###########################################################
        # TRAIN
        ###########################################################
    
        base_model.eval()
        base_model.fc.train()
    
        error = 0
        corrects = 0
    
    
        for batch_idx, (feat, label, idx) in enumerate(train_loader):
    
            feat = feat.to(DEVICE)
            label = label.to(DEVICE)
    
    
            params = dict(
                base_model.fc.named_parameters()
            )
    
    
            #######################################################
            # Per-sample gradients BEFORE optimizer.step()
            #######################################################
    
            sample_grads, output = per_sample_grad_fn(
                params,
                feat,
                label
            )
    
    
            flat_grads = []
    
            for i in range(feat.size(0)):
    
                g = []
    
                for name in params.keys():
    
                    g.append(
                        sample_grads[name][i].reshape(-1)
                    )
    
                flat_grads.append(
                    torch.cat(g).cpu()
                )
    
    
            #######################################################
            # Optimizer update using already computed gradients
            #######################################################
    
            optimizer.zero_grad()
    
    
            for name, param in base_model.fc.named_parameters():
    
                param.grad = sample_grads[name].mean(dim=0)
    
    
            optimizer.step()
    
    
            #######################################################
            # Save train TracIn information
            #######################################################
    
            training_history.append(
                {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "lr": optimizer.param_groups[0]["lr"],
    
                    # indices in original train dataset
                    "indices": idx.cpu(),
    
                    # gradients associated with these indices
                    "gradients": torch.stack(flat_grads)
                }
            )
    
    
            #######################################################
            # Metrics
            #######################################################
    
            loss = criterion(output, label)
    
            error += loss.item()
    
    
            _, predicted = torch.max(
                output,
                1
            )
    
            corrects += (
                predicted == label
            ).sum().item()
    
    
    
        error /= len(train_loader)
    
        train_error.append(error)
    
        accuracy = corrects * 100 / num_train
    
        train_accuracy.append(accuracy)
    
    
    
        if epoch == num_epoch - 1:
    
            print(f"Train accuracy : {accuracy:.1f}")
            print(f"Train error : {error:.2f}")
    
    
    
        ###########################################################
        # Validation
        ###########################################################
        
        base_model.eval()
        
        error = 0
        corrects = 0
        
        
        # store one gradient per validation point
        val_gradients_epoch = []
        val_indices_epoch = []
        
        
        for feat, label, idx in val_loader:
        
            feat = feat.to(DEVICE)
            label = label.to(DEVICE)
        
        
            params = dict(
                base_model.fc.named_parameters()
            )
        
        
            #######################################################
            # Compute per-sample gradients only for selected points
            #######################################################
        
            if epoch in checkpoint_epochs:
        
        
                sample_grads, output = per_sample_grad_fn(
                    params,
                    feat,
                    label
                )
        
        
                for i in range(feat.size(0)):
        
                    current_idx = idx[i].item()
        
        
                    # keep only selected validation samples
                    if current_idx not in selected_val:
                        continue
        
        
                    g = []
        
        
                    for name in params.keys():
        
                        g.append(
                            sample_grads[name][i].reshape(-1)
                        )
        
        
                    # gradient of ONE validation point
                    val_gradients_epoch.append(
                        torch.cat(g).cpu()
                    )
        
        
                    # original validation dataset index
                    val_indices_epoch.append(
                        current_idx
                    )
        
        
            else:
        
                with torch.no_grad():
        
                    output = base_model.fc(feat)
        
        
        
            #######################################################
            # Metrics
            #######################################################
        
            loss = criterion(
                output,
                label
            )
        
            error += loss.item()
        
        
            _, predicted = torch.max(
                output,
                1
            )
        
        
            corrects += (
                predicted == label
            ).sum().item()
    
    
            if epoch == num_epoch - 1:
    
                all_predicted.append(
                    predicted.cpu()
                )
    
                all_labels.append(
                    label.cpu()
                )
    
                all_idx.append(
                    idx.cpu()
                )
    
                all_outputs.append(
                    output.cpu()
                )
    
    
    
        ###########################################################
        # Save validation gradients
        ###########################################################
        
        if epoch in checkpoint_epochs:
        
            validation_history[epoch] = {
        
                "indices":
                    torch.tensor(
                        val_indices_epoch
                    ),
        
                "gradients":
                    torch.stack(
                        val_gradients_epoch
                    )
        
            }
    
    
    
        val_error_epoch /= len(val_loader)
    
        val_error.append(
            val_error_epoch
        )
    
    
        val_accuracy.append(
            corrects * 100 / num_val
        )
    
    
        if epoch == num_epoch - 1:
    
            print(
                f"Val accuracy : {corrects*100/num_val:.1f}"
            )
    
            print(
                f"Val error : {val_error_epoch:.2f}"
            )
    
    
        ###########################################################
        # Scheduler
        ###########################################################
    
        scheduler.step(
            val_error_epoch
        )
    
    
    #SAVE TRAINING
    plt.figure(figsize=(4,6))
    plt.subplot(2,1,1)
    plt.plot(range(1,num_epoch+1),train_error,color="blue",label="train")
    plt.plot(range(1,num_epoch+1),val_error,color="orange",label="validation")
    plt.title("Loss")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(range(1,num_epoch+1),train_accuracy,color="blue",label="train")
    plt.plot(range(1,num_epoch+1),val_accuracy,color="orange",label="validation")
    plt.plot(range(1,num_epoch+1),[1/num_classes*100 for i in range(1,num_epoch+1)],color="red",label="random")
    plt.title("Accuracy")
    plt.savefig(results_folder/'track_train_curv.png', dpi=300, bbox_inches="tight")
    plt.close()
    # ============================================================
    # Save TracIn information
    # ============================================================
    
    
    torch.save(
        training_history,
        results_folder / "training_history.pt"
    )
    
    
    torch.save(
        validation_history,
        results_folder / "validation_history.pt"
    )
