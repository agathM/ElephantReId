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
    with open(MODEL_CONFIG / arg.config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)  
    

    #where to save precomputed features
    # features_folder = FEATURES_PATH / model_name / model_cfg['features_folder']
    ## crop_folder = CROP_PATH / "body/entry_selection_17_06"
    selection_type = model_cfg['selection_type']
    crop_type = model_cfg['crop_type']
    date = model_cfg['date']
    split_version = model_cfg['split_version']
    crop_version = model_cfg['crop_version']
    mode = model_cfg['mode']
    version = model_cfg['version']
    # crop_path = CROP_PATH / crop_type / f"{selection_type}_{date}" / split_version
    crop_path = CROP_PATH / crop_type / crop_version
    results_folder = RESULTS_PATH / version / crop_type
    model_path = MODEL_PATH / version / crop_type
    split_file = CROP_PATH / model_cfg['split_file']
    features_folder = FEATURES_PATH / version / crop_type
    os.makedirs(features_folder,exist_ok=True)
    os.makedirs(model_path,exist_ok=True)
    os.makedirs(results_folder,exist_ok=True)
    model_path = model_path / 'model.pth'
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
    for epoch in range(num_epoch):
        ### TRAINING
        base_model.eval() # freeze the other parameters
        base_model.fc.train() # train last layer
        error = 0 ; corrects =0
        for feat,label,_ in train_loader:
            feat = feat.to(DEVICE) ; label = label.to(DEVICE)
            output = base_model.fc(feat)
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            error+= loss.item()
            _,predicted = torch.max(output,1)
            corrects += (predicted==label).sum()
        error /= len(train_loader)
        train_error.append(error)
        accuracy = corrects*100/num_train
        train_accuracy.append(accuracy.item())
        if epoch==num_epoch-1:
            print(f"Train accuracy : {accuracy:.1f}")
            print(f"Train error : {error:.2f}")
        ### VALIDATION 
        base_model.eval()
        error = 0 ; corrects =0
        with torch.no_grad():
            for feat,label,idx in val_loader:
                feat = feat.to(DEVICE) ; label = label.to(DEVICE)
                output = base_model.fc(feat)
                error+= criterion(output,label).item()
                _,predicted = torch.max(output,1)
                corrects += (predicted==label).sum().item()
                if epoch==num_epoch-1:
                    all_predicted.append(predicted.cpu());all_labels.append(label.cpu());all_idx.append(idx.cpu())
                    all_outputs.append(output.cpu())
                    
        error /= len(val_loader)
        val_error.append(error)
        val_accuracy.append(corrects*100/num_val)
        if epoch==num_epoch-1:
            print(f"Val accuracy : {corrects*100/num_val:.1f}")
            print(f"Val error : {error:.2f}")
        #scheduler for lr
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(error)
        new_lr = optimizer.param_groups[0]["lr"]
    
        # if new_lr != old_lr:
        #     print(f"LR reduced from {old_lr:.2e} to {new_lr:.2e}")
    
    
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
    plt.savefig(results_folder/'train_curv.png', dpi=300, bbox_inches="tight")
    plt.close()
    
    # SAVE DATA
    torch.save({
        "outputs": torch.cat(all_outputs),
        "labels": torch.cat(all_labels),
        "idx": torch.cat(all_idx),
        "class_names": class_names,
        "val_error": val_error,
    }, results_folder / "val_predictions.pt")

    # SAVE PREDICTIONS

    all_outputs = torch.cat(all_outputs)
    
    # Convert logits to probabilities
    all_probs = torch.softmax(all_outputs, dim=1)
    
    out_df = pd.read_excel(
        split_file,
        sheet_name="samples",
        engine="odf",
        header=0
    )
    out_df = out_df[out_df['mode']==mode]
    out_df = out_df[out_df['split']=="val"].reset_index(drop=True)
    # Save logits
    out_df['logits'] = [
        [round(v, 2) for v in x.tolist()]
        for x in all_outputs.cpu().numpy()
    ]
    
    # Save probabilities
    out_df['pred'] = [
        [round(v, 2) for v in x.tolist()]
        for x in all_probs.cpu().numpy()
    ]
    
    # Softmax confidence
    out_df['conf_pred_label'] = (
        all_probs.max(dim=1).values
        .cpu()
        .numpy()
        .round(2)
    )
    
    # Margin
    top2_probs = torch.topk(all_probs, k=2, dim=1).values
    out_df['margin'] = (
        (top2_probs[:,0] - top2_probs[:,1])
        .cpu()
        .numpy()
        .round(2)
    )
    
    # Entropy
    entropy = -(all_probs * torch.log(all_probs + 1e-8)).sum(dim=1)
    out_df['entropy'] = (
        entropy
        .cpu()
        .numpy()
        .round(2)
    )
    
    
    # Predicted class
    out_df['pred_label'] = all_probs.argmax(dim=1).cpu().numpy()
    out_df['pred_label'] = out_df['pred_label'].apply(
        lambda x: class_names[x]
    )
    # float_cols = out_df.select_dtypes(include=["float"]).columns
    # out_df[float_cols] = out_df[float_cols].round(2)
    out_df.to_csv(
        results_folder / "val_pred_metadata.csv",
        index=False,
        float_format="%.2f"
    )

    #SAVE MODEL
    torch.save(base_model.state_dict(),model_path)
    
if __name__=="__main__":
    main()

