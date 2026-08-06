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
import os

def save_confusion_matrix(cm, class_names,out_dir):
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'confusion_matrix'), dpi=300, bbox_inches="tight")

#PARAMETERS
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "ResNet50_v1" #une couche linéar (2048,11)
RESULTS_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/results/classifiers")
FEATURES_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features")
CROP_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops")
MODEL_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/models/identification")

#where to save precomputed features
features_folder = FEATURES_PATH / "entry_selection_17_06/body"
features_folder = FEATURES_PATH / "entry_selection_17_06/head_remove_bad_crops"
features_folder = FEATURES_PATH / "entry_selection_17_06/right_ear_remove_bad_crops" 
os.makedirs(features_folder,exist_ok=True)
## crop_folder = CROP_PATH / "body/entry_selection_17_06"
crop_file = CROP_PATH / "body/crop_summary_body_entry_selection_17_06.csv"
crop_file = CROP_PATH / "crop_summary_Head_beach_17_06_remove_bad_crops.csv"
crop_file = CROP_PATH / "crop_summary_Right_Ear_beach_17_06_remove_bad_crops.csv" #containing path to crops
classif_name = "body_entry_selection"
classif_name = "head_entry_selection_17_06"
classif_name = "right_ear_entry_selection_17_06"
print(classif_name)
results_folder = RESULTS_PATH / model_name / classif_name
os.makedirs(results_folder,exist_ok=True)

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
train_features_dataset, val_features_dataset, train_data, val_data = load_data(base_model,crop_file,results_folder,device,features_folder)
print(train_data.label_map)
print(val_data.label_map)
num_classes = len(torch.unique(train_features_dataset.tensors[1]))
setattr(base_model,classifier_name,ClassificationHead(dim_out=num_classes,dim_in=dim_in))
assert(getattr(base_model,classifier_name).out_features==num_classes)
base_model.to(device)


#ENTRAINEMENT
if not os.path.exists(MODEL_PATH / f'{model_name}_{classif_name}.pth'):
    g = torch.Generator().manual_seed(seed)
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
    val_loader = DataLoader(val_features_dataset,batch_size=batch_size)
    all_predicted, all_labels, all_idx = [],[],[]
    for epoch in range(num_epoch):
        ### TRAINING
        base_model.eval() # freeze the other parameters
        base_model.fc.train() # train last layer
        error = 0 ; corrects =0
        for feat,label,_ in train_loader:
            feat = feat.to(device) ; label = label.to(device)
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
        ### VALIDATION 
        base_model.eval()
        error = 0 ; corrects =0
        with torch.no_grad():
            for feat,label,idx in val_loader:
                feat = feat.to(device) ; label = label.to(device)
                output = base_model.fc(feat)
                error+= criterion(output,label).item()
                _,predicted = torch.max(output,1)
                corrects += (predicted==label).sum().item()
                if epoch==num_epoch-1:
                    all_predicted.append(predicted.cpu());all_labels.append(label.cpu());all_idx.append(idx.cpu())
        error /= len(val_loader)
        val_error.append(error)
        val_accuracy.append(corrects*100/num_val)
        #scheduler for lr
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(error)
        new_lr = optimizer.param_groups[0]["lr"]
    
        if new_lr != old_lr:
            print(f"LR reduced from {old_lr:.2e} to {new_lr:.2e}")

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
    plt.title("Accuracy")
    plt.savefig(results_folder/'train_curv.png', dpi=300, bbox_inches="tight")

    #SAVE CONFUSION MATRIX
    all_labels = torch.cat(all_labels) ; all_predicted = torch.cat(all_predicted)
    cm = confusion_matrix(all_labels,all_predicted)
    class_names = train_data.classes
    save_confusion_matrix(cm, class_names,results_folder)

    #SAVE MODEL
    torch.save(base_model.state_dict(),MODEL_PATH / f'{model_name}_{classif_name}.pth')

else:
    state_dict = torch.load(MODEL_PATH / f"{model_name}_{classif_name}.pth", map_location=device)
    base_model.load_state_dict(state_dict)
    base_model.to(device)
    base_model.eval()

#GRAD_CAM
classes = dict()
for key,value in train_data.label_map.items():
    classes[value] = key
    
#choix : un gradcam par individu : meilleur guess et moins bon guess
import torch.nn.functional as F
from collections import defaultdict

model = base_model.eval()

#indiv > correct, wrong pred
per_indiv_stats = defaultdict(lambda: {
    "correct": [],
    "wrong": []
})

with torch.no_grad():
    for feat, label, idx in val_loader:
        feat = feat.to(device) ; label = label.to(device)
        output = model.fc(feat)
        probs = F.softmax(output, dim=1)

        conf, pred = torch.max(probs, dim=1)

        for i in range(len(label)):
            true_label = label[i].item()
            pred_label = pred[i].item()
            confidence = conf[i].item()

            entry = {
                "idx": idx[i].item(),
                "label": true_label,
                "pred": pred_label,
                "conf": confidence
            }

            if pred_label == true_label:
                per_indiv_stats[true_label]["correct"].append(entry)
            else:
                per_indiv_stats[true_label]["wrong"].append(entry)

#selection of surest good prediction, less sure good prediction, surest wrong prediction, less sure wrong prediction
selected_images = []

for _, data in per_indiv_stats.items():
    correct = data["correct"]
    wrong = data["wrong"]

    if len(correct) > 0:
        best_correct = max(correct, key=lambda x: x["conf"])
        worst_correct = min(correct, key=lambda x: x["conf"])
        best_correct['name'] = "best_correct"
        worst_correct['name'] = "worst_correct"
        selected_images += [best_correct, worst_correct]

    if len(wrong) > 0:
        best_wrong = max(wrong, key=lambda x: x["conf"])
        worst_wrong = min(wrong, key=lambda x: x["conf"])
        best_wrong['name'] = "best_wrong"
        worst_wrong['name'] = "worst_wrong"
        selected_images += [best_wrong, worst_wrong]
                
for sample in selected_images:
    img = val_data.__getitem__(sample["idx"])[0]
    label = sample["label"]
    pred = sample["pred"]
    conf = sample["conf"]
    name = sample['name']
    print(label,pred,conf,name)

    out_dir_class = results_folder / class_names[label]
    os.makedirs(out_dir_class, exist_ok=True)

    save_grad_cam(
        base_model,
        class_names,
        img.view(1, 3, 224, 224),
        label,
        name,
        out_dir=out_dir_class
    )


