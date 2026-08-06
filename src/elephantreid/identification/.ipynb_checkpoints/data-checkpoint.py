import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models,transforms,datasets
import torch
from pathlib import Path
import copy


PRECOMPUTED_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/precomputed_features")


# class CustomImageDataset(Dataset):
#     """from a file containing all crop_path and metadata
#     generate a train or val dataset"""
#     def __init__(self, img_file:str, save_path:str, mode:str='train', transform=None):
#         self.df = pd.read_csv(img_file,header=0)
#         self.samples = []
#         self.transform = transform
#         self.df.sort_values(['indiv_name','entry','crop_path'],inplace=True)
#         indivs = self.df['indiv_name'].unique()
#         for indiv in indivs:
#             entries = self.df[self.df['indiv_name']==indiv]['entry'].unique()
#             if len(entries)>1:
#                 train_entries, val_entries = train_test_split(
#                     entries,
#                     test_size=0.25,
#                     random_state=42
#                     # ,stratify=True
#                 )
#                 if mode=='train':
#                     mask = self.df["entry"].isin(train_entries) & (self.df["indiv_name"] == indiv)
#                 else:
#                     mask = self.df["entry"].isin(val_entries) & (self.df["indiv_name"] == indiv)
#                 for _, row in self.df[mask].iterrows():
#                     self.samples.append(
#                         {
#                             "crop_path": row["crop_path"],
#                             "identity" : indiv,
#                             "metadata" : row.to_dict()
#                         }
#                     )
                
#         identities = sorted({sample['identity'] for sample in self.samples})
#         self.label_map = {identity: idx for idx, identity in enumerate(identities)}
#         self.classes_nb = len(self.label_map)
#         self.classes = identities
#         #SAVE SAMPLES TO CSV
#         samples_df = pd.DataFrame([
#             {
#                 "crop_path": s["crop_path"],
#                 "identity": s["identity"],
#                 **s["metadata"] #dictionnary unpacking
#             }
#             for s in self.samples
#         ])
        
#         samples_df.to_csv(os.path.join(save_path,f"{mode}_data.csv"), index=True)
        
#         print(f"[INFO] Saved samples to {save_path}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path = self.samples[idx]['crop_path']
#         image = Image.open(img_path).convert("RGB")
#         label = self.label_map[self.samples[idx]['identity']]
#         if self.transform:
#             image = self.transform(image)
#         return image, label, idx

class CustomImageDataset(Dataset):

    def __init__(
        self,
        img_file,
        crop_path,
        crop_type,
        split="train",
        transform=None,
        label_map=None,
        mode=None
    ):

        self.df = pd.read_excel(
            img_file,
            sheet_name="samples",
            engine="odf"
        )
        if mode:
            self.df = self.df[self.df['mode']==mode]
        self.df = self.df[
            self.df["split"] == split
        ].reset_index(drop=True)
        print(f"#{split} crops by individuals")
        summary = (
            self.df.groupby("indiv_name")
            .agg(
                nb_crops=("indiv_name", "size"),
                nb_entries=("entry", "nunique"),
                nb_videos=("video", "nunique"),
            )
            .sort_values("nb_crops", ascending=False)
        )
        
        print(summary)

        self.samples = self.df.to_dict("index")
        self.transform = transform

        self.label_map = label_map
        self.classes = sorted(label_map.keys())
        self.classes_nb = len(label_map)
        self.crop_path = crop_path
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.crop_path / self.samples[idx]['crop_id']
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[self.samples[idx]['indiv_name']]
        if self.transform:
            image = self.transform(image)
        return image, label, idx
        
def precompute_features_14th(
    model: models.ResNet, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is f(g(x)) where f is the last layer and g is
    the rest of the model, it is not necessary to recompute g(x) at
    each epoch as g is fixed. Hence you can precompute g(x) and
    create a new dataset

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation

    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    loader = DataLoader(dataset)
    model.eval() #no dropout no Batchnorm
    all_features = [] ; all_labels = [] ; all_idx = []
    def hook(module,input,output):
        feat = F.adaptive_max_pool2d(output, (6, 6))
        feat = feat.flatten(1)
        feat = feat / (feat.norm(p=2, dim=1, keepdim=True)+1e-12)
        all_features.append(feat.cpu())
    handle = model.layer4[0].register_forward_hook(hook)
    with torch.no_grad(): #no gradients and no graph
        for images, labels, idx in loader:
            images = images.to(device)
            _ = model(images)
            all_labels.append(labels.cpu())
            all_idx.append(idx.cpu())
    handle.remove()
    all_features = torch.cat(all_features,dim=0) ; all_labels = torch.cat(all_labels,dim=0)
    data2 = TensorDataset(all_features,all_labels)
    return data2

def precompute_features(
    model: models.ResNet, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features before fc layer precomputed by the model.
    the features should have the same idx as the input dataset
    """
    loader = DataLoader(dataset)
    model.eval() #no dropout no Batchnorm
    all_features = [] ; all_labels = [] ; all_idx = []
    def hook(module,input,output):
        all_features.append(input[0].to(device)) #small dataset can be on GPU
    handle = model.fc.register_forward_hook(hook)
    with torch.no_grad(): #no gradients and no graph
        for images, labels, idx in loader:
            images = images.to(device)
            _ = model(images)
            all_labels.append(labels.to(device)) #small dataset can be on GPU
            all_idx.append(idx)
    handle.remove()
    all_features = torch.cat(all_features,dim=0) ; all_labels = torch.cat(all_labels,dim=0)
    all_idx = torch.cat(all_idx,dim=0)
    data2 = TensorDataset(all_features,all_labels,all_idx)
    return data2

# def load_data(model:models.ResNet, input_transform, crop_path: Path, file: str, crop_type:str, device: torch.device, precomputed_folder,label_map:None):
#     """load precomputed features for training only head classification
#     ResNet50 model"""
#     precomputed_folder = PRECOMPUTED_PATH / precomputed_folder
#     if not os.path.exists(os.path.join(precomputed_folder,'val_data.pt')):
#         train_data = CustomImageDataset(
#             Path(file),
#             crop_path,
#             crop_type,
#             transform=input_transform,
#             label_map=label_map
#         )
        
#         val_data = CustomImageDataset(
#             Path(file),
#             crop_path,
#             crop_type,
#             split="val",
#             transform=input_transform,
#             label_map=label_map
#         )
#         num_classes = train_data.classes_nb
        
#         #FEATURE COMPUTATION
#         model.to(device)
#         print("train size:", len(train_data))
#         print("val size:", len(val_data))
#         train_features_dataset = precompute_features(model, train_data, device=device)
#         val_features_dataset = precompute_features(model, val_data, device=device)
#         os.makedirs(precomputed_folder,exist_ok=True)
#         torch.save(train_features_dataset,os.path.join(precomputed_folder,'train_feat.pt'))
#         torch.save(val_features_dataset,os.path.join(precomputed_folder,'val_feat.pt'))
#         torch.save(train_data,os.path.join(precomputed_folder,'train_data.pt'))
#         torch.save(val_data,os.path.join(precomputed_folder,'val_data.pt'))
#     else:
#         train_features_dataset = torch.load(os.path.join(precomputed_folder,'train_feat.pt'),weights_only=False)
#         val_features_dataset = torch.load(os.path.join(precomputed_folder,'val_feat.pt'),weights_only=False)
#         train_data = torch.load(os.path.join(precomputed_folder,'train_data.pt'),weights_only=False)
#         val_data = torch.load(os.path.join(precomputed_folder,'val_data.pt'),weights_only=False)
#     return train_features_dataset, val_features_dataset, train_data, val_data


def load_data(model:models.ResNet, input_transform, crop_path: Path, file: str, crop_type:str, device: torch.device, precomputed_folder,mode):

    precomputed_folder = PRECOMPUTED_PATH / precomputed_folder

    label_map_path = os.path.join(precomputed_folder, "label_map.pt")

    if not os.path.exists(label_map_path):

        # Create a global label map
        df = pd.read_excel(
            file,
            sheet_name="samples",
            engine="odf",
            header=0
        )
        if mode:
            df = df[df['mode']==mode]

        identities = sorted(df["indiv_name"].unique())

        label_map = {
            identity: idx 
            for idx, identity in enumerate(identities)
        }

        train_data = CustomImageDataset(
            Path(file),
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

        num_classes = len(label_map)

        # FEATURE COMPUTATION
        model.to(device)

        print("train size:", len(train_data))
        print("val size:", len(val_data))
        print("classes:", label_map)

        train_features_dataset = precompute_features(
            model,
            train_data,
            device=device
        )

        val_features_dataset = precompute_features(
            model,
            val_data,
            device=device
        )

        os.makedirs(precomputed_folder, exist_ok=True)

        torch.save(
            train_features_dataset,
            os.path.join(precomputed_folder,'train_feat.pt')
        )

        torch.save(
            val_features_dataset,
            os.path.join(precomputed_folder,'val_feat.pt')
        )

        torch.save(
            train_data,
            os.path.join(precomputed_folder,'train_data.pt')
        )

        torch.save(
            val_data,
            os.path.join(precomputed_folder,'val_data.pt')
        )

        # Save label mapping
        torch.save(
            label_map,
            label_map_path
        )

    else:
        train_features_dataset = torch.load(
            os.path.join(precomputed_folder,'train_feat.pt'),
            weights_only=False
        )

        val_features_dataset = torch.load(
            os.path.join(precomputed_folder,'val_feat.pt'),
            weights_only=False
        )

        train_data = torch.load(
            os.path.join(precomputed_folder,'train_data.pt'),
            weights_only=False
        )

        val_data = torch.load(
            os.path.join(precomputed_folder,'val_data.pt'),
            weights_only=False
        )

        # Load label mapping
        label_map = torch.load(
            label_map_path,
            weights_only=False
        )

        print("Loaded label map:", label_map)

    return train_features_dataset, val_features_dataset, train_data, val_data, label_map