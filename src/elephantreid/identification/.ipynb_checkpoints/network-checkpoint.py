import os
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import torch
from torchvision import models, datasets, transforms
import seaborn as sns
import numpy as np
class ClassificationHead(nn.Module):
    def __init__(self,dim_out,dim_in):
        super(ClassificationHead, self).__init__()
        self.out_features = dim_out
        # self.linear1 = nn.Linear(in_features=dim_in, out_features=512)
        # self.linear2 = nn.Linear(in_features=512, out_features=128)
        # self.relu = nn.ReLU()
        # self.linear3 = nn.Linear(in_features=128, out_features=self.out_features)
        self.linear = nn.Linear(in_features=dim_in, out_features=self.out_features)

    def forward(self, x):
        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu(x)
        return self.linear(x)

def save_confusion_matrix(cm, class_names,out_dir,police_size=8):
    plt.figure(figsize=(8, 8))
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, cm_sum, where=cm_sum != 0)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        annot_kws={"size": police_size}, #taille police
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Normalized confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'confusion_matrix_normalized'), dpi=300, bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        annot_kws={"size": police_size},
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'confusion_matrix'), dpi=300, bbox_inches="tight")
    plt.close()
    

# def save_grad_cam(model: models.ResNet, classes : dict, image: torch.tensor, label:int, pred_name:str, out_dir:str) -> None:
#     """
#     Displays Grad-CAM for top-3 classes + Lowest class.
#     """
#     activations, gradients = None, None
#     device = next(model.parameters()).device
#     image = image.to(device)
#     ### dernière couche de convolution
#     last_name = None
#     last_conv = None
#     flag = False
#     for name, m in model.named_modules():
#         if isinstance(m,nn.Conv2d):
#             last_name = name
#             last_conv = m
#     for name, param in model.named_parameters():
#         name = '.'.join(name.split('.')[:-1])
#         if name == last_name:
#             flag=True
#         param.requires_grad = flag
#     def forward_hook(module,input,output):
#         ### hook qui affecte à l'activation de la couche
#         nonlocal activations
#         activations = output
    
#     def backward_hook(module,grad_input,grad_output):
#         ### hook qui affecte à gradients la dérivée de la sortie par rapport à l'activation de la couche module
#         nonlocal gradients
#         gradients = grad_output[0] #tuple gradient par rapport aux sorties (ici une)
    
#     handle_forward = last_conv.register_forward_hook(forward_hook)
#     handle_backward = last_conv.register_full_backward_hook(backward_hook)
#     model.eval()
#     output = model(image)
#     values,indexes = torch.topk(output, 3)
#     v, i = torch.topk(output,1,largest=False)
#     indexes = indexes.tolist()[0] +  i.tolist()[0]
#     values = values.tolist()[0] +  v.tolist()[0]
#     plot_image = (image[0].permute(1,2,0).detach().cpu()).clone()
#     plot_image-= plot_image.min()
#     plot_image/=plot_image.max()
#     extent = 0,224,0,224
#     sp = 3
#     plt.figure(figsize=(6,6))
#     plt.subplot(3,3,1)
#     plt.imshow(plot_image,extent=extent)
#     plt.title(classes[label])
#     for i,v in zip(indexes,values):
#         model.zero_grad()
#         output[0,i].backward(retain_graph=True)
#         alpha = gradients.mean(dim=(2,3),keepdim=True) #for each image along depth axis, compute gradient of class v of each activations (i,j) and take the mean -> #depth alpha
#         #alpha k is the relative importance of image k in the predicted output
#         grad_cam = F.relu((alpha*activations).sum(dim=(1),keepdim=True)) # bigger activations + bigger relative importance of image k -> bigger values
#         if grad_cam.max()==grad_cam.min():
#             print(grad_cam.min(),classes[label],pred_name)
#         grad_cam -= grad_cam.min()
#         grad_cam /= grad_cam.max()
#         grad_cam = grad_cam[0,0].detach().cpu()
#         plt.subplot(3,3,sp)
#         plt.imshow(plot_image,extent=extent)
#         plt.imshow(grad_cam,cmap="viridis",alpha=0.5,interpolation="bilinear",extent=extent) # interpolation
#         plt.title(f"{classes[i]} : {v:.2f}")
#         sp+=1
#     #save figure
#     out_path = os.path.join(out_dir,f"{pred_name}.jpg")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     handle_forward.remove(), handle_backward.remove()

def save_grad_cam(model: models.ResNet, classes : dict, image: torch.tensor, label:int, out_name:str) -> None:
    """
    Displays Grad-CAM for top-1 classes + Lowest class.
    """
    activations, gradients = None, None
    device = next(model.parameters()).device
    image = image.to(device)
    ### dernière couche de convolution
    last_name = None
    last_conv = None
    flag = False
    for name, m in model.named_modules():
        if isinstance(m,nn.Conv2d):
            last_name = name
            last_conv = m
    for name, param in model.named_parameters():
        name = '.'.join(name.split('.')[:-1])
        if name == last_name:
            flag=True
        param.requires_grad = flag
    def forward_hook(module,input,output):
        ### hook qui affecte à l'activation de la couche
        nonlocal activations
        activations = output
    
    def backward_hook(module,grad_input,grad_output):
        ### hook qui affecte à gradients la dérivée de la sortie par rapport à l'activation de la couche module
        nonlocal gradients
        gradients = grad_output[0] #tuple gradient par rapport aux sorties (ici une)
    
    handle_forward = last_conv.register_forward_hook(forward_hook)
    handle_backward = last_conv.register_full_backward_hook(backward_hook)
    model.eval()
    output = model(image)
    values,indexes = torch.topk(output, 1)
    v, i = torch.topk(output,1,largest=False)
    indexes = indexes.tolist()[0] +  i.tolist()[0]
    values = values.tolist()[0] +  v.tolist()[0]
    plot_image = (image[0].permute(1,2,0).detach().cpu()).clone()
    plot_image-= plot_image.min()
    plot_image/=plot_image.max()
    extent = 0,224,0,224
    sp = 2
    plt.figure(figsize=(4,4))
    plt.subplot(2,2,1)
    plt.imshow(plot_image,extent=extent)
    plt.title(classes[label])
    for i,v in zip(indexes,values):
        model.zero_grad()
        output[0,i].backward(retain_graph=True)
        alpha = gradients.mean(dim=(2,3),keepdim=True) #for each image along depth axis, compute gradient of class v of each activations (i,j) and take the mean -> #depth alpha
        #alpha k is the relative importance of image k in the predicted output
        grad_cam = F.relu((alpha*activations).sum(dim=(1),keepdim=True)) # bigger activations + bigger relative importance of image k -> bigger values
        grad_cam -= grad_cam.min()
        grad_cam /= grad_cam.max()
        grad_cam = grad_cam[0,0].detach().cpu()
        plt.subplot(2,2,sp)
        plt.imshow(plot_image,extent=extent)
        plt.imshow(grad_cam,cmap="viridis",alpha=0.5,interpolation="bilinear",extent=extent) # interpolation
        plt.title(f"{classes[i]} : {v:.2f}")
        sp+=1
    #save figure
    plt.tight_layout()
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close()
    handle_forward.remove(), handle_backward.remove()