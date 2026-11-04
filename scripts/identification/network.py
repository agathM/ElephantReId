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

# def save_grad_cam(model: models.ResNet, classes : dict, image: torch.tensor, label:int, out_name:str) -> None:
#     """
#     Displays Grad-CAM for top-1 classes + Lowest class.
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
#     probs = torch.softmax(output, dim=1)
#     values,indexes = torch.topk(probs, 2) #best predictions
#     conf = values[:,0]
#     margin = values[:,0] - values[:,1]
#     # v, i = torch.topk(output,1,largest=False) #worst prediction
#     # indexes = indexes.tolist()[0] +  i.tolist()[0]
#     # values = values.tolist()[0] +  v.tolist()[0]
#     plot_image = (image[0].permute(1,2,0).detach().cpu()).clone()
#     plot_image-= plot_image.min()
#     plot_image/=plot_image.max()
#     extent = 0,224,0,224
#     sp = 1
#     plt.figure(figsize=(4,4))
#     plt.subplot(1,3,1)
#     plt.imshow(plot_image,extent=extent)
#     plt.title(classes[label])
#     for i,v in zip(indexes,values):
#         sp+=1
#         model.zero_grad()
#         output[0,i].backward(retain_graph=True)
#         alpha = gradients.mean(dim=(2,3),keepdim=True) #for each image along depth axis, compute gradient of class v of each activations (i,j) and take the mean -> #depth alpha
#         #alpha k is the relative importance of image k in the predicted output
#         grad_cam = F.relu((alpha*activations).sum(dim=(1),keepdim=True)) # bigger activations + bigger relative importance of image k -> bigger values
#         grad_cam -= grad_cam.min()
#         grad_cam /= grad_cam.max()
#         grad_cam = grad_cam[0,0].detach().cpu()
#         plt.subplot(2,1,sp)
#         plt.imshow(plot_image,extent=extent)
#         plt.imshow(grad_cam,cmap="viridis",alpha=0.5,interpolation="bilinear",extent=extent) # interpolation
#         plt.title(f"{classes[i]} : {v:.2f}")
#     #save figure
#     plt.tight_layout()
#     frame_name = "_".join(outpath.stem.split("_")[1:5])
#     new_name = (
#         f"conf_{conf:.2f}"
#         f"_margin_{margin:.2f}"
#         f"_{frame_name}"
#         f"{out_name.suffix}"
#     )
#     plt.savefig(out_name.parent / new_name, dpi=300, bbox_inches="tight")
#     plt.close()
#     handle_forward.remove(), handle_backward.remove()

def save_grad_cam(
    model: models.ResNet,
    classes: dict,
    image: torch.Tensor,
    label: int,
    out_name,
) -> None:
    """
    Save the original image and Grad-CAM overlays for the top-2
    predicted classes.
    """

    device = next(model.parameters()).device
    model.eval()

    image = image.to(device)
    extent = 0,224,0,224

    # --------------------------------------------------------------
    # Target the deepest conv block actually used to produce the
    # classification features. Depending on the trained head, the
    # features come from layer3 (dim 1024) or layer4 (dim 2048) ;
    # we forward only up to that block, then apply avgpool + head.
    # --------------------------------------------------------------
    fc_in_features = (
        model.fc.linear.in_features
        if hasattr(model.fc, "linear")
        else model.fc.in_features
    )
    if fc_in_features == 1024:
        target_block, use_layer4 = model.layer3, False
    elif fc_in_features == 2048:
        target_block, use_layer4 = model.layer4, True
    else:
        raise ValueError(
            f"Unsupported classification head input dim: {fc_in_features} "
            "(expected 1024 or 2048)."
        )

    target_conv = target_block[len(target_block) - 1]

    activations = None

    def forward_hook(module, inputs, output):
        nonlocal activations
        activations = output

    handle = target_conv.register_forward_hook(forward_hook)

    try:
        # ----------------------------------------------------------
        # Partial forward: conv1..(layer3|layer4) then avgpool + head
        # ----------------------------------------------------------
        x = model.conv1(image)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        if use_layer4:
            x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        output = model.fc(x)

        if activations is None:
            raise RuntimeError("Activations were not captured.")

        probs = torch.softmax(output, dim=1)

        # Top-2 predictions
        values, indexes = torch.topk(
            probs,
            k=2,
            dim=1,
        )
        conf = values[:,0].detach().cpu().item()
        margin = (values[:,0] - values[:,1]).detach().cpu().item()

        # ----------------------------------------------------------
        # Original image
        # ----------------------------------------------------------
        plot_image = (
            image[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .clone()
        )

        plot_image -= plot_image.min()

        if plot_image.max() > 0:
            plot_image /= plot_image.max()

        # ----------------------------------------------------------
        # Figure
        # ----------------------------------------------------------
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )

        # ----------------------------------------------------------
        # Original image
        # ----------------------------------------------------------
        axes[0].imshow(plot_image)
        axes[0].set_title(
            f"Original\nGT: {classes[int(label)]}"
        )
        axes[0].axis("off")
        
        for col, (class_idx, confidence) in enumerate(
            zip(indexes[0], values[0]),
            start=1,
        ):
            class_idx = int(class_idx)
        
            gradients = torch.autograd.grad(
                outputs=output[0, class_idx],
                inputs=activations,
                retain_graph=True,
            )[0]
        
            alpha = gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )
        
            grad_cam = (
                alpha * activations
            ).sum(dim=1)
        
            grad_cam = F.relu(grad_cam)
            grad_cam = grad_cam[0].detach().cpu()
        
            min_val = grad_cam.min()
            max_val = grad_cam.max()
        
            if max_val > min_val:
                grad_cam = (
                    grad_cam - min_val
                ) / (max_val - min_val)
            else:
                grad_cam.zero_()
        
            axes[col].imshow(plot_image,extent=extent)
            axes[col].imshow(
                grad_cam,
                cmap="viridis",
                alpha=0.5,
                interpolation="bilinear",extent=extent
            )
        
            axes[col].set_title(
                f"{classes[class_idx]} : {float(confidence.detach().cpu()):.2f}"
            )
            axes[col].axis("off")
        
        plt.tight_layout()
        crop_id = out_name.stem
        if crop_id.startswith("v3_"):
            crop_id = crop_id[len("v3_"):]
        new_name = (
            f"conf_{conf:.2f}"
            f"_margin_{margin:.2f}"
            f"_{crop_id}"
            f"{out_name.suffix}"
        )
        plt.savefig(out_name.parent / new_name, dpi=300, bbox_inches="tight")
        plt.close(fig)
    finally:
        handle.remove()


