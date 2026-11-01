import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from pathlib import Path
import argparse

from elephantreid.identification import ClassificationHead


# ============================================================
# CONFIGURATION
# ============================================================

image_path = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/Ear/v3") / "SEB_183770_frame_000855_left_1718.png"
num_classes = 13

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# MODIFIER CES CHEMINS
# ------------------------------------------------------------

MODEL_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/models/identification/v8/Ear/model.pth"
)

OUTPUT_DIR = Path(
    "/pbs/home/a/ameurisse/results/visualize/resnet50/visualization_v8_v2"
)

N_TOP = 10


# ============================================================
# TRANSFORM = EXACTEMENT CELUI DE TON ENTRAINEMENT
# ============================================================

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

input_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normalize
])


# ============================================================
# DENORMALISATION UNIQUEMENT POUR AFFICHER L'IMAGE
# ============================================================

def denormalize(x):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    return (
        x.cpu() * std + mean
    ).clamp(0, 1)


# ============================================================
# CHARGEMENT MODELE
# ============================================================

def load_model(num_classes):

    model = models.resnet50(
        weights="DEFAULT"
    )

    model.fc = ClassificationHead(
        dim_out=num_classes,
        dim_in=2048
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# EXTRACTION DES ACTIVATIONS ET GRADIENTS
# ============================================================

class ActivationExtractor:

    def __init__(self, model):

        self.activations = {}
        self.gradients = {}

        self.layers = {
            "conv1": model.conv1,
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        }

        self.hooks = []

        for name, layer in self.layers.items():

            self.hooks.append(
                layer.register_forward_hook(
                    self.make_forward_hook(name)
                )
            )

            self.hooks.append(
                layer.register_full_backward_hook(
                    self.make_backward_hook(name)
                )
            )

    def make_forward_hook(self, name):

        def hook(module, input, output):

            self.activations[name] = output

        return hook

    def make_backward_hook(self, name):

        def hook(module, grad_input, grad_output):

            self.gradients[name] = grad_output[0]

        return hook

    def remove(self):

        for hook in self.hooks:
            hook.remove()


# ============================================================
# CALCUL CONTRIBUTION DES CARTES
# ============================================================

def get_top_maps(
    activation,
    gradient,
    n_top=10
):

    """
    activation : [1, C, H, W]
    gradient   : [1, C, H, W]

    On calcule pour chaque canal :

        poids = moyenne du gradient

        contribution =
            poids * activation

    Puis on garde les 10 canaux ayant
    la plus forte contribution positive.
    """

    # --------------------------------------------------------
    # Importance du canal
    # --------------------------------------------------------

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    # --------------------------------------------------------
    # Contribution spatiale
    # --------------------------------------------------------

    contribution = (
        weights * activation
    )

    # --------------------------------------------------------
    # On garde seulement ce qui pousse
    # vers la classe prédite
    # --------------------------------------------------------

    contribution_positive = F.relu(
        contribution
    )

    # --------------------------------------------------------
    # Score global de chaque carte
    # --------------------------------------------------------

    scores = contribution_positive.sum(
        dim=(2, 3)
    )[0]

    n_top = min(
        n_top,
        scores.shape[0]
    )

    top_scores, top_indices = torch.topk(
        scores,
        n_top
    )

    top_maps = activation[
        0,
        top_indices
    ]

    return (
        top_indices.detach().cpu(),
        top_maps.detach().cpu(),
        top_scores.detach().cpu()
    )


# ============================================================
# NORMALISATION POUR AFFICHAGE
# ============================================================

def normalize_map(x):

    x = x.numpy()

    min_val = x.min()
    max_val = x.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(x)

    return (
        x - min_val
    ) / (
        max_val - min_val
    )


# ============================================================
# AFFICHER LES 10 CARTES D'UN LAYER
# ============================================================

def save_layer(
    layer_name,
    activation,
    gradient,
    output_dir
):

    (
        indices,
        maps,
        scores
    ) = get_top_maps(
        activation,
        gradient,
        N_TOP
    )

    layer_dir = (
        output_dir /
        layer_name
    )

    layer_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # MONTAGE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(15, 6)
    )

    for ax, rank, channel, activation_map, score in zip(
        axes.flat,
        range(1, 11),
        indices,
        maps,
        scores
    ):

        activation_map = normalize_map(
            activation_map
        )

        ax.imshow(
            activation_map,
            cmap="viridis"
        )

        ax.set_title(
            f"#{rank}  channel {channel.item()}\n"
            f"contribution = {score.item():.2f}"
        )

        ax.axis("off")

    fig.suptitle(
        f"{layer_name} — "
        f"10 cartes contribuant le plus "
        f"à la prédiction",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        layer_dir / "top10.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PIPELINE
# ============================================================

def visualize(
    image_path,
    model,
    class_names,
    output_dir
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. IMAGE ORIGINALE
    # ========================================================

    original = Image.open(
        image_path
    ).convert("RGB")

    plt.figure(figsize=(6, 6))

    plt.imshow(original)

    plt.axis("off")

    plt.title("Image originale")

    plt.savefig(
        output_dir / "00_original.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 2. INPUT TRANSFORM
    # ========================================================

    x = input_transform(
        original
    )

    # --------------------------------------------------------
    # IMPORTANT :
    #
    # x est réellement l'entrée du réseau.
    #
    # Pour l'affichage seulement, on retire la normalisation.
    # --------------------------------------------------------

    x_display = denormalize(
        x
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(
        x_display.permute(1, 2, 0).numpy()
    )

    plt.axis("off")

    plt.title(
        "Image après input transform"
    )

    plt.savefig(
        output_dir / "01_input_transform.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 3. IMAGE -> RESEAU
    # ========================================================

    x = x.unsqueeze(0).to(DEVICE)

    extractor = ActivationExtractor(
        model
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    output = model(x)

    # ========================================================
    # 4. PREDICTION FINALE
    # ========================================================

    probabilities = torch.softmax(
        output,
        dim=1
    )

    pred_idx = (
        probabilities
        .argmax(dim=1)
        .item()
    )

    pred_probability = (
        probabilities[0, pred_idx]
        .item()
    )

    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(
        f"Classe : {class_names[pred_idx]}"
    )

    print(
        f"Probabilité : "
        f"{pred_probability:.4f}"
    )

    # ========================================================
    # 5. BACKPROPAGATION DE LA PREDICTION
    # ========================================================

    model.zero_grad()

    target = output[
        0,
        pred_idx
    ]

    target.backward()

    # ========================================================
    # 6. VISUALISATION DES GRANDS LAYERS
    # ========================================================

    for layer_name in [
        "conv1",
        "layer1",
        "layer2",
        "layer3",
        "layer4"
    ]:

        activation = (
            extractor.activations[
                layer_name
            ]
        )

        gradient = (
            extractor.gradients[
                layer_name
            ]
        )

        print(
            f"{layer_name}: "
            f"{tuple(activation.shape)}"
        )

        save_layer(
            layer_name,
            activation,
            gradient,
            output_dir
        )

    extractor.remove()

    print()
    print(
        f"Résultats enregistrés dans :\n"
        f"{output_dir}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # À remplacer par les vraies classes de ton dataset.
    # L'ordre doit être EXACTEMENT celui utilisé à l'entraînement.
    # --------------------------------------------------------

    class_names = [
        f"class_{i}"
        for i in range(num_classes)
    ]

    model = load_model(
        num_classes
    )

    visualize(
        image_path=image_path,
        model=model,
        class_names=class_names,
        output_dir=OUTPUT_DIR
    )