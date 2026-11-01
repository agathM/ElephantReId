# Summary:
# Train a classification head on crops with known individual identities,
# then predict the identities of crops with missing indiv_name values.
# ResNet50 features are computed once per image transformation and cached.
# Predictions are saved with the original crop metadata, predicted label,
# confidence, margin, and seed.

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms


# ============================================================
# PARAMETERS
# ============================================================

SEEDS = [1, 10, 42, 50, 60]

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

version = "v9"
crop_type = "Ear"
features_name = "brightness_normalize_resnet50"

BASE_PATH = Path("/sps/mnhn/ameurisse/ElephantReId")

# Input CSV containing crop metadata, crop_id and indiv_name.
CROPS_FILE = (
    BASE_PATH
    / "data/processed/crops/classifiers/split/split_seg_for_annotation.csv"
)

# Directory containing the crop images.
CROP_PATH = (
    BASE_PATH
    / "data/processed/crops/classifiers"
    / crop_type
    / "v5"
)

# Features are stored in a separate directory for each transformation.
FEATURES_PATH = (
    BASE_PATH
    / "data/processed/precomputed_features"
    / version
    / crop_type
    / features_name
)

# Output directory.
RESULTS_PATH = CROPS_FILE.parent

BATCH_SIZE = 64
NUM_EPOCHS = 60
LR = 0.001
DROPOUT = 0.5

# Image transformations.
LUMINANCE = np.array([0.299, 0.587, 0.114])


def transform_brightness_normalized(image, target_mean=0.5):
    """Normalize image mean luminance while preserving RGB channels."""

    array = np.asarray(image).astype(np.float32) / 255.0
    luminance = array @ LUMINANCE
    current_mean = luminance.mean()

    if current_mean == 0:
        return image

    factor = target_mean / current_mean
    normalized = np.clip(array * factor, 0, 1)

    return Image.fromarray((normalized * 255).astype(np.uint8))


TRANSFORMATIONS = [
    ("brightness_normalized", transform_brightness_normalized),
]


def build_input_transform(image_transform):
    """Build the preprocessing pipeline for ResNet50."""

    return transforms.Compose([
        image_transform,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def set_seed(seed):
    """Set random seeds for reproducible training."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASETS
# ============================================================

class GlobalCropDataset(Dataset):
    """Load crops for feature extraction."""

    def __init__(self, crop_path, crop_ids, transform):
        self.crop_path = crop_path
        self.crop_ids = crop_ids
        self.transform = transform

    def __len__(self):
        return len(self.crop_ids)

    def __getitem__(self, idx):
        image_path = self.crop_path / self.crop_ids[idx]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        return self.transform(image)


# ============================================================
# RESNET50 FEATURE EXTRACTION AND CACHE
# ============================================================

@torch.no_grad()
def compute_features(model, crop_path, crop_ids, input_transform):
    """Compute 2048-dimensional ResNet50 features."""

    dataset = GlobalCropDataset(
        crop_path=crop_path,
        crop_ids=crop_ids,
        transform=input_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    all_features = []

    for images in loader:
        images = images.to(DEVICE)
        features = model(images)
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def compute_or_load_features(
    model,
    crop_path,
    crop_ids,
    cache_dir,
    input_transform,
):
    """Load cached features and compute any missing ones."""

    cache_dir.mkdir(parents=True, exist_ok=True)

    features_file = cache_dir / "features.pt"
    crop_ids_file = cache_dir / "crop_ids.pt"

    if features_file.exists() and crop_ids_file.exists():
        cached_features = torch.load(
            features_file,
            map_location="cpu",
            weights_only=True,
        )
        cached_ids = torch.load(
            crop_ids_file,
            map_location="cpu",
            weights_only=False,
        )

        # Rebuild the cache if its dimensions are inconsistent.
        if len(cached_ids) != len(cached_features):
            cached_features = torch.empty((0, 2048))
            cached_ids = []
    else:
        cached_features = torch.empty((0, 2048))
        cached_ids = []

    # Retain only requested IDs, in their original cache order.
    requested_set = set(crop_ids)
    keep_indices = [
        i for i, crop_id in enumerate(cached_ids)
        if crop_id in requested_set
    ]

    cached_features = cached_features[keep_indices]
    cached_ids = [cached_ids[i] for i in keep_indices]

    cached_set = set(cached_ids)
    missing_ids = [
        crop_id for crop_id in crop_ids
        if crop_id not in cached_set
    ]

    if missing_ids:
        print(f"Computing features for {len(missing_ids)} crops...")

        new_features = compute_features(
            model,
            crop_path,
            missing_ids,
            input_transform,
        )

        cached_features = torch.cat(
            [cached_features, new_features],
            dim=0,
        )
        cached_ids.extend(missing_ids)

        torch.save(cached_features, features_file)
        torch.save(cached_ids, crop_ids_file)

    # Ensure the returned features follow the requested crop order.
    feature_index = {
        crop_id: i for i, crop_id in enumerate(cached_ids)
    }
    order = [feature_index[crop_id] for crop_id in crop_ids]

    return cached_features[order], crop_ids


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

class ClassificationHead(nn.Module):
    """Linear classification head with dropout regularization."""

    def __init__(self, dim_in, num_classes, dropout=0.5):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.linear(self.dropout(x))


def train_one_seed(
    seed,
    train_features,
    train_labels,
    num_classes,
):
    """Train the classifier on all crops with known identities."""

    set_seed(seed)

    model = ClassificationHead(
        dim_in=train_features.shape[1],
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    dataset = TensorDataset(train_features, train_labels)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    accuracies = []

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

        losses.append(total_loss / total_samples)
        accuracies.append(100 * total_correct / total_samples)

    return model, losses, accuracies


@torch.no_grad()
def predict(model, features):
    """Predict labels, top-class probabilities, and probability margins."""

    model.eval()

    all_probabilities = []

    loader = DataLoader(
        TensorDataset(features),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    for (batch_features,) in loader:
        batch_features = batch_features.to(DEVICE)
        logits = model(batch_features)
        probabilities = torch.softmax(logits, dim=1)

        all_probabilities.append(probabilities.cpu())

    probabilities = torch.cat(all_probabilities, dim=0)

    top_values, top_indices = probabilities.topk(
        k=min(2, probabilities.shape[1]),
        dim=1,
    )

    predicted_indices = top_indices[:, 0]
    confidence = top_values[:, 0]

    if probabilities.shape[1] >= 2:
        margin = top_values[:, 0] - top_values[:, 1]
    else:
        margin = top_values[:, 0]

    return predicted_indices, confidence, margin


# ============================================================
# MAIN
# ============================================================

def main():
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Reading crops from: {CROPS_FILE}")

    df = pd.read_csv(CROPS_FILE)

    required_columns = {"crop_id", "indiv_name"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in input CSV: {missing_columns}"
        )

    # Normalize crop IDs and distinguish known from unknown identities.
    df["crop_id"] = df["crop_id"].astype("string").str.strip()
    df["indiv_name"] = df["indiv_name"].astype("string").str.strip()

    known_mask = (
        df["split"]=="train"
    )

    unknown_mask = ~known_mask

    known_df = df.loc[known_mask].copy()
    unknown_df = df.loc[unknown_mask].copy()

    print(f"Rows in input CSV: {len(df)}")
    print(f"Crops with known identity: {len(known_df)}")
    print(f"Crops with missing identity: {len(unknown_df)}")

    if known_df.empty:
        raise ValueError("No crops with known indiv_name are available.")

    if unknown_df.empty:
        raise ValueError("No crops with missing indiv_name need prediction.")

    # Keep only rows with usable crop IDs and existing image files.
    df = df[df["crop_id"].notna() & df["crop_id"].ne("")].copy()

    df = df[
        df["crop_id"].map(
            lambda crop_id: (CROP_PATH / str(crop_id)).is_file()
        )
    ].copy()

    known_mask = (
        df["indiv_name"].notna()
        & df["indiv_name"].ne("")
    )

    known_df = df.loc[known_mask].copy()
    unknown_df = df.loc[~known_mask].copy()

    if known_df.empty or unknown_df.empty:
        raise ValueError(
            "After checking crop files, either the training set "
            "or the prediction set is empty."
        )

    # A crop ID must identify a single image.
    known_df = known_df.drop_duplicates("crop_id", keep="first")
    unknown_df = unknown_df.drop_duplicates("crop_id", keep="first")

    # Remove any accidental overlap between the training and prediction sets.
    known_ids = set(known_df["crop_id"])
    unknown_df = unknown_df[
        ~unknown_df["crop_id"].isin(known_ids)
    ].copy()

    if unknown_df.empty:
        raise ValueError("No distinct unknown crops remain to predict.")

    # Sort class names to keep a consistent label mapping.
    class_names = sorted(known_df["indiv_name"].unique().tolist())
    label_to_index = {
        name: index for index, name in enumerate(class_names)
    }

    num_classes = len(class_names)

    print(f"Number of known classes: {num_classes}")

    if num_classes < 2:
        raise ValueError(
            "At least two known individuals are needed to compute "
            "a meaningful classification margin."
        )

    # Load pretrained ResNet50 once.
    weights = models.ResNet50_Weights.DEFAULT
    base_model = models.resnet50(weights=weights)
    base_model.fc = nn.Identity()
    base_model = base_model.to(DEVICE).eval()

    all_predictions = []

    for transform_name, image_transform in TRANSFORMATIONS:
        print("\n" + "=" * 70)
        print(f"Transformation: {transform_name}")
        print("=" * 70)

        output_dir = RESULTS_PATH / transform_name
        output_dir.mkdir(parents=True, exist_ok=True)

        input_transform = build_input_transform(image_transform)

        # Compute or load features for all required crops.
        all_crop_ids = list(dict.fromkeys(
            known_df["crop_id"].tolist()
            + unknown_df["crop_id"].tolist()
        ))

        all_features, cached_ids = compute_or_load_features(
            model=base_model,
            crop_path=CROP_PATH,
            crop_ids=all_crop_ids,
            cache_dir=FEATURES_PATH / transform_name,
            input_transform=input_transform,
        )

        crop_to_index = {
            crop_id: i for i, crop_id in enumerate(cached_ids)
        }

        train_indices = torch.tensor(
            [crop_to_index[cid] for cid in known_df["crop_id"]],
            dtype=torch.long,
        )

        unknown_indices = torch.tensor(
            [crop_to_index[cid] for cid in unknown_df["crop_id"]],
            dtype=torch.long,
        )

        train_features = all_features[train_indices].float()
        unknown_features = all_features[unknown_indices].float()

        train_labels = torch.tensor(
            [
                label_to_index[name]
                for name in known_df["indiv_name"]
            ],
            dtype=torch.long,
        )

        # Train a separate classifier for every seed.
        for seed in SEEDS:
            print(f"\nTraining seed {seed}...")

            model, losses, accuracies = train_one_seed(
                seed=seed,
                train_features=train_features,
                train_labels=train_labels,
                num_classes=num_classes,
            )

            # Save training curves.
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(losses)
            axes[0].set_title("Training loss")
            axes[0].set_xlabel("Epoch")

            axes[1].plot(accuracies)
            axes[1].set_title("Training accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy (%)")

            fig.tight_layout()
            fig.savefig(
                output_dir / f"train_curves_seed_{seed}.png",
                dpi=150,
            )
            plt.close(fig)

            print(
                f"Final training loss: {losses[-1]:.4f} | "
                f"Training accuracy: {accuracies[-1]:.2f}%"
            )

            predicted_indices, confidence, margin = predict(
                model,
                unknown_features,
            )

            # Keep the original metadata for every predicted crop.
            predictions_df = unknown_df.copy().reset_index(drop=True)

            predictions_df["pred_label"] = [
                class_names[index]
                for index in predicted_indices.tolist()
            ]
            predictions_df["margin"] = margin.numpy()
            predictions_df["conf_pred_label"] = confidence.numpy()
            predictions_df["seed"] = seed
            predictions_df["transformation"] = transform_name

            output_file = (
                output_dir / f"predictions_missing_indiv_seed_{seed}.csv"
            )

            predictions_df.to_csv(output_file, index=False)

            all_predictions.append(predictions_df)

            print(f"Saved: {output_file}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Combine predictions from all seeds and transformations.
    if all_predictions:
        combined_df = pd.concat(all_predictions, ignore_index=True)

        combined_file = RESULTS_PATH / "all_predictions_missing_indiv.csv"
        combined_df.to_csv(combined_file, index=False)

        print("\n" + "=" * 70)
        print(f"Combined predictions saved to: {combined_file}")
        print(f"Total prediction rows: {len(combined_df)}")
        print("=" * 70)


if __name__ == "__main__":
    main()