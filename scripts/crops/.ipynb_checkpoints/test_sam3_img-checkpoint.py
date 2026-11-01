
"""
Summary:
Segment every image in an input folder with SAM 3 using multiple text prompts.
Save individual binary masks and overlay images for each image and prompt.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/raw/seg_images"
)

OUTPUT_DIR = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/results/image_segmentations"
)

PROMPTS = [
    "elephant head",
    # "elephant ears",
]

MASK_THRESHOLD = 0.5
OVERLAY_ALPHA = 0.45

# Supported image extensions.
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

# ============================================================
# Output directories
# ============================================================

MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def safe_name(text):
    """Convert a prompt into a filesystem-friendly name."""
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def save_binary_mask(mask, output_path):
    """Save a binary segmentation mask as a PNG image."""
    mask = np.asarray(mask).squeeze()

    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask dimensions: {mask.shape}")

    binary = (mask > MASK_THRESHOLD).astype(np.uint8) * 255
    cv2.imwrite(str(output_path), binary)


def create_overlay(image_bgr, mask):
    """Overlay a segmentation mask in green on the original image."""
    result = image_bgr.copy()
    foreground = mask > 0

    green = np.array([0, 255, 0], dtype=np.float32)

    result[foreground] = (
        result[foreground].astype(np.float32) * (1 - OVERLAY_ALPHA)
        + green * OVERLAY_ALPHA
    ).astype(np.uint8)

    return result


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    image_paths = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No images found in {INPUT_DIR}")

    print(f"Images found: {len(image_paths)}")
    print(f"Prompts: {PROMPTS}")

    # Initialise SAM 3 once and reuse the model for all images.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_sam3_image_model()
    model.to(device)
    model.eval()

    processor = Sam3Processor(model)

    for image_path in tqdm(image_paths, desc="Processing images"):
        # Preserve the relative folder structure to avoid filename collisions.
        relative_path = image_path.relative_to(INPUT_DIR)
        image_stem = relative_path.with_suffix("")

        image_mask_dir = MASK_DIR
        image_overlay_dir = OVERLAY_DIR

        image_mask_dir.mkdir(parents=True, exist_ok=True)
        image_overlay_dir.mkdir(parents=True, exist_ok=True)

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print(f"Warning: cannot read {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Encode the image once, then query it with every prompt.
        with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ),
            ):
                inference_state = processor.set_image(pil_image)
            
                for prompt in PROMPTS:
                    prompt_name = safe_name(prompt)
                    output = processor.set_text_prompt(
                        state=inference_state,
                        prompt=prompt,
                    )

                    masks = output.get("masks")
    
                    if masks is None or len(masks) == 0:
                        print(f"No masks: {image_path.name} | {prompt}")
                        continue
    
                    masks = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    
                    # Normalise mask dimensions to (N, H, W).
                    if masks.ndim == 2:
                        masks = masks[None, ...]
    
                    height, width = image_bgr.shape[:2]
    
                    # Combine all detected objects for this prompt.
                    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
                    for mask_index, mask in enumerate(masks):
                        mask = np.squeeze(mask)
    
                        if mask.ndim != 2:
                            continue
    
                        if mask.shape != (height, width):
                            mask = cv2.resize(
                                mask.astype(np.float32),
                                (width, height),
                                interpolation=cv2.INTER_NEAREST,
                            )
    
                        binary = (mask > MASK_THRESHOLD).astype(np.uint8) * 255
                        combined_mask[binary > 0] = 255
    
                        # Also save each detected object separately.
                        individual_path = (
                            image_mask_dir
                            / prompt_name / f"{image_stem}_object_{mask_index:02d}.png"
                        )
                        individual_path.parent.mkdir(parents=True, exist_ok=True)
                        save_binary_mask(mask, individual_path)
    
                    # Save the combined mask for this image and prompt.
                    # combined_path = (
                    #     image_mask_dir / prompt_name / f"{image_stem}.png"
                    # )
                    # combined_path.parent.mkdir(parents=True, exist_ok=True)
                    # cv2.imwrite(str(combined_path), combined_mask)
    
                    # Save a visual overlay.
                    overlay = create_overlay(image_bgr, combined_mask)
    
                    overlay_path = (
                        image_overlay_dir / prompt_name / f"{image_stem}.jpg"
                    )
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(overlay_path), overlay)

    print("\nProcessing complete.")
    print(f"Masks:   {MASK_DIR}")
    print(f"Overlays: {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
