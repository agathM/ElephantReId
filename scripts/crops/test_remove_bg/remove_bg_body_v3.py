"""
Remove the background of body crops using MobileSAM.

For every crop in body/v2, the script:
- parses the video and frame id from the crop filename,
- gets the elephant bounding box from v2_body.csv, linked by video and frame columns,
- segments the elephant with MobileSAM using the bounding box as prompt,
- sets every background pixel to 0 (black),
- saves the cropped result in body/v3 with the same filename.
"""

import re
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from mobile_sam import sam_model_registry, SamPredictor

# PARAMETERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_device(0)
print(DEVICE)
BODY_CSV_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/raw/v2_body.csv")
INPUT_DIR = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/body/v2")
OUTPUT_DIR = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/body/v3")

MODEL_TYPE = "vit_t"
SAM_CHECKPOINT = Path("/pbs/home/a/ameurisse/ElephantReId/MobileSAM/weights/mobile_sam.pt")

CROP_NAME_RE = re.compile(r"video_(?P<video>.+)_frame_(?P<frame>\d+)_body_(?P<idx>\d+)\.png")

# Initializing the MobileSAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
sam.to(device=DEVICE)
sam.eval()
predictor = SamPredictor(sam)


def segment_elephant(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return np.ones((h, w), dtype=bool)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array([x1, y1, x2, y2])[None, :],
        multimask_output=False,
    )
    return masks[0] > 0.5


def remove_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask[..., None], image, 0).astype(np.uint8)


def main() -> None:
    body = pd.read_csv(BODY_CSV_PATH)
    body = body.drop_duplicates(subset=["video", "frame"])
    body_lookup = {(row["video"], row["frame"]): row for _, row in body.iterrows()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    crop_paths = sorted(INPUT_DIR.glob("video_*_frame_*_body_*.png"))
    skipped = 0

    for crop_path in tqdm(crop_paths, desc="remove background"):
        match = CROP_NAME_RE.match(crop_path.name)
        if match is None:
            skipped += 1
            print(f"Cannot parse filename: {crop_path.name}")
            continue

        video = match.group("video")
        frame = int(match.group("frame"))

        row = body_lookup.get((video, frame))
        if row is None:
            skipped += 1
            print(f"Missing csv row for {crop_path.name}")
            continue

        image = cv2.imread(str(crop_path))
        if image is None:
            skipped += 1
            print(f"Cannot read image: {crop_path}")
            continue

        # The crop is already the bounding box region, so the prompt box in
        # crop coordinates spans the full image.
        box_x1, box_y1 = 0.0, 0.0
        box_x2, box_y2 = float(row["x2"]) - float(row["x1"]), float(row["y2"]) - float(row["y1"])

        mask = segment_elephant(image, box_x1, box_y1, box_x2, box_y2)
        result = remove_background(image, mask)

        output_path = OUTPUT_DIR / crop_path.name
        cv2.imwrite(str(output_path), result)

    print(f"Done: {len(crop_paths) - skipped} saved, {skipped} skipped")


if __name__ == "__main__":
    main()
