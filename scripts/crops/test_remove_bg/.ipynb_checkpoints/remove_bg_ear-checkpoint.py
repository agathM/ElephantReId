"""
Remove the background of ear crops using MobileSAM.

Same as remove_background.py but uses the ear bounding box (from
v2_head_ears.csv) as the MobileSAM prompt instead of the body bounding box.

For 10 videos listed in v2_head_ears.csv, the script:
- keeps only detections whose label is Right_Ear or Left_Ear,
- keeps the most confident ear detection per (video, frame),
- extracts 3 frames per video (the first, the last and the middle frame present in the csv),
- segments the elephant with MobileSAM using the ear bounding box as prompt,
- replaces the background with a light blue color,
- saves a two-panel comparison (original left, background removed right) in test_remove_bg.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from mobile_sam import sam_model_registry, SamPredictor

# PARAMETERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_device(0)
print(DEVICE)
CROP_CSV_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/raw/v2_head_ears.csv")
VIDEO_ROOT = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw")
OUTPUT_DIR = Path("/pbs/home/a/ameurisse/test_remove_bg_v2")
N_VIDEOS = 10
EAR_LABELS = ["Right_Ear", "Left_Ear"]

MODEL_TYPE = "vit_t"
SAM_CHECKPOINT = Path("/pbs/home/a/ameurisse/ElephantReId/MobileSAM/weights/mobile_sam.pt")

LIGHT_BLUE_BGR = (230, 216, 173)

# Initializing the MobileSAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
sam.to(device=DEVICE)
sam.eval()
predictor = SamPredictor(sam)


def find_video(video_name: str) -> Path:
    for candidate in VIDEO_ROOT.rglob(f"{video_name}.mp4"):
        return candidate
    return None


def extract_frame(video_path: Path, frame_id: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


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
    background = np.full_like(image, LIGHT_BLUE_BGR)
    return np.where(mask[..., None], image, background)


def plot_comparison(image: np.ndarray, result: np.ndarray, output_path: Path,
                    x1: float, y1: float, x2: float, y2: float) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    ax_left.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
    ax_left.add_patch(rect)
    ax_left.set_title("Original")
    ax_left.axis("off")

    ax_right.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax_right.set_title("Without background")
    ax_right.axis("off")

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def main() -> None:
    ear = pd.read_csv(CROP_CSV_PATH)
    ear = ear[ear["label"].isin(EAR_LABELS)]
    best = ear.loc[ear.groupby(["video", "frame"])["conf"].idxmax()]
    best = best.drop_duplicates(subset=["video", "frame"])

    # videos = best["video"].drop_duplicates().head(N_VIDEOS)
    videos = ["SEB_180392", "SEB_180424", "SEB_180466","SEB_180356", "SEB_180361", "SEB_180362"                                                                                                                          "SEB_125081", "SEB_125090", "SEB_126178"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for video in videos:
        frames = sorted(best[best["video"] == video]["frame"].drop_duplicates().tolist())
        if len(frames)>=3:
            frame_ids = [frames[0], frames[len(frames) // 2], frames[-1]]
        else:
            frame_ids = frames

        for frame_id in frame_ids:
            row = best[(best["video"] == video) & (best["frame"] == frame_id)].iloc[0]
            video_path = find_video(video)
            if video_path is None:
                print(f"Missing video: {video}.mp4")
                continue

            frame = extract_frame(video_path, frame_id)
            if frame is None:
                print(f"Cannot read frame {frame_id} of {video}")
                continue

            mask = segment_elephant(frame, row["x1"], row["y1"], row["x2"], row["y2"])
            result = remove_background(frame, mask)

            output_path = OUTPUT_DIR / f"{video}_frame_{int(frame_id):06d}_ear.png"
            plot_comparison(frame, result, output_path, row["x1"], row["y1"], row["x2"], row["y2"])
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
