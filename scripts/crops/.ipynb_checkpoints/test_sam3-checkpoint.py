
# Summary:
# Segment an elephant's head throughout a video using SAM 3.
# Save binary masks and an annotated output video.

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from sam3.model_builder import build_sam3_video_predictor
import os
import psutil
import argparse

process = psutil.Process(os.getpid())


def print_memory(label):
    """Print current process RAM and CUDA memory usage."""
    ram_gb = process.memory_info().rss / (1024 ** 3)

    print(f"\n[{label}]")
    print(f"Process RAM: {ram_gb:.2f} GiB")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"CUDA allocated: {allocated:.2f} GiB")
        print(f"CUDA reserved:  {reserved:.2f} GiB")
        print(f"CUDA peak:      {peak:.2f} GiB")


# =========================
# Configuration
# =========================

# VIDEO_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw/wcf/LayonM2016-06-22_04-38-22.mp4")
# VIDEO_PATH = VIDEO_PATH.parent / "Savanne2016-02-27_10-51-05.mp4"
# VIDEO_PATH = VIDEO_PATH.parent / f"{VIDEO_PATH.stem}_2fps.mp4"
VIDEO_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras")
OUTPUT_DIR = Path("/pbs/home/a/ameurisse/ElephantReId/test")

PROMPT = "elephant ears"
PROMPT_FRAME = 0
OVERLAY_ALPHA = 0.45
MASK_THRESHOLD = 0.5

# =========================
# Initialization
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR = OUTPUT_DIR / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

def main():
    VIDEO_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras")
    OUTPUT_DIR = Path("/pbs/home/a/ameurisse/ElephantReId/test")
    
    PROMPT = "elephant ears"
    PROMPT_FRAME = 0
    OVERLAY_ALPHA = 0.45
    MASK_THRESHOLD = 0.5
    
    # =========================
    # Initialization
    # =========================
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR = OUTPUT_DIR / "masks"
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(
        description=(
            "test SAM3."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    args = parser.parse_args()
    VIDEO_PATH = VIDEO_PATH / args.video
    OUTPUT_VIDEO = OUTPUT_DIR / f"{PROMPT}_{VIDEO_PATH.stem}.mp4"
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #a partir trop gr nombre de frame -> out of memory
    cap.release()
    
    if not 0 <= PROMPT_FRAME < n_frames:
        raise ValueError(f"PROMPT_FRAME must be between 0 and {n_frames - 1}")
    
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {OUTPUT_VIDEO}")
    
    print_memory("Before model initialization")
    predictor = build_sam3_video_predictor()
    session_id = None
    print_memory("After model initialization")
    try:
        print("response")
        response = predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": str(VIDEO_PATH.resolve()),
            }
        )
        print(response)
        session_id = response["session_id"]
    
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": PROMPT_FRAME,
                "text": PROMPT,
            }
        )
    
        frame_masks = {}
    
        propagation = predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_index": PROMPT_FRAME,
                "propagation_direction": "forward",
            }
        )
    
        for result in tqdm(propagation, desc="Segmenting"):
            frame_idx = result["frame_index"]
            print(frame_idx)
            if frame_idx%100==0:
                print_memory(f"{frame_idx}")
            outputs = result["outputs"]
    
            masks = outputs.get("out_binary_masks")
            if masks is None:
                continue
    
            masks = np.asarray(masks)
            if masks.ndim == 2:
                masks = masks[None, ...]
    
            combined = np.zeros((height, width), dtype=np.uint8)
    
            for mask in masks:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                combined[mask > 0] = 255
    
            frame_masks[frame_idx] = combined
    
        cap = cv2.VideoCapture(str(VIDEO_PATH))
    
        for frame_idx in tqdm(range(n_frames), desc="Saving"):
            ok, frame = cap.read()
            if not ok:
                break
    
            mask = frame_masks.get(
                frame_idx,
                np.zeros((height, width), dtype=np.uint8),
            )
    
            # cv2.imwrite(
            #     str(MASK_DIR / f"frame_{frame_idx:06d}.png"),
            #     mask,
            # )
    
            foreground = mask > 0
            frame[foreground] = (
                frame[foreground] * (1 - OVERLAY_ALPHA)
                + np.array([0, 255, 0]) * OVERLAY_ALPHA
            ).astype(np.uint8)
    
            writer.write(frame)
    
        cap.release()
    
        print(f"Video saved to: {OUTPUT_VIDEO}")
        print(f"Masks saved to: {MASK_DIR}")
    
    finally:
        writer.release()
    
        if session_id is not None:
            predictor.handle_request(
                request={
                    "type": "close_session",
                    "session_id": session_id,
                }
            )
if __name__=="__main__":
    main()
