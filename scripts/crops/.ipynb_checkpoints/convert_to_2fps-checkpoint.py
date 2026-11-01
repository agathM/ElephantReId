"""
Summary:
Convert a video to 2 FPS by keeping one frame every 0.5 seconds.
Frames are decoded sequentially to avoid loading the entire video into RAM.
"""

from pathlib import Path

import cv2
from tqdm import tqdm


# =========================
# Configuration
# =========================

VIDEO_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras/SEB_181362.mp4")
# VIDEO_PATH = VIDEO_PATH.parent / "Savanne2016-02-27_10-51-05.mp4"
OUTPUT_PATH = VIDEO_PATH.parent / f"{VIDEO_PATH.stem}_2fps.mp4"

TARGET_FPS = 2.0


# =========================
# Video conversion
# =========================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if original_fps <= 0:
    cap.release()
    raise ValueError(f"Invalid FPS reported for: {VIDEO_PATH}")

if TARGET_FPS > original_fps:
    cap.release()
    raise ValueError(
        f"Target FPS ({TARGET_FPS}) exceeds original FPS ({original_fps:.2f})."
    )

# Keep one frame every N source frames.
frame_step = max(1, round(original_fps / TARGET_FPS))
output_fps = original_fps / frame_step

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    output_fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Cannot create output video: {OUTPUT_PATH}")

frame_idx = 0
saved_frames = 0

with tqdm(total=total_frames, desc="Converting video") as pbar:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_step == 0:
            writer.write(frame)
            saved_frames += 1

        frame_idx += 1
        pbar.update(1)

cap.release()
writer.release()

print(f"Input video:  {VIDEO_PATH}")
print(f"Output video: {OUTPUT_PATH}")
print(f"Original FPS: {original_fps:.3f}")
print(f"Output FPS:   {output_fps:.3f}")
print(f"Frames read:  {frame_idx}")
print(f"Frames saved: {saved_frames}")
print(f"Duration:     {saved_frames / output_fps:.2f} seconds")
