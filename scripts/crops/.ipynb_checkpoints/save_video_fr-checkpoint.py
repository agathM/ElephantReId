"""
Summary:
Extract selected frames from a video and save them as PNG images.
Output filenames follow the format video_name_fr_0001.png.
"""

from pathlib import Path

import cv2


def extract_frames(video_data, video_dir, output_dir):
    """
    Extract selected frames from a video.

    Args:
        video_data: Tuple (video_name, frame_indices), where frame_indices
                    is a list of zero-based frame indices.
        video_dir: Directory containing the input video.
        output_dir: Directory where extracted frames will be saved.
    """
    video_name = video_data

    video_path = Path(video_dir) / video_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_stem = Path(video_name).stem
    frame_indices = [0,total_frames//4,total_frames//2,3*(total_frames//4)]
    try:
        for frame_idx in sorted(set(frame_indices)):
            if not 0 <= frame_idx < total_frames:
                print(
                    f"Warning: frame {frame_idx} is out of range "
                    f"for {video_name} ({total_frames} frames)."
                )
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()

            if not success:
                print(f"Warning: cannot read frame {frame_idx} from {video_name}.")
                continue

            output_path = (
                output_dir / f"{video_stem}_fr_{frame_idx:04d}.png"
            )

            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Cannot save frame: {output_path}")

            print(f"Saved: {output_path}")

    finally:
        cap.release()


# ============================================================
# Example
# ============================================================

VIDEO_DIR = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras"
)

OUTPUT_DIR = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/raw/frames"
)

videos = [
    "SEB_181362.mp4",
"SEB_12026.mp4", "SEB_20946.mp4", "SEB_62049.mp4", "SEB_126507.mp4", "SEB_146015.mp4","SEB_181362.mp4", "SEB_180363.mp4"]

for v in videos:
    extract_frames(v, VIDEO_DIR, OUTPUT_DIR)