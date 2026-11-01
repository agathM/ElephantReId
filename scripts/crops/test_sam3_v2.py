
"""
Summary:
Convert videos to a target FPS and segment selected objects with SAM 3
on one specified frame per video, using a list of text prompts.
Save the converted videos, binary masks, and annotated reference frames.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from sam3.model_builder import build_sam3_video_predictor


# ============================================================
# Configuration
# ============================================================

VIDEO_DIR = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras"
)

OUTPUT_DIR = Path(
    "/pbs/home/a/ameurisse/ElephantReId/test"
)

# Each tuple contains:
# (video filename, reference frame index in the ORIGINAL video)
VIDEOS_TO_PROCESS = [
    ("SEB_181362.mp4", 0),
    ("SEB_12026.mp4", 0),
    ("SEB_20946.mp4",0), 
    ("SEB_62049.mp4",0),
    ("SEB_126507.mp4",0),
    ("SEB_146015.mp4",0),
    ("SEB_181362.mp4",0),
    ("SEB_180363.mp4",0)
]

# Target FPS of the output videos.
TARGET_FPS = 2.0

# Objects to segment on the reference frame.
PROMPTS = [
    "elephant head",
    "elephant ears",
]

# Visualisation settings.
OVERLAY_ALPHA = 0.45
MASK_THRESHOLD = 0.5

# Save individual binary masks as PNG files.
SAVE_MASKS = True

# ============================================================
# Helper functions
# ============================================================


def convert_video_to_target_fps(video_path, output_path, target_fps):
    """
    Convert a video to a lower FPS by decoding frames sequentially.

    Returns:
        original_fps, output_fps, original_frame_count,
        retained_original_indices
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if original_fps <= 0 or target_fps <= 0:
        cap.release()
        raise ValueError("FPS values must be positive.")

    if target_fps > original_fps:
        cap.release()
        raise ValueError(
            f"Target FPS ({target_fps}) exceeds original FPS "
            f"({original_fps:.2f})."
        )

    # Select frames at regular intervals.
    frame_step = max(1, round(original_fps / target_fps))
    output_fps = original_fps / frame_step

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video: {output_path}")

    retained_indices = []
    frame_idx = 0

    try:
        with tqdm(
            total=total_frames,
            desc=f"Converting {video_path.name}",
        ) as pbar:
            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_idx % frame_step == 0:
                    writer.write(frame)
                    retained_indices.append(frame_idx)

                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()
        writer.release()

    print(f"  Original FPS: {original_fps:.3f}")
    print(f"  Output FPS:   {output_fps:.3f}")
    print(f"  Frames read:  {frame_idx}")
    print(f"  Frames kept:  {len(retained_indices)}")

    return (
        original_fps,
        output_fps,
        frame_idx,
        retained_indices,
    )


def extract_frame(video_path, frame_index):
    """Read one frame from a video without loading the whole video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise ValueError(
            f"Cannot read frame {frame_index} from {video_path}"
        )

    return frame


def save_mask(mask, path):
    """Convert a SAM mask to a binary uint8 PNG."""
    mask = np.asarray(mask).squeeze()

    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    binary = (mask > MASK_THRESHOLD).astype(np.uint8) * 255
    cv2.imwrite(str(path), binary)


def overlay_mask(frame, mask, alpha=OVERLAY_ALPHA):
    """Overlay a binary mask in green on a copy of the frame."""
    result = frame.copy()
    foreground = mask > 0

    green = np.array([0, 255, 0], dtype=np.float32)

    result[foreground] = (
        result[foreground].astype(np.float32) * (1 - alpha)
        + green * alpha
    ).astype(np.uint8)

    return result


# ============================================================
# Main processing
# ============================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    converted_dir = OUTPUT_DIR / "converted"
    masks_dir = OUTPUT_DIR / "masks"
    overlays_dir = OUTPUT_DIR / "overlays"

    converted_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Initialise SAM 3 once and reuse it across videos.
    predictor = build_sam3_video_predictor()

    for video_name, first_frame in VIDEOS_TO_PROCESS:
        video_path = VIDEO_DIR / video_name
        video_stem = Path(video_name).stem

        converted_path = (
            converted_dir / f"{video_stem}_{TARGET_FPS:g}fps.mp4"
        )

        print(f"\n{'=' * 60}")
        print(f"Video: {video_name}")
        print(f"Reference frame: {first_frame}")

        # Step 1: Convert the video to the target FPS.
        (
            original_fps,
            output_fps,
            original_frame_count,
            retained_indices,
        ) = convert_video_to_target_fps(
            video_path,
            converted_path,
            TARGET_FPS,
        )

        if not 0 <= first_frame < original_frame_count:
            raise ValueError(
                f"Reference frame {first_frame} is outside "
                f"the video ({original_frame_count} frames)."
            )

        # Step 2: Find the corresponding frame in the converted video.
        # If it was not retained, extract it from the original video
        # and insert it into the converted video's frame sequence.
        frame_step = max(1, round(original_fps / TARGET_FPS))

        converted_frame_index = int(
            np.argmin(np.abs(np.asarray(retained_indices) - first_frame))
        )

        selected_original_index = retained_indices[
            converted_frame_index
        ]

        if selected_original_index != first_frame:
            print(
                f"  Frame {first_frame} was not retained during "
                "conversion. The closest retained frame will be used "
                f"(original frame {selected_original_index})."
            )

        # Use the actual retained frame for consistent SAM input/output.
        reference_frame = extract_frame(
            converted_path,
            converted_frame_index,
        )

        height, width = reference_frame.shape[:2]

        # Step 3: Start a SAM 3 session on the converted video.
        session_id = None

        try:
            response = predictor.handle_request(
                request={
                    "type": "start_session",
                    "resource_path": str(converted_path.resolve()),
                }
            )

            session_id = response["session_id"]

            combined_masks = []

            # Step 4: Segment each prompt on the reference frame only.
            for prompt_idx, prompt in enumerate(PROMPTS):
                print(f"  Segmenting prompt: {prompt}")

                predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": converted_frame_index,
                        "text": prompt,
                    }
                )

                # Request only the output for the selected frame.
                # No propagate_in_video request is made.
                response = predictor.handle_request(
                    request={
                        "type": "get_frame",
                        "session_id": session_id,
                        "frame_index": converted_frame_index,
                    }
                )

                outputs = response.get("outputs", response)

                masks = outputs.get("out_binary_masks")

                if masks is None:
                    print(f"  No mask returned for prompt: {prompt}")
                    continue

                masks = np.asarray(masks)

                if masks.ndim == 2:
                    masks = masks[None, ...]

                prompt_mask = np.zeros(
                    (height, width),
                    dtype=np.uint8,
                )

                for mask_idx, mask in enumerate(masks):
                    mask = np.squeeze(mask)

                    if mask.ndim != 2:
                        continue

                    # SAM may return masks at a lower resolution.
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    binary = (
                        mask > MASK_THRESHOLD
                    ).astype(np.uint8) * 255

                    prompt_mask[binary > 0] = 255

                combined_masks.append((prompt, prompt_mask))

                if SAVE_MASKS:
                    safe_prompt = (
                        prompt.lower()
                        .replace(" ", "_")
                        .replace("/", "_")
                    )

                    mask_path = (
                        masks_dir
                        / f"{video_stem}_frame{first_frame:06d}_"
                        f"{safe_prompt}.png"
                    )

                    save_mask(prompt_mask, mask_path)

            # Step 5: Save an overlay of the reference frame.
            overlay = reference_frame.copy()

            for prompt, mask in combined_masks:
                overlay = overlay_mask(
                    overlay,
                    mask,
                    alpha=OVERLAY_ALPHA,
                )

            overlay_path = (
                overlays_dir
                / f"{video_stem}_frame{first_frame:06d}.jpg"
            )

            cv2.imwrite(str(overlay_path), overlay)

            # Step 6: Create a converted video with the overlay
            # applied only to the selected reference frame.
            output_video_path = (
                OUTPUT_DIR
                / f"{video_stem}_{TARGET_FPS:g}fps_segmented.mp4"
            )

            cap = cv2.VideoCapture(str(converted_path))

            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (width, height),
            )

            if not writer.isOpened():
                cap.release()
                raise RuntimeError(
                    f"Cannot create output video: {output_video_path}"
                )

            try:
                frame_idx = 0

                while True:
                    ok, frame = cap.read()

                    if not ok:
                        break

                    if frame_idx == converted_frame_index:
                        frame = overlay

                    writer.write(frame)
                    frame_idx += 1
            finally:
                cap.release()
                writer.release()

            print(f"  Converted video: {converted_path}")
            print(f"  Overlay image:   {overlay_path}")
            print(f"  Segmented video: {output_video_path}")

        finally:
            if session_id is not None:
                predictor.handle_request(
                    request={
                        "type": "close_session",
                        "session_id": session_id,
                    }
                )

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()
