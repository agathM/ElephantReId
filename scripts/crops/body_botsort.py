# Summary:
# Run MegaDetector on videos ordered by camera and creation time,
# while keeping BoT-SORT persistent across successive videos
# separated by less than one minute.
# If successive videos keep the same track ID, this suggests the
# same trajectory across videos, but different track IDs do not
# necessarily mean different individuals.

import subprocess
from pathlib import Path

import cv2
import pandas as pd
import supervision as sv
import torch
from tqdm import tqdm
from PytorchWildlife.models import detection as pw_detection
from trackers import BoTSORTTracker


# ============================================================
# PARAMETERS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.cuda.set_device(0)

print(f"Device: {DEVICE}")

VIDEO_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras"
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".MP4", ".AVI"}

TARGET_FPS = 2

# Maximum gap between two videos for the tracker to be kept
MAX_VIDEO_GAP_SECONDS = 60

CSV_PATH = (
    Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops")
    / f"all_body_botsort_{MAX_VIDEO_GAP_SECONDS}.csv"
)

test_videos = [
    "SEB_126056",
    "SEB_126057",
    "SEB_126058",
    "SEB_126059",
    "SEB_126060",
    "SEB_126061",
]


# ============================================================
# MEGADETECTOR
# ============================================================

detection_model = pw_detection.MegaDetectorV6(
    device=DEVICE,
    pretrained=True,
    version="MDV6-yolov10-e",
)


# ============================================================
# VIDEO METADATA
# ============================================================

def get_creation_time(src: Path):
    """Return the video creation time as a timezone-aware UTC datetime."""

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format_tags=creation_time",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(src),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    creation_time = result.stdout.strip()

    if not creation_time:
        return None

    dt = pd.to_datetime(
        creation_time,
        utc=True,
        errors="coerce",
    )

    if pd.isna(dt):
        print(
            f"Warning: could not parse creation_time "
            f"for {src}: {creation_time}"
        )
        return None

    return dt


def get_camera(src: Path, video_root: Path):
    """
    Return the camera name from the directory structure.

    Expected structure:
        VIDEO_PATH / camera / individual / mode / video
    """

    relative_path = src.relative_to(video_root)

    return relative_path.parts[0]


def get_video_metadata(video_root: Path):
    """Find all videos and retrieve their camera and creation time."""

    rows = []

    for src in video_root.rglob("*"):

        if src.suffix not in VIDEO_EXTENSIONS:
            continue

        creation_time = get_creation_time(src)

        if creation_time is not None and src.suffix == ".mp4":
            print("Creation time for:", src)

        elif creation_time is None and src.suffix != ".mp4":
            print("No Creation time for:", src)

        camera = get_camera(src, video_root)

        rows.append(
            {
                "path": src,
                "video": src.stem,
                "camera": camera,
                "creation_time": creation_time,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(
        by=["camera", "creation_time"],
        na_position="last",
    ).reset_index(drop=True)

    return df


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    source_path: Path,
    video_name: str,
    detection_model,
    tracker,
    target_fps: int = 2,
):
    """
    Process one video using the provided persistent BoT-SORT tracker.

    The tracker is passed from the caller so that it can be kept
    between successive videos.
    """

    cap = cv2.VideoCapture(str(source_path))

    if not cap.isOpened():
        print(f"Cannot open {source_path}")
        return pd.DataFrame()

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        print(f"Invalid FPS for {source_path}")
        cap.release()
        return pd.DataFrame()

    stride = max(
        1,
        round(source_fps / target_fps),
    )

    frame_id = 0

    detections_rows = []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    pbar = tqdm(
        total=total_frames,
        desc=video_name,
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % stride == 0:

            # ------------------------------------------------
            # MegaDetector
            # ------------------------------------------------

            results_det = detection_model.single_image_detection(
                frame,
                img_path=frame_id,
            )

            if (
                results_det is not None
                and len(results_det["detections"].confidence) > 0
            ):

                det = results_det["detections"]

                detections = sv.Detections(
                    xyxy=det.xyxy,
                    confidence=det.confidence,
                    class_id=det.class_id,
                )

                # ------------------------------------------------
                # BoT-SORT
                # ------------------------------------------------

                tracked = tracker.update(
                    detections,
                    frame=frame,
                )

                if tracked.tracker_id is not None:

                    for bbox, conf, track_id in zip(
                        tracked.xyxy,
                        tracked.confidence,
                        tracked.tracker_id,
                    ):

                        track_id = int(track_id)
                        conf = float(conf)

                        detections_rows.append(
                            {
                                "video": video_name,
                                "frame": frame_id,
                                "track": track_id,
                                "label": "body",
                                "conf": round(conf, 2),
                                "x1": int(round(bbox[0])),
                                "y1": int(round(bbox[1])),
                                "x2": int(round(bbox[2])),
                                "y2": int(round(bbox[3])),
                            }
                        )

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if not detections_rows:
        return pd.DataFrame()

    return pd.DataFrame(detections_rows)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Retrieve and sort video metadata
    # --------------------------------------------------------

    videos_df = get_video_metadata(VIDEO_PATH)

    videos_df = videos_df[
        videos_df.video.isin(test_videos)
    ]

    if videos_df.empty:
        print("No videos found.")
        return

    print(
        f"Found {len(videos_df)} videos "
        f"from {videos_df['camera'].nunique()} cameras."
    )

    # --------------------------------------------------------
    # Process each camera independently
    # --------------------------------------------------------

    all_dfs = []

    for camera, camera_videos in videos_df.groupby(
        "camera",
        sort=False,
    ):

        print(
            f"\n{'=' * 60}\n"
            f"Camera: {camera}\n"
            f"{'=' * 60}"
        )

        # New BoT-SORT tracker for each camera.
        # Track IDs cannot therefore be shared between cameras.
        tracker = BoTSORTTracker()

        previous_creation_time = None

        for _, video_info in camera_videos.iterrows():

            src = video_info["path"]
            video_name = video_info["video"]
            creation_time = video_info["creation_time"]

            # ------------------------------------------------
            # Decide whether the tracker should be reset
            # ------------------------------------------------

            reset_tracker = False

            if (
                previous_creation_time is not None
                and creation_time is not None
            ):

                gap_seconds = (
                    creation_time - previous_creation_time
                ).total_seconds()

                if gap_seconds >= MAX_VIDEO_GAP_SECONDS:

                    reset_tracker = True

                    print(
                        f"\nGap of {gap_seconds:.1f}s before "
                        f"{video_name} -> resetting BoT-SORT"
                    )

                else:

                    print(
                        f"\nGap of {gap_seconds:.1f}s before "
                        f"{video_name} -> keeping BoT-SORT"
                    )

            elif previous_creation_time is None:

                print(
                    f"\nFirst video for camera {camera} "
                    f"-> initializing BoT-SORT"
                )

            else:

                print(
                    f"\nNo creation time for {video_name} "
                    f"-> keeping current BoT-SORT"
                )

            if reset_tracker:
                tracker = BoTSORTTracker()

            # ------------------------------------------------
            # Process video
            # ------------------------------------------------

            df = process_video(
                source_path=src,
                video_name=video_name,
                detection_model=detection_model,
                tracker=tracker,
                target_fps=TARGET_FPS,
            )

            if not df.empty:
                all_dfs.append(df)

            # ------------------------------------------------
            # Update previous video timestamp
            # ------------------------------------------------

            if creation_time is not None:
                previous_creation_time = creation_time

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not all_dfs:
        print("No detections found.")
        return

    all_df = pd.concat(
        all_dfs,
        ignore_index=True,
    )

    all_df.to_csv(
        CSV_PATH,
        index=False,
    )

    print(
        f"\nSaved {len(all_df)} detections to {CSV_PATH}"
    )

    print(
        f"Number of videos: "
        f"{all_df['video'].nunique()}"
    )

    print(
        f"Number of tracks: "
        f"{all_df['track'].nunique()}"
    )


if __name__ == "__main__":
    main()