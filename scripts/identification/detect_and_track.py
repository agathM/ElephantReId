"""
YOLO video tracking and annotation pipeline.

This script:
- Loads a YOLO detection model from a dict.
- Runs object tracking on all videos contained in the input directory that have not been treated.
- Uses ByteTrack or BoT-SORT to assign track IDs to detections.
- Add indiv name by joining to a segmented ear bounding box from the frame_database
- Exports all detections to a CSV file containing:
    video, frame, track_id, label, confidence, x1, y1, x2, y2, indiv
- First check that detections without ids can be match with the frame_database or the video_database, if the images were not saved the video will be treated otherwise detection frame_database is updated
"""
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv
import warnings
import pickle
import os
import torch
from datetime import datetime
import pdb

DEVICE = 0 if torch.cuda.is_available() else "cpu"

VIDEO_EXTENSIONS = {
    ".mp4",".AVI",".MP4"
}

BORDER = 1

REPO_ROOT = Path.cwd()

CSV_COLUMNS = [
    "image",
    "video",
    "frame",
    "track_id",
    "detection",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "area",
    "ratio",
    "indiv"
]

CROPS_DTYPES = {
    "video": str,
    "image": str,
    "track_id": int,
    "detection": str,
    "area": int,
    "ratio": float,
    "indiv": str,
    "frame": int,
    "conf": float,
    "x1": int,
    "x2": int,
    "y1": int,
    "y2": int,
}


# ============================================================
# Utility functions
# ============================================================
def get_detection_id(row):
    video = row['video']
    frame = row['frame']
    if video in single_indiv_videos:
        return single_indiv_videos[video]
    rows = frame_database_df[(frame_database_df.video==video) & (frame_database_df.frame==frame)]
    def intersection_rate(box1: dict, box2: dict) -> float:
        """
        Compute the intersection area between two bounding boxes.

        Each box must contain the keys: x1, x2, y1, y2.
        """
        def area(box: dict) -> float:
            return abs(box["x1"] - box["x2"]) * abs(box["y1"] - box["y2"])
        x_left = max(box1["x1"], box2["x1"])
        y_top = max(box1["y1"], box2["y1"])
        x_right = min(box1["x2"], box2["x2"])
        y_bottom = min(box1["y2"], box2["y2"])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        return (x_right - x_left) * (y_bottom - y_top) / area(box1)
    best_id, max_rate = None, 0
    for _,dt_row in rows.iterrows():
        rate = intersection_rate(dt_row, row)
        if rate>identification_threshold and rate>max_rate:
            if best_id is not None:
                warnings.warn("Duplicate detection detected.")
                print(best_id,dt_row["indiv"],row.video,row.frame)
            best_id = dt_row["indiv"]
            max_rate = rate
    return best_id

def get_partially_treated_video_to_treat(partially_treated_videos):
    """Actualize detection database with indiv from frame database
    If there exists a detection without image being saved, the video has to be treated again
    Returns videos already treated that needs to be treated again"""

    if not os.path.exists(crops_file_path):
        return partially_treated_videos
    videos_to_treat = set()
    detections = pd.read_csv(crops_file_path)
    mask = (
        detections["video"].isin(partially_treated_videos)
        & detections["indiv"].isna()
    )

    for idx, row in detections[mask].iterrows():
        detection_id = get_detection_id(row)
        label = row["detection"]
        image = row["image"]

        if detection_id is not None:
            path = crops_dir / "unidentified" / label.replace(" ", "_") / image

            if os.path.exists(path):
                os.rename(path, crops_dir / label.replace(" ", "_") / image)
                detections.loc[idx, "indiv"] = detection_id
            else:
                videos_to_treat.add(row["video"])
    detections = detections[~detections.video.isin(videos_to_treat)] #avoid duplicates
    detections.drop_duplicates(inplace=True) #safeguard
    detections.to_csv(crops_file_path,index=False)
    return videos_to_treat

def get_stride(video_path, target_fps):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    cap.release()

    if fps == 0:
        return 1

    return max(
        1,
        round(fps / target_fps)
    )


def append_rows(rows):
    """Ajoute les lignes d'une video terminée a la fin du CSV.
    """
    if not rows:
        return

    crops_file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    df = pd.DataFrame(
        rows,
        columns=CSV_COLUMNS,
    )
    if os.path.exists(crops_file_path):
        df_all = pd.read_csv(crops_file_path)
        df = pd.concat([df, df_all])
    df.to_csv(crops_file_path,index=False)


def tracked_video_exists(tracked_videos_dir, video_path):
    """Return True if a video with the same stem already exists in tracked_videos_dir."""
    if tracked_videos_dir is None or not Path(tracked_videos_dir).exists():
        return False

    stem = Path(video_path).stem
    return any(
        path.is_file() and path.stem == stem
        for path in Path(tracked_videos_dir).iterdir()
    )


def _track_color(track_id):
    """Deterministic BGR color for a given track id."""
    return (
        37 * track_id % 256,
        17 * track_id % 256,
        29 * track_id % 256,
    )


def draw_tracked_detections(frame, detections):
    """Draw bounding boxes, track ids and individual ids on a frame."""
    annotated = frame.copy()

    for detection in detections:
        bbox = detection.get("bbox")
        if bbox is None:
            continue

        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        track_id = int(detection.get("track_id", -1))
        color = _track_color(track_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{detection.get('label', '')} id:{track_id}"
        if detection.get("indiv") is not None:
            text += f" indiv:{detection['indiv']}"

        (text_w, text_h), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )
        text_top = max(0, y1 - text_h - 5)
        text_bottom = max(text_h + 5, y1)
        cv2.rectangle(
            annotated,
            (x1, text_top),
            (x1 + text_w + 5, text_bottom),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 2, text_bottom - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return annotated


def save_tracked_video(video_path, detections_by_frame, output_path):
    """Write an annotated copy of a video.

    Detections are stored at inference frames only. The last known detections
    are reused for frames skipped by the detection stride, so boxes remain
    visible in the output video.
    """
    if not detections_by_frame:
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"Cannot open video to save tracked annotations: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        warnings.warn(f"Cannot create tracked video: {output_path}")
        return False

    detection_frames = sorted(detections_by_frame)
    detection_index = 0
    current_detections = []
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            while (
                detection_index < len(detection_frames)
                and detection_frames[detection_index] <= frame_number
            ):
                current_detections = detections_by_frame[
                    detection_frames[detection_index]
                ]
                detection_index += 1

            writer.write(draw_tracked_detections(frame, current_detections))
            frame_number += 1
    finally:
        cap.release()
        writer.release()

    return True


# ============================================================
# Video processing
# ============================================================


def process_video_seg(
        model,
        video_path,
        csv_rows,
        conf_thresh,
        tracker,
        treated_videos,
        keep_detections_without_id,
        crops_dir,
        target_fps,
        reset_tracker,
        detections_by_frame=None
):
    """
    Process one video.

    YOLO tracking is performed at a reduced FPS for efficiency.

    All detections are stored in csv_rows, which is filled only for this video:
    the caller is responsible for writing them once the video is done.
    detections_by_frame, when provided, is filled with the detections used to
    produce the optional annotated video.
    """

    def get_mask_bbox(mask, image_width, image_height, border=BORDER):
        """Bounding box du masque, bornes incluses comme dans ear_segment_detect."""
        ys, xs = np.where(mask)

        if len(xs) == 0:
            return None

        x1 = max(0, int(xs.min()) - border)
        y1 = max(0, int(ys.min()) - border)
        x2 = min(image_width - 1, int(xs.max()) + border)
        y2 = min(image_height - 1, int(ys.max()) + border)

        return x1, y1, x2, y2

    def crop_masked_image(image, mask, border=BORDER):
        """Crop limité au masque, fond noir hors masque."""
        bbox = get_mask_bbox(
            mask,
            image.shape[1],
            image.shape[0],
            border=border,
        )

        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox

        result = np.zeros_like(image)
        result[mask] = image[mask]

        return result[y1:y2 + 1, x1:x2 + 1]
    if reset_tracker:
        model.predictor = None

    stride = get_stride(
        video_path,
        target_fps
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    frame_number = 0

    try:
        while True:

            ret, frame = cap.read()


            if not ret:
                break



            # Run YOLO only every stride frames.
            if frame_number % stride == 0:

                results = model.track(
                    frame,
                    conf=conf_thresh,
                    tracker=tracker,
                    persist=True,
                    verbose=False,
                    device=DEVICE
                )



                result = results[0]

                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                else:
                    masks = (
                        [None] * len(result.boxes)
                        if result.boxes is not None
                        else []
                    )

                frame_h, frame_w = frame.shape[:2]

                last_detections = []



                if result.boxes is not None:


                    boxes = result.boxes



                    for i in range(
                        len(boxes)
                    ):


                        cls = int(
                            boxes.cls[i]
                            .cpu()
                            .item()
                        )


                        conf = float(
                            boxes.conf[i]
                            .cpu()
                            .item()
                        )


                        bbox = (
                            boxes.xyxy[i]
                            .cpu()
                            .numpy()
                            .tolist()
                        )


                        if boxes.id is not None:

                            track_id = int(
                                boxes.id[i]
                                .cpu()
                                .item()
                            )

                        else:

                            continue #if box not link to id ignore



                        label = model.names[cls]

                        mask = None

                        if i < len(masks) and masks[i] is not None:
                            mask_up = cv2.resize(
                                masks[i].astype(np.float32),
                                (frame_w, frame_h),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            mask = mask_up > 0.5

                        last_detections.append(
                            {
                                "bbox": bbox,
                                "cls": cls,
                                "detection": label,
                                "conf": conf,
                                "track_id": track_id,
                                "indiv": None,
                            }
                        )



                        if mask is None:
                            continue

                        bbox_mask = get_mask_bbox(
                            mask,
                            frame_w,
                            frame_h,
                            border=BORDER,
                        )

                        if bbox_mask is None:
                            continue

                        x1 = int(round(bbox_mask[0]))
                        y1 = int(round(bbox_mask[1]))
                        x2 = int(round(bbox_mask[2]))
                        y2 = int(round(bbox_mask[3]))

                        image = (
                            f"{video_path.stem}"
                            f"_fr_"
                            f"{int(frame_number):06d}"
                            f"_t{track_id}.png"
                        )
                        row =     {
                                "image": image,
                                "video":
                                    video_path.name,

                                "frame":
                                    int(frame_number),

                                "track_id":
                                    track_id,

                                "detection":
                                    label.replace(" ","_"),

                                "conf":
                                    round(conf,2),

                                "x1":
                                    x1,

                                "y1":
                                    y1,

                                "x2":
                                    x2,

                                "y2":
                                    y2,

                                "area":
                                    int(np.sum(mask)),

                                "ratio":
                                    round((x2 - x1) / max(y2 - y1, 1),1)
                            }
                        indiv = get_detection_id(row)

                        crop = crop_masked_image(
                            frame,
                            mask,
                            border=BORDER,
                        )

                        if indiv is None:
                            treated_videos['some_detect'].add(video_path.name)
                            if keep_detections_without_id and crop is not None:
                                cv2.imwrite(
                                    crops_dir
                                    / "unidentified"
                                    / label.replace(" ", "_")
                                    / image,
                                    crop,
                                )
                        elif crop is not None:
                            cv2.imwrite(
                                crops_dir / label.replace(" ", "_") / image,
                                crop,
                            )
                            # new detection with id implies that new split will have to be created
                            if os.path.exists(split_dir):
                                shutil.rmtree(split_dir)
                                print("split_dir rmtree",flush=True)
                            else:
                                print("split_dir not found",flush=True)

                        row['indiv'] = indiv
                        last_detections[-1]['indiv'] = indiv
                        csv_rows.append(row)

                if detections_by_frame is not None:
                    detections_by_frame[frame_number] = last_detections

            frame_number += 1


        if video_path.name not in treated_videos['some_detect']:
            treated_videos['all_detect'].add(video_path.name)
    finally:
        cap.release()

def process_video_crop(
        model,
        video_path,
        csv_rows,
        conf_thresh,
        tracker,
        treated_videos,
        keep_detections_without_id,
        crops_dir,
        target_fps,
        reset_tracker,
        detections_by_frame=None
):
    """
    Process one video.

    YOLO tracking is performed at a reduced FPS for efficiency.

    All detections are stored in csv_rows, which is filled only for this video:
    the caller is responsible for writing them once the video is done.
    detections_by_frame, when provided, is filled with the detections used to
    produce the optional annotated video.
    """
    if reset_tracker:
        model.predictor = None

    stride = get_stride(
        video_path,
        target_fps
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    frame_number = 0
    # # Keep last YOLO detections between inference frames
    # last_detections = []

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Run YOLO only every stride frames.
            if frame_number % stride == 0:

                results = model.track(
                    frame,
                    conf=conf_thresh,
                    tracker=tracker,
                    persist=True,
                    verbose=False,
                    device=DEVICE
                )

                result = results[0]

                last_detections = []

                if result.boxes is not None:

                    boxes = result.boxes

                    for i in range(
                            len(boxes)
                    ):

                        cls = int(
                            boxes.cls[i]
                            .cpu()
                            .item()
                        )

                        conf = float(
                            boxes.conf[i]
                            .cpu()
                            .item()
                        )

                        bbox = (
                            boxes.xyxy[i]
                            .cpu()
                            .numpy()
                            .tolist()
                        )

                        if boxes.id is not None:

                            track_id = int(
                                boxes.id[i]
                                .cpu()
                                .item()
                            )

                        else:

                            continue  # if box not link to id ignore

                        label = model.names[cls]
                        if label not in labels:
                            continue

                        last_detections.append(
                            {
                                "bbox": bbox,
                                "cls": cls,
                                "detection": label,
                                "conf": conf,
                                "track_id": track_id,
                                "indiv": None,
                            }
                        )

                        x1, y1, x2, y2 = int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(
                            round(bbox[3]))
                        image = (
                            f"{video_path.stem}"
                            f"_fr_"
                            f"{int(frame_number):06d}"
                            f"_t{track_id}.png"
                        )
                        row = {
                            "image": image,
                            "video":
                                video_path.name,

                            "frame":
                                int(frame_number),

                            "track_id":
                                track_id,

                            "detection":
                                label.replace(" ", "_"),

                            "conf":
                                round(conf,2),

                            "x1":
                                x1,

                            "y1":
                                y1,

                            "x2":
                                x2,

                            "y2":
                                y2,

                            "area":
                                round(x2 - x1) * round(y2 - y1),

                            "ratio":
                                round((x2 - x1) / max(y2 - y1, 1),1)
                        }
                        indiv = get_detection_id(row)

                        if indiv is None:
                            treated_videos['some_detect'].add(video_path.name)
                            if keep_detections_without_id:
                                cv2.imwrite(crops_dir / "unidentified" / label.replace(" ", "_") / image,
                                            frame[y1:y2, x1:x2])
                        else:
                            cv2.imwrite(crops_dir / label.replace(" ", "_") / image, frame[y1:y2, x1:x2])
                            # new detection with id implies that new split will have to be created
                            if os.path.exists(split_dir):
                                shutil.rmtree(split_dir)
                                print("split_dir rmtree")
                            else:
                                print("split_dir not found")

                        row['indiv'] = indiv
                        last_detections[-1]['indiv'] = indiv
                        csv_rows.append(row)

                if detections_by_frame is not None:
                    detections_by_frame[frame_number] = last_detections

            frame_number += 1

        if video_path.name not in treated_videos['some_detect']:
            treated_videos['all_detect'].add(video_path.name)
    finally:
        cap.release()

def process_video_megadetector(
        model,
        video_path,
        csv_rows,
        conf_thresh,
        tracker,
        treated_videos,
        keep_detections_without_id,
        crops_dir,
        target_fps,
        reset_tracker,
        detections_by_frame=None
):
    """
    Process one video with MegaDetector and an external tracker.

    MegaDetector does not integrate tracking: detections are computed first,
    then passed to the external tracker (e.g. supervision ByteTrack).
    """
    if reset_tracker and hasattr(tracker, "reset"):
        tracker.reset()

    stride = get_stride(
        video_path,
        target_fps
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    frame_number = 0

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Run MegaDetector only every stride frames.
            if frame_number % stride == 0:

                last_detections = []

                results_det = model.single_image_detection(
                    frame,
                    img_path=frame_number,
                )

                if (
                    results_det is not None
                    and results_det.get("detections") is not None
                ):

                    det = results_det["detections"]
                    keep = det.confidence >= conf_thresh

                    if np.any(keep):

                        class_id = getattr(det, "class_id", None)
                        detections = sv.Detections(
                            xyxy=det.xyxy[keep],
                            confidence=det.confidence[keep],
                            class_id=(
                                class_id[keep]
                                if class_id is not None
                                else None
                            ),
                        )

                        # The tracker is external to MegaDetector.
                        tracked = tracker.update_with_detections(
                            detections
                        )

                        if tracked.tracker_id is not None:

                            for bbox, conf, track_id in zip(
                                tracked.xyxy,
                                tracked.confidence,
                                tracked.tracker_id,
                            ):

                                track_id = int(track_id)

                                if track_id < 0:
                                    continue

                                conf = float(conf)
                                bbox = [float(value) for value in bbox]

                                x1 = int(round(bbox[0]))
                                y1 = int(round(bbox[1]))
                                x2 = int(round(bbox[2]))
                                y2 = int(round(bbox[3]))

                                label = "body_detect"

                                last_detections.append(
                                    {
                                        "bbox": bbox,
                                        "detection": label,
                                        "conf": conf,
                                        "track_id": track_id,
                                        "indiv": None,
                                    }
                                )

                                image = (
                                    f"{video_path.stem}"
                                    f"_fr_"
                                    f"{int(frame_number):06d}"
                                    f"_t{track_id}.png"
                                )
                                row = {
                                    "image": image,
                                    "video":
                                        video_path.name,

                                    "frame":
                                        int(frame_number),

                                    "track_id":
                                        track_id,

                                    "detection":
                                        label.replace(" ", "_"),

                                    "conf":
                                        round(conf,2),

                                    "x1":
                                        x1,

                                    "y1":
                                        y1,

                                    "x2":
                                        x2,

                                    "y2":
                                        y2,

                                    "area":
                                        round(x2 - x1) * round(y2 - y1),

                                    "ratio":
                                        round((x2 - x1) / max(y2 - y1, 1),1)
                                }
                                indiv = get_detection_id(row)

                                crop = frame[y1:y2, x1:x2]

                                if indiv is None:
                                    treated_videos['some_detect'].add(video_path.name)
                                    if keep_detections_without_id and crop.size:
                                        cv2.imwrite(
                                            crops_dir
                                            / "unidentified"
                                            / label.replace(" ", "_")
                                            / image,
                                            crop,
                                        )
                                elif crop.size:
                                    cv2.imwrite(
                                        crops_dir / label.replace(" ", "_") / image,
                                        crop,
                                    )
                                    # new detection with id implies that new split will have to be created
                                    if os.path.exists(split_dir):
                                        shutil.rmtree(split_dir)
                                        print("split_dir rmtree")
                                    else:
                                        print("split_dir not found")

                                row['indiv'] = indiv
                                last_detections[-1]['indiv'] = indiv
                                csv_rows.append(row)

                if detections_by_frame is not None:
                    detections_by_frame[frame_number] = last_detections

            frame_number += 1

        if video_path.name not in treated_videos['some_detect']:
            treated_videos['all_detect'].add(video_path.name)
    finally:
        cap.release()


# ============================================================
# Main
# ============================================================


def main(metadata:dict):
    global frame_database_df, crops_dir, crops_file_path, identification_threshold, single_indiv_videos, \
        save_videos_with_detections, tracked_videos_dir, labels,split_dir, rerun
    home_dir = Path(metadata['home_dir'])
    split_dir = Path(metadata['split_dir'])
    print
    frame_database = home_dir / metadata['frame_database'] # REPO_ROOT / "database"
    frame_database_df = pd.read_csv(frame_database)
    frame_database_df.dropna(subset=["indiv"],inplace=True)
    model_name = metadata['model_name']
    crops_dir = home_dir / metadata['crops_dir'] / model_name
    crops_file_path = home_dir / metadata['crops_dir'] / ( model_name + ".csv")
    videos_dir = home_dir / metadata['videos_dir']
    keep_detections_without_id = metadata["keep_detections_without_id"]
    save_videos_with_detections = metadata["save_videos_with_detections"]
    labels_to_keep = metadata["labels_to_keep"]
    rerun = metadata["rerun"]
    test = metadata["test"]
    tracked_videos_dir = None
    if save_videos_with_detections:
        tracked_videos_rel = metadata['tracked_videos_dir']
        if tracked_videos_rel is None:
            raise KeyError(
                "save_videos_with_detections is True but neither "
                "'tracked_videos_dir' nor 'tracked_videos' is configured"
            )
        tracked_videos_dir = home_dir / tracked_videos_rel
        tracked_videos_dir.mkdir(parents=True, exist_ok=True)
    conf = metadata["conf"]
    target_fps = metadata["fps"]
    identification_threshold = metadata["identification_threshold"]
    single_indiv_videos_df = metadata['video_database']
    single_indiv_videos_df = single_indiv_videos_df[single_indiv_videos_df.several_indivs==False].filter(['indiv','video'])
    single_indiv_videos = {row['video']:row['indiv'] for _,row in single_indiv_videos_df.iterrows()}
    model_path = REPO_ROOT / "models" / (model_name + '.pt')
    #MODEL LOD+
    if  not os.path.exists(model_path):
        from PytorchWildlife.models import detection as pw_detection
        import supervision as sv
        detection_task = "megadetector"
        model = pw_detection.MegaDetectorV6(
            device="cuda" if torch.cuda.is_available() else "cpu",
            pretrained=True,
            version=model_name,
        )
        model.predictor.args.verbose = False
        tracker = metadata['tracker']
        if tracker is None or isinstance(tracker, str):
            tracker = sv.ByteTrack()
        labels = ["body_detect"]
        print("MegaDetector classes:", getattr(model, "CLASS_NAMES", labels))
    else:
        detection_task = "detect" if "detect" in model_name else "seg"
        model = YOLO(
            model_path
        )
        tracker = metadata['tracker'].lower() + '.yaml'
        labels = list(model.names.values())

        print(
            "Classes:",
            model.names
        )
    labels2 = []
    for i in range(len(labels)):
        label = labels[i]
        for label_to_keep in labels_to_keep:
            if label_to_keep in label:
                os.makedirs(crops_dir / label.replace(" ","_"), exist_ok=True)
                os.makedirs(crops_dir / label.replace(" ","_") / "unidentified", exist_ok=True)
                labels2.append(label)
    labels = labels2


    # Load camera/time maps
    video_to_camera, video_to_time = {}, {}
    for name, dst in [("video_to_camera.pkl", video_to_camera), ("video_to_time.pkl", video_to_time)]:
        pkl = home_dir / name
        if pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    dst.update(pickle.load(f))
            except Exception as e:
                print(f"[WARN] {pkl}: {e}")

    def _parse_dt(v):
        if v is None or isinstance(v, datetime):
            return v
        s = str(v).strip().replace("T", " ")
        fmt = "%Y/%m/%d %H:%M:%S"
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            try:
                return datetime.fromisoformat(s)
            except Exception:
                return None

    video_paths = set(
        p.name
        for p in videos_dir.glob("*")
        if p.is_file() and p.suffix in VIDEO_EXTENSIONS
    )

    treated_path = crops_dir / "treated_videos.pkl"
    if not treated_path.exists():
        treated_videos = {"all_detect": set(), "some_detect": set()}
        videos_to_treat = video_paths.copy()
    else:
        with open(treated_path, "rb") as f:
            treated_videos = pickle.load(f)
        if rerun:
            videos_to_treat = video_paths.copy() - treated_videos["all_detect"]
            videos_to_treat.update(get_partially_treated_video_to_treat(treated_videos['some_detect']))
        else:
            videos_to_treat = video_paths.copy() - treated_videos["all_detect"].union(treated_videos['some_detect'])

    print(
        f"{len(video_paths)} videos found, "
        f"{len(treated_videos['all_detect'].union(treated_videos['some_detect']))} treated, "
        f"{len(videos_to_treat)} to treat"
    )
    # Sort by camera then time
    items = []
    for p in videos_to_treat:
        stem = Path(p).stem
        cam = video_to_camera.get(stem) or video_to_camera.get(p)
        t = _parse_dt(video_to_time.get(stem) or video_to_time.get(p))
        items.append((p, cam, t))
    items.sort(key=lambda x: (x[1] if x[1] is not None else "~", x[2] if x[2] is not None else datetime.max))

    # Decide reset_tracker: keep tracker if same camera and < 5 min apart
    videos_with_reset = []
    prev_cam, prev_t = None, None
    for p, cam, t in items:
        reset_tracker = True
        if prev_cam and cam and prev_t and t and cam == prev_cam and abs((t - prev_t).total_seconds()) < 300:
            reset_tracker = False
        videos_with_reset.append((p, reset_tracker))
        prev_cam, prev_t = cam, t

    n_done = 0
    n_failed = []
    
    for p, reset_tracker in videos_with_reset:
        rows = []
        detections_by_frame = {}
        args = {"model": model,
                "video_path": videos_dir / p,
                "csv_rows": rows,
                "conf_thresh": conf,
                "tracker": tracker,
                "treated_videos": treated_videos,
                "keep_detections_without_id": keep_detections_without_id,
                "crops_dir": crops_dir,
                "target_fps": target_fps,
                "reset_tracker": reset_tracker,
                "detections_by_frame": detections_by_frame}
        try:
            process_fn = {
                "seg": process_video_seg,
                "detect": process_video_crop,
                "megadetector": process_video_megadetector,
            }[detection_task]
            try:
                process_fn(**args)
            except TypeError:
                args.pop("reset_tracker", None)
                process_fn(**args)

        except Exception as e:
            n_failed.append(p)
            print(
                f"[ECHEC] {p} : {type(e).__name__}: {e} "
                f"({len(rows)} detections non ecrites)"
            )
            continue

        append_rows(rows)

        # Save an annotated copy only when the video contains at least one
        # detection without an individual id and has not been saved yet.
        if (
            save_videos_with_detections
            and p in treated_videos['some_detect']
            and not tracked_video_exists(tracked_videos_dir, p)
        ):
            output_path = tracked_videos_dir / model_name / f"{Path(p).stem}.mp4"
            if save_tracked_video(
                videos_dir / p,
                detections_by_frame,
                output_path,
            ):
                print(f"[VIDEO] annotations enregistrees : {output_path}")

        with open(treated_path, "wb") as f:
            pickle.dump(treated_videos,f)
        n_done += 1
        print(
            f"[{n_done}/{len(videos_to_treat)}] {p} : "
            f"{len(rows)} detections"
        )
        if test and n_done>2:
            break

    print(
        f"\nTermine : {n_done} videos ecrits dans {crops_file_path}"
    )

    if n_failed:
        print(
            f"{len(n_failed)} videos en echec, a rejouer au prochain lancement : "
            f"{n_failed}"
        )

    if os.path.exists(crops_file_path):
        df = pd.read_csv(crops_file_path)
        df_video_label = df.filter(['video', 'detection']).drop_duplicates()
        video_count_by_label = (
            df_video_label
            .groupby('detection')['video']
            .nunique()
            .reset_index(name='n_videos')
        )

        print(video_count_by_label)
        return crops_file_path
    else:
        print('No detections')
        return None

