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

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import warnings
import pickle
import os
import torch
from datetime import datetime

DEVICE = 0 if torch.cuda.is_available() else "cpu"

VIDEO_EXTENSIONS = {
    ".mp4",".AVI",".MP4"
}

BORDER = 1


CSV_COLUMNS = [
    "image",
    "video",
    "frame",
    "track_id",
    "label",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "area",
    "ratio",
    "indiv"
]


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
    id = None
    max_intersection = 0
    for _,dt_row in rows.iterrows():
        intersection = intersection_rate(dt_row, row)
        if intersection>identification_threshold and intersection>max_intersection:
            if id is not None:
                warnings.warn("Duplicate detection detected.")
            id = dt_row["indiv"]
            max_intersection = intersection
    return id

def get_partially_treated_video_to_treat(partially_treated_videos):
    # Find detections matching the frame_database whose segmentation was not saved
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
        label = row["label"]
        image = row["image"]

        if detection_id is not None:
            path = crops_dir / "unidentified" / label.replace(" ", "_") / image

            if os.path.exists(path):
                os.rename(path, crops_dir / label.replace(" ", "_") / image)
                detections.loc[idx, "indiv"] = detection_id
            else:
                videos_to_treat.add(row["video"])
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

    pd.DataFrame(
        rows,
        columns=CSV_COLUMNS,
    ).to_csv(
        crops_file_path,
        mode="a",
        header=not crops_file_path.exists(),
        index=False,
    )


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
        reset_tracker
):
    """
    Process one video.

    YOLO tracking is performed at a reduced FPS for efficiency.

    All detections are stored in csv_rows, which is filled only for this video:
    the caller is responsible for writing them once the video is done.
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
                                "label": label,
                                "conf": conf,
                                "track_id": track_id
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

                        x1, y1, x2, y2 = bbox_mask

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
                                    frame_number,

                                "track_id":
                                    track_id,

                                "label":
                                    label.replace(" ","_"),

                                "conf":
                                    conf,

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
                                    round(x2 - x1) / max(round(y2 - y1), 1)
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
                        row['indiv'] = indiv
                        csv_rows.append(row)


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
        reset_tracker
):
    """
    Process one video.

    YOLO tracking is performed at a reduced FPS for efficiency.

    All detections are stored in csv_rows, which is filled only for this video:
    the caller is responsible for writing them once the video is done.
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

                        last_detections.append(
                            {
                                "bbox": bbox,
                                "cls": cls,
                                "label": label,
                                "conf": conf,
                                "track_id": track_id
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
                                frame_number,

                            "track_id":
                                track_id,

                            "label":
                                label.replace(" ", "_"),

                            "conf":
                                conf,

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
                                round(x2 - x1) / max(round(y2 - y1), 1)
                        }
                        indiv = get_detection_id(row)

                        if indiv is None:
                            treated_videos['some_detect'].add(video_path.name)
                            if keep_detections_without_id:
                                cv2.imwrite(crops_dir / "unidentified" / label.replace(" ", "_") / image,
                                            frame[y1:y2, x1:x2])
                        else:
                            cv2.imwrite(crops_dir / label.replace(" ", "_") / image, frame[y1:y2, x1:x2])
                        row['indiv'] = indiv
                        csv_rows.append(row)

            frame_number += 1

        if video_path.name not in treated_videos['some_detect']:
            treated_videos['all_detect'].add(video_path.name)
    finally:
        cap.release()


# ============================================================
# Main
# ============================================================


def main(metadata:dict):
    global frame_database_df, crops_dir, crops_file_path, identification_threshold, single_indiv_videos
    home_dir = Path(metadata['home_dir'])
    frame_database = home_dir / metadata['frame_database']
    frame_database_df = pd.read_csv(frame_database)
    model_name = metadata['model_name']
    crops_dir = home_dir / metadata['crops_dir'] / model_name
    model_path = home_dir / metadata['model_dir'] / (model_name + ".pt")
    tracker = metadata['tracker']
    crops_file_path = home_dir / metadata['crops_dir'] / ( model_name + ".csv")
    videos_dir = home_dir / metadata['videos_dir']
    keep_detections_without_id = metadata["keep_detections_without_id"]
    conf = metadata["conf"]
    target_fps = metadata["fps"]
    identification_threshold = metadata["identification_threshold"]
    detection_task = "detect" if "detect" in model_name else "seg"
    single_indiv_videos_df = pd.read_csv(home_dir / metadata['video_database'])
    single_indiv_videos_df = single_indiv_videos_df[single_indiv_videos_df.several_indivs==False].filter(['indiv','video'])
    single_indiv_videos = {row['video']:row['indiv'] for _,row in single_indiv_videos_df.iterrows()}

    model = YOLO(
        model_path
    )


    print(
        "Classes:",
        model.names
    )
    for label in model.names.values():
        os.makedirs(crops_dir / label.replace(" ","_"), exist_ok=True)
        os.makedirs(crops_dir / label.replace(" ","_") / "unidentified", exist_ok=True)

    # Load camera/time maps
    video_to_camera, video_to_time = {}, {}
    for name, dst in [("video_to_camera.pkl", video_to_camera), ("video_to_time.pkl", video_to_time)]:
        pkl = Path("/home/agath/Documents/MNHN/code") / name
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
        videos_to_treat = video_paths.copy() - treated_videos["all_detect"]
        videos_to_treat.update(get_partially_treated_video_to_treat(treated_videos['some_detect']))

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

    for p, reset_tracker in videos_with_reset[:10]:

        rows = []
        args = {"model": model,
                "video_path": videos_dir / p,
                "csv_rows": rows,
                "conf_thresh": conf,
                "tracker": tracker,
                "treated_videos": treated_videos,
                "keep_detections_without_id": keep_detections_without_id,
                "crops_dir": crops_dir,
                "target_fps": target_fps,
                "reset_tracker": reset_tracker}
        try:
            if detection_task == "seg":
                try:
                    process_video_seg(**args)
                except TypeError:
                    args.pop("reset_tracker", None)
                    process_video_seg(**args)
            else:
                try:
                    process_video_crop(**args)
                except TypeError:
                    args.pop("reset_tracker", None)
                    process_video_crop(**args)

        except Exception as e:
            n_failed.append(p)
            print(
                f"[ECHEC] {p} : {type(e).__name__}: {e} "
                f"({len(rows)} detections non ecrites)"
            )
            continue

        append_rows(rows)
        with open(treated_path, "wb") as f:
            pickle.dump(treated_videos,f)
        n_done += 1
        print(
            f"[{n_done}/{len(videos_to_treat)}] {p} : "
            f"{len(rows)} detections"
        )

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
        df_video_label = df.filter(['video', 'label']).drop_duplicates()
        video_count_by_label = (
            df_video_label
            .groupby('label')['video']
            .nunique()
            .reset_index(name='n_videos')
        )

        print(video_count_by_label)
    else:
        print('No detections')

