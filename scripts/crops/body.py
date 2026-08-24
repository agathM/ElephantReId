"""
Track MegaDetector detections over videos.

The script:
- Runs MegaDetector on frames sampled at the target FPS.
- Converts detections to Supervision format.
- Uses ByteTrack to assign persistent track IDs.
- Computes the sum of confidence scores for each track.
- Keeps only the track with the highest accumulated confidence.
- Saves the selected track detections into a CSV file.
"""


import os
import numpy as np
import supervision as sv
import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from typing import Callable
from supervision import VideoInfo, VideoSink
from pathlib import Path
import argparse
import pandas as pd

# PARAMETERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_device(0)
print(DEVICE)
file_selection = "entry_selection.ods"
crops_type = "body"
crop_selection = "entry_07_30_beach"
BODY_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops") / f"{crops_type}_{crop_selection}.csv"
TO_CROP_VIDEO_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw") / file_selection
Video_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras")
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

# Initializing the MegaDetectorV6 model for image detection
# Valid versions are MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e or MDV6-rtdetr-c
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov10-e")
#classifier = SpeciesNetClassifier(model_name="models/classification",device="cpu")



def process_video(
        source_path: Path,
        video_name: str,
        detection_model,
        target_fps: int = 2
):

    cap = cv2.VideoCapture(str(source_path))

    if not cap.isOpened():
        print(f"Cannot open {source_path}")
        return pd.DataFrame()


    source_fps = cap.get(cv2.CAP_PROP_FPS)

    stride = max(
        1,
        round(source_fps / target_fps)
    )


    tracker = sv.ByteTrack()


    frame_id = 0

    detections_rows = []

    track_scores = {}


    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )


    pbar = tqdm(
        total=total_frames,
        desc=video_name
    )


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        if frame_id % stride == 0:


            results_det = detection_model.single_image_detection(
                frame,
                img_path=frame_id
            )


            if (
                results_det is not None
                and len(results_det["detections"].confidence) > 0
            ):

                det = results_det["detections"]


                # Convert PytorchWildlife detections
                # to Supervision format

                detections = sv.Detections(
                    xyxy=det.xyxy,
                    confidence=det.confidence,
                    class_id=det.class_id
                )


                tracked = tracker.update_with_detections(
                    detections
                )


                if tracked.tracker_id is not None:

                    for bbox, conf, track_id in zip(
                        tracked.xyxy,
                        tracked.confidence,
                        tracked.tracker_id
                    ):

                        track_id = int(track_id)
                        conf = float(conf)


                        track_scores[track_id] = (
                            track_scores.get(track_id, 0)
                            + conf
                        )


                        detections_rows.append(
                            {
                                "video": video_name,
                                "frame": frame_id,
                                "track_id": track_id,
                                "label": "body",
                                "conf": round(conf,2),
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


    df = pd.DataFrame(
        detections_rows
    )


    best_track = max(
        track_scores,
        key=track_scores.get
    )


    print(
        f"{video_name}: keep track {best_track} "
        f"(confidence sum={track_scores[best_track]:.2f})"
    )


    return df[
        df.track_id == best_track
    ].copy()



def main():

    all_dfs = []


    dt = pd.read_excel(
        TO_CROP_VIDEO_PATH,
        header=0,
        engine="odf"
    )


    for _, row in dt.iterrows():

        video = row["video"]
        camera = row["camera"]
        indiv = row["indiv_name"]
        mode = row["mode"]


        src = None

        for ext in [
            ".mp4",
            ".MP4",
            ".AVI",
            ".avi"
        ]:

            candidate = (
                Video_PATH
                / camera
                / indiv
                / mode
                / f"{video}{ext}"
            )

            if candidate.exists():
                src = candidate
                break


        if src is None:
            print(f"Missing video: {video}")
            continue


        df = process_video(
            source_path=src,
            video_name=video,
            detection_model=detection_model,
            target_fps=2
        )


        if not df.empty:
            all_dfs.append(df)



    if len(all_dfs) == 0:
        print("No detections found")
        return


    all_df = pd.concat(
        all_dfs,
        ignore_index=True
    )


    all_df.to_csv(
        BODY_PATH,
        index=False
    )


if __name__ == "__main__":
    main()