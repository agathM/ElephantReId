"""body crops stored into .csv
use .csv files to known which video belongs to whom
avoids recalculating crops if incorrect individual"""

import os
import numpy as np
import supervision as sv
import torch
from PytorchWildlife.models import detection as pw_detection
#from speciesnet.classifier import SpeciesNetClassifier
#from speciesnet.utils import PreprocessedImage
from PytorchWildlife import utils as pw_utils
from torchvision import transforms
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
crop_selection = "entry_07_30"
BODY_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/processed/crops") / f"{crops_type}_{crop_selection}.csv"
TO_CROP_VIDEO_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw") / file_selection
Video_PATH = Path("/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras")
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

# Initializing the MegaDetectorV6 model for image detection
# Valid versions are MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e or MDV6-rtdetr-c
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov10-e")
#classifier = SpeciesNetClassifier(model_name="models/classification",device="cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([480,480]),
    transforms.ToTensor(),
])



# def callback(frame: np.ndarray, index: int,target_folder: str,frame_id: int) -> np.ndarray:
#     results_det = detection_model.single_image_detection(frame, img_path=index)
#     crop_id = 0
#     for xyxy in results_det["detections"].xyxy:
#         cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
#         # classifier doesn't work -> blank
#         # preprocess_img = PreprocessedImage(transform(cropped_image),480,480)
#         # arr = preprocess_img.arr / 255
#         # arr = np.transpose(arr, (1, 2, 0))
#         # batch_arr = np.expand_dims(arr, axis=0)
#         # batch_tensor = torch.from_numpy(batch_arr.astype(np.float32)).to(DEVICE)
#         # logits = classifier.model(batch_tensor)
#         # scores = torch.softmax(logits, dim=-1)
#         # scores, indices = torch.topk(scores, k=1, dim=-1)
#         # label = classifier.labels[indices.item()]
#         # label, idx
#         # 5033744e-5b47-40e4-b7b6-9fda865778a0;mammalia;proboscidea;elephantidae;elephas;maximus;asian elephant 1566
#         # 55631055-3e0e-4b7a-9612-dedebe9f78b0;mammalia;proboscidea;elephantidae;loxodonta;africana;african elephant 2276
#         # %matplotlib inline
#         # plt.imshow(cropped_image)
#         # plt.show()
#         crop_id+=1
#         img = Image.fromarray(cropped_image)
#         file_path = f"frame_{frame_id}_crop_{crop_id}.png"
#         img.save(os.path.join(target_folder,file_path))
#     return None

def callback(frame: np.ndarray, index: int, video_name: str):
    results_det = detection_model.single_image_detection(frame, img_path=index)

    if (
        results_det is None
        or len(results_det["detections"].confidence) == 0
    ):
        return None

    max_conf_idx = np.argmax(results_det["detections"].confidence)

    return {
        "video": video_name,
        "frame": index,
        "label": "body",
        "conf": float(results_det["detections"].confidence[max_conf_idx]),
        "x1": round(float(results_det["detections"].xyxy[max_conf_idx, 0])),
        "y1": round(float(results_det["detections"].xyxy[max_conf_idx, 1])),
        "x2": round(float(results_det["detections"].xyxy[max_conf_idx, 2])),
        "y2": round(float(results_det["detections"].xyxy[max_conf_idx, 3])),
    }
    
def process_video(
    source_path: str,
    video_name: str,
    callback,
    target_fps: int = 2,
):
    cap = cv2.VideoCapture(str(source_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, round(fps / target_fps))

    frame_idx = 0
    rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            result = callback(
                frame=frame,
                index=frame_idx,
                video_name=video_name
            )

            if result is not None:
                rows.append(result)

        frame_idx += 1

    cap.release()

    return rows

def main() -> None:
    """
    Extract body crops from a list of identified individuals in a .csv and the data raw folder 
    and save them into the data processed folder > crops > body
    """
    all_rows = [] ; videos_treated = []
    if BODY_PATH.exists():
        crop_df = pd.read_csv(BODY_PATH)
        videos_treated = set(crop_df['video'].unique())
        all_rows = crop_df.to_dict(orient="records")
        
    dt = pd.read_excel(TO_CROP_VIDEO_PATH,header=0,engine='odf')
    for _,row in dt.iterrows():
        indiv = row['indiv_name'] ; mode = row['mode']
        src = Video_PATH / row['camera'] / row['indiv_name'] / row['mode'] / (row['video'] + ".mp4")
        if not src.exists():
            src = Video_PATH / row['camera'] / row['indiv_name'] / row['mode'] / (row['video'] + ".AVI")
            if not src.exists():
                src = Video_PATH / row['camera'] / row['indiv_name'] / row['mode'] / (row['video'] + ".MP4")
                continue
        video = row['video']
        if video in videos_treated:
            continue
        rows = process_video(
                    source_path=src,
                    video_name=video,
                    callback=callback,
                    target_fps=2,
                )
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(
        BODY_PATH,
        index=False,
    )
if __name__=="__main__":
    main()