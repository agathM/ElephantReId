"""
Segment the elephant using the BODY bounding box as MobileSAM prompt,
then keep only the part of the segmentation inside the HEAD bounding box.

For selected videos:
- head detections are used to select frames,
- body bounding box is used as MobileSAM prompt,
- final mask = MobileSAM body segmentation INTERSECTED with head bbox,
- background is replaced by light blue,
- saves a two-panel comparison.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from mobile_sam import sam_model_registry, SamPredictor


# ============================================================
# PARAMETERS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.cuda.set_device(0)

print(f"Device: {DEVICE}")


CROP_CSV_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/raw/v2_head_ears.csv"
)

BODY_CSV_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/classifiers/raw/v2_body.csv"
)

VIDEO_ROOT = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/raw"
)

OUTPUT_DIR = Path(
    "/pbs/home/a/ameurisse/test_remove_bg_head_v2"
)

HEAD_LABELS = ["Head"]

MODEL_TYPE = "vit_t"

SAM_CHECKPOINT = Path(
    "/pbs/home/a/ameurisse/ElephantReId/MobileSAM/weights/mobile_sam.pt"
)

LIGHT_BLUE_BGR = (230, 216, 173)


# ============================================================
# INITIALIZE MOBILE SAM
# ============================================================

sam = sam_model_registry[MODEL_TYPE](
    checkpoint=str(SAM_CHECKPOINT)
)

sam.to(device=DEVICE)
sam.eval()

predictor = SamPredictor(sam)


# ============================================================
# VIDEO FUNCTIONS
# ============================================================

def find_video(video_name: str) -> Path | None:

    for extension in [".mp4", ".avi", ".AVI"]:

        for candidate in VIDEO_ROOT.rglob(
            f"{video_name}{extension}"
        ):
            return candidate

    return None


def extract_frame(
    video_path: Path,
    frame_id: int
) -> np.ndarray | None:

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_id)
    )

    ret, frame = cap.read()

    cap.release()

    return frame if ret else None


# ============================================================
# MOBILE SAM SEGMENTATION
# ============================================================

def segment_elephant(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> np.ndarray:

    """
    Segment the elephant using the BODY bounding box
    as the MobileSAM prompt.
    """

    h, w = image.shape[:2]

    x1 = max(
        0,
        int(round(x1))
    )

    y1 = max(
        0,
        int(round(y1))
    )

    x2 = min(
        w,
        int(round(x2))
    )

    y2 = min(
        h,
        int(round(y2))
    )

    if x2 <= x1 or y2 <= y1:
        return np.zeros(
            (h, w),
            dtype=bool
        )

    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(
            [x1, y1, x2, y2]
        )[None, :],
        multimask_output=False,
    )

    return masks[0] > 0.5


# ============================================================
# HEAD BBOX MASK
# ============================================================

def create_bbox_mask(
    image_shape,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> np.ndarray:

    """
    Create a rectangular boolean mask corresponding
    to the HEAD bounding box.
    """

    h, w = image_shape[:2]

    x1 = max(
        0,
        int(round(x1))
    )

    y1 = max(
        0,
        int(round(y1))
    )

    x2 = min(
        w,
        int(round(x2))
    )

    y2 = min(
        h,
        int(round(y2))
    )

    bbox_mask = np.zeros(
        (h, w),
        dtype=bool
    )

    if x2 > x1 and y2 > y1:

        bbox_mask[
            y1:y2,
            x1:x2
        ] = True

    return bbox_mask


# ============================================================
# FINAL MASK
# ============================================================

def get_head_mask(
    image: np.ndarray,
    body_box,
    head_box
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    1. Segment elephant using BODY bbox.
    2. Create rectangular mask from HEAD bbox.
    3. Intersect the two masks.

    final_mask = body_segmentation AND head_bbox
    """

    # --------------------------------------------------------
    # MobileSAM segmentation using BODY bbox
    # --------------------------------------------------------

    body_mask = segment_elephant(
        image,
        body_box[0],
        body_box[1],
        body_box[2],
        body_box[3]
    )

    # --------------------------------------------------------
    # HEAD bbox mask
    # --------------------------------------------------------

    head_bbox_mask = create_bbox_mask(
        image.shape,
        head_box[0],
        head_box[1],
        head_box[2],
        head_box[3]
    )

    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    final_mask = (
        body_mask
        & head_bbox_mask
    )

    return (
        body_mask,
        head_bbox_mask,
        final_mask
    )


# ============================================================
# REMOVE BACKGROUND
# ============================================================

def remove_background(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:

    background = np.full_like(
        image,
        LIGHT_BLUE_BGR
    )

    return np.where(
        mask[..., None],
        image,
        background
    )


# ============================================================
# PLOT
# ============================================================

def plot_comparison(
    image: np.ndarray,
    result_body_head: np.ndarray,
    result_head: np.ndarray,
    output_path: Path,
    body_box,
    head_box
) -> None:

    fig, (
        ax_left,
        ax_middle,
        ax_right
    ) = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    # ========================================================
    # 1. ORIGINAL
    # ========================================================

    ax_left.imshow(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )

    # Body bbox
    bx1, by1, bx2, by2 = body_box

    body_rect = plt.Rectangle(
        (bx1, by1),
        bx2 - bx1,
        by2 - by1,
        fill=False,
        edgecolor="red",
        linewidth=2
    )

    ax_left.add_patch(body_rect)

    # Head bbox
    hx1, hy1, hx2, hy2 = head_box

    head_rect = plt.Rectangle(
        (hx1, hy1),
        hx2 - hx1,
        hy2 - hy1,
        fill=False,
        edgecolor="cyan",
        linewidth=2
    )

    ax_left.add_patch(head_rect)

    ax_left.set_title(
        "Original\n"
        "red = body / cyan = head"
    )

    ax_left.axis("off")


    # ========================================================
    # 2. BODY SAM ∩ HEAD BBOX
    # ========================================================

    ax_middle.imshow(
        cv2.cvtColor(
            result_body_head,
            cv2.COLOR_BGR2RGB
        )
    )

    ax_middle.set_title(
        "MobileSAM body bbox\n"
        "∩ head bbox"
    )

    ax_middle.axis("off")


    # ========================================================
    # 3. HEAD BBOX → MobileSAM
    # ========================================================

    ax_right.imshow(
        cv2.cvtColor(
            result_head,
            cv2.COLOR_BGR2RGB
        )
    )

    ax_right.set_title(
        "MobileSAM head bbox"
    )

    ax_right.axis("off")


    # ========================================================
    # SAVE
    # ========================================================

    fig.tight_layout()

    fig.savefig(
        str(output_path),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    body = pd.read_csv(
        BODY_CSV_PATH
    )

    head = pd.read_csv(
        CROP_CSV_PATH
    )

    # --------------------------------------------------------
    # Keep head detections
    # --------------------------------------------------------

    head = head[
        head["label"].isin(
            HEAD_LABELS
        )
    ].copy()

    # Most confident head per video/frame
    best = head.loc[
        head.groupby(
            ["video", "frame"]
        )["conf"].idxmax()
    ]

    best = best.drop_duplicates(
        subset=[
            "video",
            "frame"
        ]
    )

    # --------------------------------------------------------
    # Videos
    # --------------------------------------------------------

    videos = [
        "SEB_180392",
        "SEB_180424",
        "SEB_180466",
        "SEB_180356",
        "SEB_180361",
        "SEB_180362",
        "SEB_125081",
        "SEB_125090",
        "SEB_126178"
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    for video in videos:

        frames = sorted(
            best[
                best["video"] == video
            ]["frame"]
            .drop_duplicates()
            .tolist()
        )

        print(
            f"\n{video}"
        )

        print(
            f"Available frames: {frames}"
        )

        if len(frames) >= 3:

            frame_ids = [
                frames[0],
                frames[len(frames) // 2],
                frames[-1]
            ]

        else:

            frame_ids = frames

        # ----------------------------------------------------
        # Process frames
        # ----------------------------------------------------

        for frame_id in frame_ids:

            # ------------------------------------------------
            # HEAD bbox
            # ------------------------------------------------

            head_rows = head[
                (head["video"] == video)
                & (head["frame"] == frame_id)
            ]

            if len(head_rows) == 0:

                print(
                    f"[SKIP] No head bbox: "
                    f"{video}, frame {frame_id}"
                )

                continue

            # Most confident head
            head_row = head_rows.loc[
                head_rows["conf"].idxmax()
            ]

            head_box = (
                float(head_row["x1"]),
                float(head_row["y1"]),
                float(head_row["x2"]),
                float(head_row["y2"])
            )

            # ------------------------------------------------
            # BODY bbox
            # ------------------------------------------------

            body_rows = body[
                (body["video"] == video)
                & (body["frame"] == frame_id)
            ]

            if len(body_rows) == 0:

                print(
                    f"[SKIP] No body bbox: "
                    f"{video}, frame {frame_id}"
                )

                continue

            body_row = body_rows.iloc[0]

            body_box = (
                float(body_row["x1"]),
                float(body_row["y1"]),
                float(body_row["x2"]),
                float(body_row["y2"])
            )

            # ------------------------------------------------
            # Video
            # ------------------------------------------------

            video_path = find_video(
                video
            )

            if video_path is None:

                print(
                    f"[SKIP] Missing video: "
                    f"{video}"
                )

                continue

            # ------------------------------------------------
            # Frame
            # ------------------------------------------------

            frame = extract_frame(
                video_path,
                frame_id
            )

            if frame is None:

                print(
                    f"[SKIP] Cannot read frame "
                    f"{frame_id} of {video}"
                )

                continue

            # ========================================================
            # METHOD 1
            # MobileSAM avec BODY bbox
            # puis intersection avec HEAD bbox
            # ========================================================
            
            body_mask = segment_elephant(
                frame,
                body_box[0],
                body_box[1],
                body_box[2],
                body_box[3]
            )
            
            head_bbox_mask = create_bbox_mask(
                frame.shape,
                head_box[0],
                head_box[1],
                head_box[2],
                head_box[3]
            )
            
            final_mask_body_head = (
                body_mask
                & head_bbox_mask
            )
            
            result_body_head = remove_background(
                frame,
                final_mask_body_head
            )
            
            
            # ========================================================
            # METHOD 2
            # MobileSAM directement avec HEAD bbox
            # ========================================================
            
            head_mask = segment_elephant(
                frame,
                head_box[0],
                head_box[1],
                head_box[2],
                head_box[3]
            )
            
            result_head = remove_background(
                frame,
                head_mask
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_path = (
                OUTPUT_DIR
                / (
                    f"{video}"
                    f"_frame_{int(frame_id):06d}"
                    f"_head.png"
                )
            )

            plot_comparison(
                    frame,
                    result_body_head,
                    result_head,
                    output_path,
                    body_box,
                    head_box
                )

            print(
                f"[OK] Saved {output_path}"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()