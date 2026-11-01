# English summary:
# Compare MobileSAM and SAM3 head segmentation using body/head bounding boxes.
# The figure contains 5 panels arranged as 3 panels on the first row and
# 2 panels on the second row. Segmentation panels are cropped around the
# head bounding box to show only the relevant image area.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from PIL import Image

from mobile_sam import sam_model_registry, SamPredictor

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


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
    "/pbs/home/a/ameurisse/test_remove_bg_head"
)


HEAD_LABELS = ["Head"]


# ============================================================
# MOBILE SAM
# ============================================================

MODEL_TYPE = "vit_t"

SAM_CHECKPOINT = Path(
    "/pbs/home/a/ameurisse/ElephantReId/MobileSAM/weights/mobile_sam.pt"
)

sam = sam_model_registry[MODEL_TYPE](
    checkpoint=str(SAM_CHECKPOINT)
)

sam.to(device=DEVICE)
sam.eval()

mobile_sam_predictor = SamPredictor(sam)


# ============================================================
# SAM3
# ============================================================

# Set this if SAM3 is installed from the official repository.
# If the checkpoint is not already available locally, the official
# model builder can download it from Hugging Face.
SAM3_CHECKPOINT = None

sam3_model = build_sam3_image_model(
    device=DEVICE,
    checkpoint_path=SAM3_CHECKPOINT,
    load_from_HF=True,
    enable_segmentation=True,
)

sam3_model.eval()

sam3_processor = Sam3Processor(
    sam3_model,
    device=DEVICE,
)


# ============================================================
# DISPLAY PARAMETERS
# ============================================================

LIGHT_BLUE_BGR = (230, 216, 173)

# Fraction of the head box size added around the crop.
CROP_MARGIN = 0.30


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
# GENERIC BBOX UTILITIES
# ============================================================

def clip_box(
    image: np.ndarray,
    box
) -> tuple[int, int, int, int]:

    h, w = image.shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(0, min(w, int(round(x1))))
    y1 = max(0, min(h, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))

    return x1, y1, x2, y2


def create_bbox_mask(
    image_shape,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> np.ndarray:

    h, w = image_shape[:2]

    x1 = max(0, min(w, int(round(x1))))
    y1 = max(0, min(h, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))

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
# MOBILE SAM SEGMENTATION
# ============================================================

def segment_mobile_sam(
    image: np.ndarray,
    box
) -> np.ndarray:

    """
    Segment an elephant using a bounding box as the MobileSAM prompt.
    """

    x1, y1, x2, y2 = clip_box(
        image,
        box
    )

    if x2 <= x1 or y2 <= y1:

        return np.zeros(
            image.shape[:2],
            dtype=bool
        )

    mobile_sam_predictor.set_image(image)

    masks, _, _ = mobile_sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(
            [x1, y1, x2, y2]
        )[None, :],
        multimask_output=False,
    )

    return masks[0] > 0.5


# ============================================================
# SAM3 SEGMENTATION
# ============================================================

def box_xyxy_to_sam3_box(
    box,
    image_shape
) -> list:

    """
    Convert an absolute xyxy box to the normalized SAM3
    [center_x, center_y, width, height] format.
    """

    h, w = image_shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(0, min(w, float(x1)))
    y1 = max(0, min(h, float(y1)))
    x2 = max(0, min(w, float(x2)))
    y2 = max(0, min(h, float(y2)))

    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h

    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return [
        cx,
        cy,
        bw,
        bh,
    ]


def segment_sam3(
    image: np.ndarray,
    box
) -> np.ndarray:

    """
    Segment an elephant using a bounding box as a SAM3
    geometric prompt.

    SAM3 expects:
        [center_x, center_y, width, height]

    with coordinates normalized to [0, 1].
    """

    # Convert OpenCV BGR -> RGB -> PIL
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        image_rgb
    )

    sam3_box = box_xyxy_to_sam3_box(
        box,
        image.shape
    )

    # Set image once for this prediction.
    state = sam3_processor.set_image(
        pil_image
    )

    # Positive box prompt.
    state = sam3_processor.add_geometric_prompt(
        box=sam3_box,
        label=True,
        state=state,
    )

    if "masks" not in state or len(state["masks"]) == 0:

        return np.zeros(
            image.shape[:2],
            dtype=bool
        )

    # SAM3 returns masks at original image resolution.
    masks = state["masks"]

    # Usually [N, H, W] or [N, 1, H, W].
    mask = masks[0]

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask)

    return mask.astype(bool)


# ============================================================
# FINAL MASKS
# ============================================================

def get_masks(
    image: np.ndarray,
    body_box,
    head_box
):

    # --------------------------------------------------------
    # MobileSAM using BODY bbox
    # --------------------------------------------------------

    mobile_body_mask = segment_mobile_sam(
        image,
        body_box
    )

    # --------------------------------------------------------
    # MobileSAM using HEAD bbox
    # --------------------------------------------------------

    mobile_head_mask = segment_mobile_sam(
        image,
        head_box
    )

    # --------------------------------------------------------
    # SAM3 using BODY bbox
    # --------------------------------------------------------

    sam3_body_mask = segment_sam3(
        image,
        body_box
    )

    # --------------------------------------------------------
    # SAM3 using HEAD bbox
    # --------------------------------------------------------

    sam3_head_mask = segment_sam3(
        image,
        head_box
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
    # Intersection with HEAD bbox
    # --------------------------------------------------------

    mobile_body_head_mask = (
        mobile_body_mask
        & head_bbox_mask
    )

    sam3_body_head_mask = (
        sam3_body_mask
        & head_bbox_mask
    )

    return (
        mobile_body_head_mask,
        mobile_head_mask,
        sam3_body_head_mask,
        sam3_head_mask,
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
# CROP AROUND HEAD
# ============================================================

def crop_around_head(
    image: np.ndarray,
    head_box,
    margin: float = CROP_MARGIN
) -> np.ndarray:

    """
    Crop the image around the head bounding box.

    The margin is expressed as a fraction of the head
    bounding-box width/height.
    """

    h, w = image.shape[:2]

    x1, y1, x2, y2 = head_box

    box_w = x2 - x1
    box_h = y2 - y1

    margin_x = box_w * margin
    margin_y = box_h * margin

    crop_x1 = max(
        0,
        int(round(x1 - margin_x))
    )

    crop_y1 = max(
        0,
        int(round(y1 - margin_y))
    )

    crop_x2 = min(
        w,
        int(round(x2 + margin_x))
    )

    crop_y2 = min(
        h,
        int(round(y2 + margin_y))
    )

    return image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


# ============================================================
# PLOT
# ============================================================

def plot_comparison(
    image: np.ndarray,
    result_mobile_body_head: np.ndarray,
    result_mobile_head: np.ndarray,
    result_sam3_body_head: np.ndarray,
    result_sam3_head: np.ndarray,
    output_path: Path,
    body_box,
    head_box
) -> None:

    # ========================================================
    # Crop segmentation images around the HEAD bbox
    # ========================================================

    cropped_mobile_body_head = crop_around_head(
        result_mobile_body_head,
        head_box
    )

    cropped_mobile_head = crop_around_head(
        result_mobile_head,
        head_box
    )

    cropped_sam3_body_head = crop_around_head(
        result_sam3_body_head,
        head_box
    )

    cropped_sam3_head = crop_around_head(
        result_sam3_head,
        head_box
    )

    # ========================================================
    # Figure: 3 images on first row, 2 on second row
    # ========================================================

    fig = plt.figure(
        figsize=(10, 6)
    )

    gs = fig.add_gridspec(
        2,
        6,
        hspace=0.25,
        wspace=0.08
    )

    ax_original = fig.add_subplot(
        gs[0, 0:2]
    )

    ax_mobile_body = fig.add_subplot(
        gs[0, 2:4]
    )

    ax_mobile_head = fig.add_subplot(
        gs[0, 4:6]
    )

    ax_sam3_body = fig.add_subplot(
        gs[1, 1:3]
    )

    ax_sam3_head = fig.add_subplot(
        gs[1, 3:5]
    )

    # ========================================================
    # 1. ORIGINAL
    # ========================================================

    ax_original.imshow(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )

    bx1, by1, bx2, by2 = body_box

    body_rect = plt.Rectangle(
        (bx1, by1),
        bx2 - bx1,
        by2 - by1,
        fill=False,
        edgecolor="red",
        linewidth=1.5
    )

    ax_original.add_patch(
        body_rect
    )

    hx1, hy1, hx2, hy2 = head_box

    head_rect = plt.Rectangle(
        (hx1, hy1),
        hx2 - hx1,
        hy2 - hy1,
        fill=False,
        edgecolor="cyan",
        linewidth=1.5
    )

    ax_original.add_patch(
        head_rect
    )

    ax_original.set_title(
        "Original\n"
        "red = body / cyan = head",
        fontsize=9
    )

    ax_original.axis("off")

    # ========================================================
    # 2. MOBILE SAM BODY ∩ HEAD
    # ========================================================

    ax_mobile_body.imshow(
        cv2.cvtColor(
            cropped_mobile_body_head,
            cv2.COLOR_BGR2RGB
        )
    )

    ax_mobile_body.set_title(
        "MobileSAM\n"
        "body ∩ head",
        fontsize=9
    )

    ax_mobile_body.axis("off")

    # ========================================================
    # 3. MOBILE SAM HEAD
    # ========================================================

    ax_mobile_head.imshow(
        cv2.cvtColor(
            cropped_mobile_head,
            cv2.COLOR_BGR2RGB
        )
    )

    ax_mobile_head.set_title(
        "MobileSAM\n"
        "head bbox",
        fontsize=9
    )

    ax_mobile_head.axis("off")

    # ========================================================
    # 4. SAM3 BODY ∩ HEAD
    # ========================================================

    ax_sam3_body.imshow(
        cv2.cvtColor(
            cropped_sam3_body_head,
            cv2.COLOR_BGR2RGB
        )
    )

    ax_sam3_body.set_title(
        "SAM3\n"
        "body ∩ head",
        fontsize=9
    )

    ax_sam3_body.axis("off")

    # ========================================================
    # 5. SAM3 HEAD
    # ========================================================

    ax_sam3_head.imshow(
        cv2.cvtColor(
            cropped_sam3_head,
            cv2.COLOR_BGR2RGB
        )
    )

    ax_sam3_head.set_title(
        "SAM3\n"
        "head bbox",
        fontsize=9
    )

    ax_sam3_head.axis("off")

    # ========================================================
    # SAVE
    # ========================================================

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
            # SEGMENTATION
            # ========================================================

            (
                mobile_body_head_mask,
                mobile_head_mask,
                sam3_body_head_mask,
                sam3_head_mask,
            ) = get_masks(
                frame,
                body_box,
                head_box
            )

            # ========================================================
            # REMOVE BACKGROUND
            # ========================================================

            result_mobile_body_head = remove_background(
                frame,
                mobile_body_head_mask
            )

            result_mobile_head = remove_background(
                frame,
                mobile_head_mask
            )

            result_sam3_body_head = remove_background(
                frame,
                sam3_body_head_mask
            )

            result_sam3_head = remove_background(
                frame,
                sam3_head_mask
            )

            # ========================================================
            # SAVE
            # ========================================================

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
                result_mobile_body_head,
                result_mobile_head,
                result_sam3_body_head,
                result_sam3_head,
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