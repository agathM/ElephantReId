# Create background-removed crops using MobileSAM.

# The script reads:
#     - videos from VIDEO_ROOT
#     - bounding boxes from CSV files

# It can be used for:
#     - body crops
#     - head crops
#     - ear crops

# The target object and the way MobileSAM is prompted are configurable.

# MASK_METHOD:
#     "target"
#         MobileSAM is prompted with the target bounding box.

#     "body"
#         MobileSAM is prompted with the body bounding box.
#         The complete MobileSAM mask is kept.

#     "body_intersect_target"
#         MobileSAM is prompted with the body bounding box.
#         The resulting mask is intersected with the target bounding box.

# Examples
# --------

# Body:
#     TARGET = "body"
#     MASK_METHOD = "target"

# Head:
#     TARGET = "head"
#     MASK_METHOD = "target"

# Ear:
#     TARGET = "ear"
#     MASK_METHOD = "target"

# Head using body as MobileSAM prompt:
#     TARGET = "head"
#     MASK_METHOD = "body_intersect_target"

# Ear using body as MobileSAM prompt:
#     TARGET = "ear"
#     MASK_METHOD = "body_intersect_target"

# The resulting image:
#     - keeps the original pixels inside the final mask
#     - replaces the background by black
#     - crops around the final mask -> not the bounding box detection
#     - adds a configurable black border around the mask


import cv2
import numpy as np
import pandas as pd
import torch
import time

from pathlib import Path
from mobile_sam import sam_model_registry, SamPredictor


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

VIDEO_ROOT = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/raw/cameras"
)

BODY_CSV_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/processed/"
    "crops/classifiers/raw/v2_body.csv"
)

HEAD_EAR_CSV_PATH = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/processed/"
    "crops/classifiers/raw/v2_head_ears.csv"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_ROOT = Path(
    "/sps/mnhn/ameurisse/ElephantReId/data/processed/"
    "crops/classifiers"
)


# ------------------------------------------------------------
# WHAT DO YOU WANT TO CROP?
#
# "body"
# "head"
# "ear"
# ------------------------------------------------------------

TARGET = "Head"


# ------------------------------------------------------------
# HOW SHOULD THE MASK BE COMPUTED?
#
# "target"
#     MobileSAM prompted with target bbox
# "body_intersect_target"
#     MobileSAM prompted with body bbox
#     AND intersected with target bbox
# ------------------------------------------------------------

MASK_METHOD = "target"


# ------------------------------------------------------------
# For ears, choose whether to keep:
#
# "all"
#     all ear detections
#
# "best"
#     most confident ear per video/frame
#
# "each"
#     process every CSV row separately
# ------------------------------------------------------------

EAR_SELECTION = "best"


# ------------------------------------------------------------
# MobileSAM
# ------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_TYPE = "vit_t"

SAM_CHECKPOINT = Path(
    "/pbs/home/a/ameurisse/ElephantReId/"
    "MobileSAM/weights/mobile_sam.pt"
)


# ------------------------------------------------------------
# Crop border
#
# Number of black pixels around the mask.
# ------------------------------------------------------------

BORDER = 1


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    OUTPUT_ROOT
    / f"{TARGET}/v4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK CONFIGURATION
# ============================================================

VALID_TARGETS = {
    "body",
    "Head",
    "Ear"
}

VALID_MASK_METHODS = {
    "target",
    "body_intersect_target"
}

if TARGET not in VALID_TARGETS:
    raise ValueError(
        f"Invalid TARGET={TARGET}. "
        f"Choose from {VALID_TARGETS}"
    )

if MASK_METHOD not in VALID_MASK_METHODS:
    raise ValueError(
        f"Invalid MASK_METHOD={MASK_METHOD}. "
        f"Choose from {VALID_MASK_METHODS}"
    )


# ============================================================
# INITIALIZE MOBILE SAM
# ============================================================

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    torch.cuda.set_device(0)


sam = sam_model_registry[MODEL_TYPE](
    checkpoint=str(SAM_CHECKPOINT)
)

sam.to(device=DEVICE)
sam.eval()

predictor = SamPredictor(sam)
MODEL_INFERENCE_TIMES = []


# ============================================================
# CSV LOADING
# ============================================================

def load_dataframes():
    """
    Load the required CSV files.

    Body boxes come from v2_body.csv.
    Head and ear boxes come from v2_head_ears.csv.
    """

    body = pd.read_csv(
        BODY_CSV_PATH
    )

    head_ear = pd.read_csv(
        HEAD_EAR_CSV_PATH
    )

    return body, head_ear


# ============================================================
# SELECT TARGET ROWS
# ============================================================

def prepare_target_dataframe(
    body: pd.DataFrame,
    head_ear: pd.DataFrame
) -> pd.DataFrame:

    if TARGET == "body":

        df = body.copy()

        # If several body detections exist,
        # keep the most confident one.
        if "conf" in df.columns:

            df = df.loc[
                df.groupby(
                    ["video", "frame"]
                )["conf"].idxmax()
            ]

        df = df.drop_duplicates(
            subset=["video", "frame"]
        )

        return df


    elif TARGET == "Head":

        df = head_ear[
            head_ear["label"] == "Head"
        ].copy()

        if "conf" in df.columns:

            df = df.loc[
                df.groupby(
                    ["video", "frame"]
                )["conf"].idxmax()
            ]

        df = df.drop_duplicates(
            subset=["video", "frame"]
        )

        return df


    elif TARGET == "Ear":

        df = head_ear[
            head_ear["label"].isin(
                ["Left_Ear", "Right_Ear"]
            )
        ].copy()

        if EAR_SELECTION == "best":

            df = df.loc[
                df.groupby(
                    ["video", "frame"]
                )["conf"].idxmax()
            ]

            df = df.drop_duplicates(
                subset=["video", "frame"]
            )

        elif EAR_SELECTION == "all":

            # Keep all rows, but avoid exact duplicates.
            df = df.drop_duplicates()

        elif EAR_SELECTION == "each":

            # Every ear detection gets its own crop.
            pass

        else:

            raise ValueError(
                f"Invalid EAR_SELECTION={EAR_SELECTION}"
            )

        return df


# ============================================================
# FIND VIDEO
# ============================================================

def find_video(
    video_name: str
) -> Path | None:

    for extension in [
        ".mp4",
        ".avi",
        ".AVI"
    ]:

        for candidate in VIDEO_ROOT.rglob(
            f"{video_name}{extension}"
        ):
            return candidate

    return None


# ============================================================
# EXTRACT FRAME
# ============================================================

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

    # CSV frame numbering:
    # frame 1 = first frame
    #
    # OpenCV:
    # frame 0 = first frame

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_id) - 1
    )

    ret, frame = cap.read()

    cap.release()

    if not ret:
        return None

    return frame


# ============================================================
# CLIP BBOX TO IMAGE
# ============================================================

def clip_box(
    box,
    image_shape
):

    h, w = image_shape[:2]

    x1 = max(
        0,
        min(w, int(round(box[0])))
    )

    y1 = max(
        0,
        min(h, int(round(box[1])))
    )

    x2 = max(
        0,
        min(w, int(round(box[2])))
    )

    y2 = max(
        0,
        min(h, int(round(box[3])))
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# CREATE RECTANGULAR BBOX MASK
# ============================================================

def create_bbox_mask(
    image_shape,
    box
) -> np.ndarray:

    h, w = image_shape[:2]

    x1, y1, x2, y2 = clip_box(
        box,
        image_shape
    )

    mask = np.zeros(
        (h, w),
        dtype=bool
    )

    if x2 > x1 and y2 > y1:

        mask[
            y1:y2,
            x1:x2
        ] = True

    return mask


# ============================================================
# MOBILE SAM
# ============================================================

def segment_with_box(
    image: np.ndarray,
    box
) -> np.ndarray:

    """
    Run MobileSAM using a bounding box prompt.
    """

    h, w = image.shape[:2]

    x1, y1, x2, y2 = clip_box(
        box,
        image.shape
    )

    if x2 <= x1 or y2 <= y1:

        return np.zeros(
            (h, w),
            dtype=bool
        )
    t_start = time.perf_counter()
    predictor.set_image(
        image
    )
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(
            [x1, y1, x2, y2]
        )[None, :],
        multimask_output=False
    )
    t_end = time.perf_counter()
    MODEL_INFERENCE_TIMES.append(
        t_end - t_start
    )

    return masks[0] > 0.5


# ============================================================
# COMPUTE FINAL MASK
# ============================================================

def compute_mask(
    image: np.ndarray,
    target_box,
    body_box
) -> np.ndarray:

    """
    Compute the final mask according to MASK_METHOD.
    """

    # --------------------------------------------------------
    # MobileSAM using target bbox
    # --------------------------------------------------------

    if MASK_METHOD == "target":

        return segment_with_box(
            image,
            target_box
        )


    # --------------------------------------------------------
    # MobileSAM using body bbox
    # then intersect with target bbox
    # --------------------------------------------------------

    if MASK_METHOD == "body_intersect_target":

        body_mask = segment_with_box(
            image,
            body_box
        )

        target_bbox_mask = create_bbox_mask(
            image.shape,
            target_box
        )

        return (
            body_mask
            & target_bbox_mask
        )


    raise ValueError(
        f"Unknown MASK_METHOD={MASK_METHOD}"
    )


# ============================================================
# CROP MASK
# ============================================================

def crop_masked_image(
    image: np.ndarray,
    mask: np.ndarray,
    border: int = 1
) -> np.ndarray | None:

    """
    1. Replace background by black.
    2. Find bounding box of mask.
    3. Add black border.
    4. Crop.
    """

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # Bounding box of the actual mask
    # --------------------------------------------------------

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    # --------------------------------------------------------
    # Add black border
    # --------------------------------------------------------

    x_min = max(
        0,
        x_min - border
    )

    y_min = max(
        0,
        y_min - border
    )

    x_max = min(
        w - 1,
        x_max + border
    )

    y_max = min(
        h - 1,
        y_max + border
    )

    # --------------------------------------------------------
    # Background = black
    # --------------------------------------------------------

    result = np.zeros_like(
        image
    )

    result[mask] = image[mask]

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    crop = result[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    return crop


# ============================================================
# MAIN
# ============================================================

def main():

    body, head_ear = load_dataframes()

    target_df = prepare_target_dataframe(
        body,
        head_ear
    )

    print()
    print(
        f"TARGET      = {TARGET}"
    )
    print(
        f"MASK_METHOD = {MASK_METHOD}"
    )
    print(
        f"OUTPUT      = {OUTPUT_DIR}"
    )
    print(
        f"Number of rows = {len(target_df)}"
    )

    # --------------------------------------------------------
    # Process each detection
    # --------------------------------------------------------

    for index, row in target_df.iterrows():

        video = row["video"]
        frame_id = int(row["frame"])
        track_id = row['track_id']

        print()
        print(
            "=" * 70
        )
        print(
            f"{video} - frame {frame_id}"
        )

        # ----------------------------------------------------
        # Target bbox
        # ----------------------------------------------------

        target_box = (
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"])
        )

        # ----------------------------------------------------
        # Body bbox
        #
        # Needed for:
        #     MASK_METHOD = "body_intersect_target"
        # ----------------------------------------------------

        body_rows = body[
            (body["video"] == video)
            & (body["frame"] == frame_id)
        ]

        if len(body_rows) == 0:

            print(
                "[SKIP] No body bbox"
            )

            continue

        if "conf" in body_rows.columns:

            body_row = body_rows.loc[
                body_rows["conf"].idxmax()
            ]

        else:

            body_row = body_rows.iloc[0]

        body_box = (
            float(body_row["x1"]),
            float(body_row["y1"]),
            float(body_row["x2"]),
            float(body_row["y2"])
        )

        # ----------------------------------------------------
        # Find video
        # ----------------------------------------------------

        video_path = find_video(
            video
        )

        if video_path is None:

            print(
                f"[SKIP] Video not found: {video}"
            )

            continue

        # ----------------------------------------------------
        # Extract frame
        # ----------------------------------------------------

        image = extract_frame(
            video_path,
            frame_id
        )

        if image is None:

            print(
                "[SKIP] Could not read frame"
            )

            continue

        # ----------------------------------------------------
        # Compute mask
        # ----------------------------------------------------

        mask = compute_mask(
            image,
            target_box,
            body_box
        )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        crop = crop_masked_image(
            image,
            mask,
            border=BORDER
        )

        if crop is None:

            print(
                "[SKIP] Empty MobileSAM mask"
            )

            continue

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        if TARGET == "Ear":

            label = str(
                row.get(
                    "label",
                    "ear"
                )
            )

            filename = (
                f"{video}"
                f"_frame_{frame_id:06d}"
                f"_{label}"
                f"_{track_id}.png"
            )

        else:

            filename = (
                f"{video}"
                f"_frame_{frame_id:06d}"
                f"_{TARGET}"
                f"_{track_id}.png"
            )

        output_path = (
            OUTPUT_DIR
            / filename
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        cv2.imwrite(
            str(output_path),
            crop
        )

        print(
            f"[OK] {output_path}"
        )

        print(
            f"     size = "
            f"{crop.shape[1]} x {crop.shape[0]}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
    n_calls = len(MODEL_INFERENCE_TIMES)

    if n_calls > 0:
        mean_time = (
            sum(MODEL_INFERENCE_TIMES)
            / n_calls
        )
        print()
        print("=" * 70)
        print(
            f"Model inference executed "
            f"{n_calls} time(s)"
        )
        print(
            f"Mean execution time: "
            f"{mean_time:.3f} s "
            f"({mean_time * 1000:.1f} ms)"
        )
        print("=" * 70)
    else:
        print()
        print("Model inference never executed")
