# ElephantReId - Identification pipeline (`pipeline_branch`)

`scripts/identification/pipeline.py` is the end-to-end entry point of the
project. It reads a YAML configuration, runs detection/tracking on videos,
filters the resulting crops, builds train/validation splits, optionally
fine-tunes a feature extractor, evaluates it with a cosine-similarity search,
and records every experiment in a summary table.

The pipeline is organized in five stages:

```text
config/pipeline.yaml
        |
        v
[1] detection + tracking      -> crops + detection CSV + treated-video state
        |
        v
[2] filtering + split         -> filtered detections + event/run/split tables
        |
        v
[3] preprocessing/augmentation + training (optional) -> weights + metrics
        |
        v
[4] cosine-similarity evaluation -> validation predictions + accuracy
        |
        v
[5] experiment summary        -> all_experiments.csv
```

All paths below are relative to `home_dir` (from `config/pipeline.yaml`) unless
stated otherwise. The configuration file itself and the `models/` directory are
relative to the repository root.

---

## 1. Usage

Run from the repository root so that flat imports
(`import train_classifier`, `import detect_and_track`, ...) and
`REPO_ROOT / "models"` resolve correctly:

```bash
python scripts/identification/pipeline.py -c config/pipeline.yaml
```

The scripts can not also be run individually.

### CLI arguments of `pipeline.py`

| Argument | Default | Description |
|---|---|---|
| `-c`, `--config` | `config/pipeline.yaml` | YAML configuration file. It is copied into the result folder. |
| `--dry-run` | `False` | Accepted by the parser but currently has no effect: the pipeline still runs all stages. |

### Module-level switches (in `pipeline.py`, not in YAML)

These constants must be edited directly in `scripts/identification/pipeline.py`:

| Constant            | Default                                                   | Description                                                                                                                                                           |
|---------------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DETECT_CHOICES`    | `['ear_seg', 'body_detect', 'ear_detect', 'head_detect']` | Detection classes to run. Each value selects a model and the labels to keep. Remove a value to skip that detection.                                                   |
| `MODE`              | `["Day", "Night"]`                                        | Video-database `mode` values kept for detection.                                                                                                                      |
| `RERUN_SOME_DETECT` | `False`                                                   | If `True`, videos with unmatched detections (`some_detect`) are re-examined and retried. If `False`, all videos already in `all_detect` or `some_detect` are skipped. |
| `TEST`              | `False`                                                   | Debug mode: detection stops after 3 videos and splitting keeps only events `0..9`.                                                                                    |
| `ASSYMETRY`         | `False`                                                   | If `False`, left and right ears share the same label (`indiv_ear_seg`, `indiv_ear_detect`). If `True`, the exact detection class is used (`indiv_left_ear_seg`, ...), i.e. one class per ear side (roughly twice as many classes). |
| `ARCHITECTURE`      | `"ResNet50"`                                              | Feature extractor (`MiewID`, `MegaDescriptor`, `ResNet152`, `ResNet50`), or `None` to read `feature_extractor.name`, `last_layer` and `last_frozen_layer` from YAML.  |
| `AUG_PARAMS`        | all `True`                                                | Enables/disables each augmentation: `rotation`, `translation`, `scaling`, `saturation`, `hue`. Parameters come from `data_augmentation.torch_transformation` of YAML. |
| `TRAIN_CLASSIFIER`  | `True`                                                    | Whether `train_classifier.py` runs **at all**. If `False`, no training happens (not even the head): the pipeline prepares data and runs cosine similarity with the pretrained backbone. |
| `FINETUNE`          | `False`                                                   | Only used when `TRAIN_CLASSIFIER = True`; decides **what** is trained. If `False`, the backbone stays fully frozen (whatever `defreeze_epoch`) and only the `ClassificationHead` is trained. If `True`, the layers between `last_frozen_layer` and `last_layer` are fine-tuned from `defreeze_epoch`. |

Typical combinations:

| `TRAIN_CLASSIFIER` | `FINETUNE` | Behaviour |
|---|---|---|
| `True`  | `True`  | Full training: head + backbone fine-tuning from `defreeze_epoch`; backbone weights saved. |
| `True`  | `False` | Head-only training on frozen-backbone features; no weights saved. |
| `False` | –      | No training: data preparation + cosine similarity on the pretrained backbone. |

### Detection models selected by `DETECT_CHOICES`

| `DETECT_CHOICES` value | Model key | Default confidence | Classes kept |
|---|---|---|---|
| `ear_seg` | `seg_ears_lr_v2` | `0.1` | `ear_seg` (matches `left_ear_seg` and `right_ear_seg`) |
| `body_detect` | `MDV6-yolov10-e` | `0.1` | `body_detect` (MegaDetector) |
| `ear_detect` | `head_ears_detect_09_08` | `0.1` | `ear_detect` (matches `left_ear_detect` and `right_ear_detect`) |
| `head_detect` | `head_ears_detect_09_08` | `0.1` | `head_detect` |

Two choices can share the same model: `ear_detect` and `head_detect` both use
`head_ears_detect_09_08`, and their `labels_to_keep` lists are merged.

---

## 2. Inputs

### 2.1 Input files and folders

```text
<home_dir>/
├── video_database.csv                  # one row per individual per video
├── frame_database.csv                  # ear localisation of individuals per frame
├── data/raw/indiv_videos/              # source videos (detection.videos_dir)
├── data/processed/crops/               # crops_dir
│   ├── <model>.csv                     # detection metadata for this model
│   ├── <model>/treated_videos.pkl      # processed-video state
│   ├── <model>/<detection_class>/      # identified crops
│   ├── <model>/unidentified/<class>/   # unmatched crops (optional)
│   └── split_files/                    # split_dir
│       ├── splits_metadata.csv
│       ├── vN_splits.csv
│       └── vN_selected_images.csv
└── data/results/                       # paths.output (results root)

<repository_root>/
└── models/
    ├── seg_ears_lr_v2.pt
    ├── head_ears_detect_09_08.pt
    └── MDV6-yolov10-e.pt               # optional: MegaDetector is downloaded otherwise
```

### 2.2 `video_database.csv`

One row per individual appearing in a video. The pipeline uses `video`, `time`,
`camera`, `mode`, `several_indivs` and `indiv`.

| Column | Description |
|---|---|
| `indiv` | Individual name (one row per individual in the video). |
| `video` | Video filename (including extension). |
| `time` | Video creation time (`YYYY/MM/DD HH:MM:SS` or ISO format). Used for event grouping and tracker reset. |
| `camera` | Camera name. Used for event grouping and tracker reset. |
| `mode` | `Day` or `Night`. Filtered with `MODE`. |
| `several_indivs` | `True`/`False`. If `False`, every detection in the video is directly assigned to this individual. |
| `entry`, `entry_day` | Optional metadata; not used by the pipeline. |
| `date` | Optional metadata; not used by the pipeline. |
| `Description` | Free text; not used by the pipeline. |

### 2.3 `frame_database.csv`

Database of localized ear detections used to associate a tracked crop with an
individual when several animals appear in the same video. The pipeline reads
`video`, `frame`, `x1`, `y1`, `x2`, `y2` and `indiv`; rows with an empty
`indiv` are dropped.

| Column | Description |
|---|---|
| `video` | Video filename. |
| `frame` | Frame number. |
| `x1`, `y1`, `x2`, `y2` | Bounding box of the segmented/identified ear. |
| `indiv` | Individual name. Required for linking. |
| `track`, `label`, `image`, `conf`, `area`, `ratio`, `avg_conf` | Typical additional detection columns (same family as the detection CSV); not required by the linking code. |

Association rule:

- if the video has `several_indivs == False`, the individual comes from
  `video_database.csv`;
- otherwise, the pipeline computes the proportion of the frame-database ear box
  that is contained in the detection box. The best match is kept if this
  proportion is greater than `detection.identification_threshold`.

### 2.4 Videos and models

- Videos must be in `detection.videos_dir` (`data/raw/indiv_videos` by default).
- YOLO models must be placed in `models/<model_key>.pt`.
- If `models/<model_key>.pt` is missing, the code treats `<model_key>` as a
  MegaDetectorV6 version and instantiates it through `PytorchWildlife`.
- For YOLO models, the processing mode is inferred from the model name:
  `"detect"` in the name -> bounding-box pipeline, otherwise segmentation
  pipeline. Detection class names must therefore contain `detect` or `seg`.

### 2.5 Input detection CSV (`crops_dir/<model>.csv`)

This CSV is produced by `detect_and_track.py` and consumed by the filtering
stage. The pipeline reads columns by name, so their order is not significant.

| Column | Type | Description |
|---|---|---|
| `image` | str | Crop filename: `<video_stem>_fr_<frame:06d>_t<track_id>.png`. |
| `video` | str | Source video filename. |
| `frame` | int | Original frame number in the video. |
| `track_id` | int | Tracker identifier. |
| `detection` | str | Detection class (e.g. `head_detect`, `left_ear_seg`). |
| `conf` | float | Detection confidence. |
| `x1`, `y1`, `x2`, `y2` | int | Bounding box. For segmentation, the box around the mask (with a 1 px border). |
| `area` | int | Mask pixel count for segmentation; `width * height` for detection. |
| `ratio` | float | `width / height` (rounded to 1 decimal by the detection code). |
| `indiv` | str / NaN | Linked individual, or empty when no match was found. Empty rows are discarded by the filtering stage. |

### 2.6 Input split files (`split_dir`)

`splits_metadata.csv` records the parameters of every already-computed split.
If a row matches the current split + filter parameters, its split version is
reused instead of recomputing a new split.

| Column | Description |
|---|---|
| `event`, `max_runs`, `train_ratio`, `min_crop_per_label_for_val`, `force_val` | Split parameters. |
| `<label>_conf`, `<label>_min_area`, `<label>_min_ratio`, `<label>_max_ratio`, `<label>_script` | One group of filter columns per detection label. Groups of labels that are not selected are filled with empty values. |
| `version` | Split version (`v1`, `v2`, ...). |

### 2.7 Image preprocessing functions

`preprocessing_function` must name a function of
`scripts/identification/image_preprocessing.py`:

- `original`
- `grayscale`
- `brightness_normalized`
- `contrast_normalized`
- `clahe`
- `histogram_equalized`

Set it to `null` to disable preprocessing.

### 2.8 YAML configuration reference

#### Database and paths

| Key | Default / example | Description |
|---|---|---|
| `dataset_date` | `"21/10/2026"` | Experiment date stored in `all_experiments.csv`. `/` are replaced by `_` when used in paths. |
| `video_database` | `"video_database.csv"` | Video database path, relative to `home_dir`. |
| `frame_database` | `"frame_database.csv"` | Frame database path, relative to `home_dir`. |
| `home_dir` | absolute path | Root of the data directory. |
| `crops_dir` | `"data/processed/crops"` | Root of crops and detection CSVs, relative to `home_dir`. |
| `split_dir` | `"data/processed/crops/split_files"` | Split tables folder, relative to `home_dir`. |
| `split_versions` | `"splits_metadata.csv"` | Split metadata filename inside `split_dir`. |
| `paths.output` | `"data/results"` | Results root, relative to `home_dir`. |
| `paths.videos` | `"data/raw/indiv_videos"` | Informational; detection uses `detection.videos_dir`. |

#### `detection`

| Key | Default | Description |
|---|---|---|
| `videos_dir` | `"data/raw/indiv_videos"` | Folder containing source videos, relative to `home_dir`. |
| `images_dir` | `"data/raw/"` | Kept for compatibility; not used by the detection code. |
| `fps` | `2` | Target inference rate. The stride is `round(video_fps / fps)`: YOLO runs every N frames. |
| `tracker` | `"ByteTrack"` | Tracker name. For YOLO, converted to `<name>.yaml`; for MegaDetector, `ByteTrack` is used through `supervision`. |
| `keep_detections_without_id` | `False` | Save unmatched crops under `<model>/unidentified/<class>/`. |
| `save_videos_with_detections` | `False` | Write an annotated video for videos that contain unmatched detections. |
| `tracked_videos_dir` | `"data/processed/tracked_videos"` | Output folder for annotated videos, relative to `home_dir`. |
| `identification_threshold` | `0.8` | Minimal proportion of the frame-database ear box that must be inside the detection box to link an individual. |

#### `filter`

One block per detection label (`head_detect`, `ear_detect`, `body_detect`,
`ear_seg` in the default file):

| Key | Description |
|---|---|
| `conf` | Minimum detection confidence, or `null` to disable. |
| `min_area` | Minimum `area`, or `null`. |
| `min_ratio` | Minimum width/height ratio, or `null`. |
| `max_ratio` | Maximum width/height ratio, or `null`. |
| `script` | `null`, a script name, or a list of script names from `scripts/identification/detection_filters/`. Each script exposes `main(df, crops_dir, model) -> filtered_df`. Only `seg_cut` is currently provided (for `ear_seg`). |

Filters are applied in this order: drop empty `indiv`, `conf`, `min_area`,
`min_ratio`, `max_ratio`, then filter scripts.

#### `split`

| Key | Default | Description |
|---|---|---|
| `event` | `"day"` | Grouping unit: `day`, `week`, `month`, or a duration such as `5min`, `1h`, `2h`. Events are always per camera. |
| `max_runs` | `2` | Maximum number of split runs. If the number of exhaustive combinations (`2^free_events`) exceeds `max_runs`, the pipeline switches to a greedy split and generates `max_runs` runs. |
| `train_ratio` | `0.75` | Target proportion of train crops. |
| `min_crop_per_label_for_val` | `10` | Labels with fewer crops than this go entirely to train. |
| `force_val` | `True` | If a label appears in exactly two events and is already in train, put the other event in validation. |

Split rules, in priority order:

1. A label that appears in only one event goes entirely to train.
2. A label with fewer than `min_crop_per_label_for_val` crops goes to train.
3. Events that introduce a label not yet present in train go to train.
4. With `force_val`, the second event of a two-event label goes to validation.
5. Remaining events are assigned by minimizing the distance to `train_ratio`.

After splitting, `pipeline.py` keeps **one crop per `(event, label)` pair**
(the last row). The copy written into the results folder is therefore
deduplicated, while the copy written by `split.py` in `split_dir` keeps all
crops.

#### `preprocessing_function` and `data_augmentation`

| Key | Description |
|---|---|
| `preprocessing_function` | Function name from `image_preprocessing.py`, or `null`. |
| `data_augmentation.fold_size` | Declared in the YAML but not used by the current training code. |
| `data_augmentation.script` | Declared in the YAML but not used by the current training code. |
| `data_augmentation.function` | Declared in the YAML but not used by the current training code. |
| `data_augmentation.torch_transformation.affine.degrees` | Rotation range (RandomAffine). |
| `...affine.translate` | Translation `(tx, ty)` or `(tx_min, tx_max, ty_min, ty_max)`. |
| `...affine.scale` | Scale range. |
| `...colorjitter.saturation` | Saturation range. |
| `...colorjitter.hue` | Hue range. |

Only fields whose flag in `AUG_PARAMS` is `True` are applied. Augmentation
starts at epoch `training_parameters.aug_epoch`.

#### `feature_extractor` and `training_parameters`

| Key | Default | Description |
|---|---|---|
| `feature_extractor.name` | `"ResNet50"` | Backbone name. Ignored when `ARCHITECTURE` is not `None`. |
| `feature_extractor.last_layer` | `null` | Last layer before the classification head (output used as features). Used when `ARCHITECTURE = None`. |
| `feature_extractor.last_frozen_layer` | `null` | Last frozen layer; layers between it and `last_layer` are fine-tuned from `defreeze_epoch`. Used when `ARCHITECTURE = None`. |
| `training_parameters.batch_size` | `64` | Batch size. |
| `training_parameters.num_epoch` | `3` | Number of epochs. Training always runs all epochs; the stop epoch is selected afterwards. |
| `training_parameters.lr` | `0.001` | Adam learning rate. |
| `training_parameters.pretrained` | `true` | Load pretrained backbone weights. |
| `training_parameters.aug_epoch` | `2` | First epoch (1-based) where augmentation is applied. |
| `training_parameters.defreeze_epoch` | `3` | First epoch (1-based) where backbone layers are unfrozen. `null` means the backbone stays frozen. |
| `training_parameters.early_stopping` | `1.0` | Minimal relative train-accuracy improvement (in %) between two epochs. If lower, the current epoch is the stop epoch. Early stopping is delayed until at least 5 epochs after defreeze/augmentation. `null` disables it. |
| `training_parameters.num_workers` | `0` | DataLoader workers. |
| `training_parameters.seed` | `42` | Global seed. |
| `training_parameters.batch_norm_only` | `False` | Fine-tune only batch-normalization layers. |
| `training_parameters.classification_head` | `"ClassificationHead"` | Classification head class from `network.py`. |
| `similarity_research.num_points` | `1` | Number of nearest train features used for the cosine majority vote. |

---

## 3. What each stage does

### 3.1 Detection and tracking (`detect_and_track.py`)

For each chosen model:

1. Videos are sorted by camera, then creation time.
2. YOLO tracking runs every `stride` frames (`stride = round(video_fps / fps)`).
3. The tracker is reset when the camera changes or when two consecutive videos
   are more than 5 minutes apart.
4. Each detection is linked to an individual using the video/frame database.
5. Identified crops are written under
   `<crops_dir>/<model>/<detection_class>/`; unmatched crops go to
   `<crops_dir>/<model>/unidentified/<detection_class>/` if
   `keep_detections_without_id` is `True`.
6. Rows are appended to `<crops_dir>/<model>.csv` after each video.
7. The state `{all_detect, some_detect}` is saved in
   `<crops_dir>/<model>/treated_videos.pkl`.

`all_detect` contains videos where every detection was matched; `some_detect`
contains videos with at least one unmatched detection.

When `RERUN_SOME_DETECT` is `True`, `get_partially_treated_video_to_treat()`
re-checks unmatched rows: crops that can now be matched are moved from
`unidentified/` to the identified folder, the CSV is updated, and only videos
whose crop file is missing are processed again.

**Important side effect:** whenever a new identified detection is written, the
whole `split_dir` is deleted (`shutil.rmtree`). All existing split files become
invalid and must be recomputed.

### 3.2 Filtering

For each selected label, the corresponding `<model>.csv` is loaded, rows without
an individual are removed, and the confidence/area/ratio/script filters from the
YAML are applied. A training label is then built:

- `ASSYMETRY = False`: `<indiv>_<body_part>` (left/right merged);
- `ASSYMETRY = True`: `<indiv>_<detection_class>`.

All selected dataframes are concatenated.

### 3.3 Split

Events are assigned to videos using the camera and `time` columns. The split
either enumerates all train/validation combinations (when
`2^free_events <= max_runs`) or uses a deterministic greedy procedure with
`max_runs` seeds. See the rules and parameters in section 2.8.

### 3.4 Training (`train_classifier.py`)

- `pipeline.py` passes the filtered `splits_df` and `images_df` DataFrames
  directly to `train_classifier.run_training()`. There are no `crop_file` or
  `split_file` arguments: the split and image tables are always loaded by the
  pipeline before training.
- Features are produced by the chosen backbone and sent to
  `ClassificationHead`.
- Loss: cross-entropy. Optimizer: Adam. Scheduler: `ReduceLROnPlateau`.
- `FINETUNE` decides what is trained: `True` -> head + the backbone layers
  between `last_frozen_layer` and `last_layer`, unfrozen from `defreeze_epoch`;
  `False` -> fully frozen backbone (whatever `defreeze_epoch`), only the
  `ClassificationHead` trains. `TRAIN_CLASSIFIER = False` skips training
  entirely.
- Augmentation is applied on the fly from `aug_epoch`.
- Each run of `splits_df` is trained independently.
- The stop epoch is chosen by the early-stopping rule; weights, predictions and
  curves correspond to that epoch.
- Weights are saved only when the backbone was actually fine-tuned.

### 3.5 Cosine similarity (`cosine_similarity.py`)

For each run:

1. Load `models/model_run_<run>.pth` if it exists; otherwise use the pretrained
   backbone.
2. Compute train and validation features with the same preprocessing as
   training (no augmentation).
3. Normalize features and find the `num_points` closest train features for each
   validation crop.
4. Predict by majority vote; `conf` is the winning-class vote fraction and
   `margin` the gap between the two best classes.

### 3.6 Experiment summary

One row per pipeline execution is appended to `all_experiments.csv`. The
versioned result folder (`v1`, `v2`, ...) identifies the experiment.

---

## 4. Outputs

### 4.1 Output folder tree

```text
<crops_dir>/
├── <model>.csv                          # detection metadata (input of filtering)
├── <model>/
│   ├── treated_videos.pkl               # {"all_detect": set, "some_detect": set}
│   ├── <detection_class>/<image>.png    # identified crops
│   └── unidentified/<class>/<image>.png # only if keep_detections_without_id
└── split_files/
    ├── splits_metadata.csv              # appended metadata, one row per split
    ├── vN_splits.csv                    # event, run, split
    └── vN_selected_images.csv           # all crops with event, no dedup

<tracked_videos_dir>/<model>/<video_stem>.mp4   # only if save_videos_with_detections

<results root>/                          # paths.output, e.g. data/results
├── all_experiments.csv                  # appended summary, one row per pipeline run
└── vN/                                  # next available version
    ├── input_metadata/
    │   ├── pipeline.yaml                # copy of the configuration used
    │   ├── <split_version>_splits.csv
    │   └── <split_version>_selected_images.csv
    ├── models/
    │   └── model_run_<run>.pth          # only if fine-tuned
    ├── train_curv_run_<run>.png         # per-run loss/accuracy curves
    ├── val_pred_early_stop.csv          # validation predictions at stop epoch
    ├── runs_summary.csv                 # per-run metrics
    ├── val_cosine_similarity_pred.csv   # cosine predictions
    └── cosine_accuracy_summary.csv      # per-run cosine accuracy
```

### 4.2 `crops_dir/<model>.csv`

Same columns as the input detection CSV described in section 2.5:
`image, video, frame, track_id, detection, conf, x1, y1, x2, y2, area, ratio,
indiv`.

Rows are prepended to the existing file after each processed video, then the
whole CSV is rewritten. Existing rows are preserved.

### 4.3 `treated_videos.pkl`

Python pickle dictionary:

| Key | Content |
|---|---|
| `all_detect` | Set of video filenames where all detections were linked to an individual. |
| `some_detect` | Set of video filenames with at least one unmatched detection. |

The file is rewritten after each processed video.

### 4.4 `split_files/splits_metadata.csv`

One appended row per computed split. Columns:
`event, max_runs, train_ratio, min_crop_per_label_for_val, force_val`, then the
filter columns (`<label>_conf`, `<label>_min_area`, `<label>_min_ratio`,
`<label>_max_ratio`, `<label>_script` for every label), then `version`.

If an identical parameter row already exists, the existing split version is
reused and no new split is computed.

### 4.5 `split_files/vN_splits.csv` and results copy

| Column | Description |
|---|---|
| `event` | Event identifier. |
| `run` | Run number (starts at 1). |
| `split` | `train` or `val`. |

The version copied into `<results>/vN/input_metadata/` has the same content.

### 4.6 `split_files/vN_selected_images.csv` and results copy

| Column | Type | Description |
|---|---|---|
| `image` | str | Crop filename. |
| `frame` | int | Original frame number. |
| `track_id` | int | Tracker identifier. |
| `detection` | str | Detection class. |
| `conf` | float | Detection confidence. |
| `x1`, `y1`, `x2`, `y2` | int | Bounding box. |
| `area` | int | Mask area or bounding-box area. |
| `ratio` | float | Width/height ratio. |
| `indiv` | str | Individual. |
| `label` | str | Training label (`indiv_body_part` or `indiv_detection`). |
| `event` | int | Event assigned from the video database. |

The `video` column is replaced by `event`. The file written by `split.py` in
`split_dir` contains all crops; the copy written into the results folder is
deduplicated on `(event, label)` (last row kept).

### 4.7 `runs_summary.csv`

Written by `train_classifier.py` after training and then updated by
`pipeline.py` with cosine accuracy and crop weights.

| Column | Description |
|---|---|
| `run` | Run identifier. |
| `detections` | Comma-separated detection classes used in the run. |
| `num_classes` | Number of training labels. |
| `num_train` | Number of train crops. |
| `num_val` | Number of validation crops. |
| `train_acc` | Train accuracy (%) at the stop epoch. |
| `val_acc` | Validation accuracy (%) at the stop epoch. |
| `best_val_acc` | Best validation accuracy (%) over epochs. |
| `best_epoch` | Epoch (1-based) of `best_val_acc`. |
| `train_loss` | Train cross-entropy error at the stop epoch. |
| `val_loss` | Validation cross-entropy error at the stop epoch. |
| `early_stop_epoch` | Stop epoch (1-based). |
| `duration_s` | Run duration in seconds. |
| `cosine_acc` | Cosine-similarity validation accuracy (%) added by `pipeline.py`. |
| `train_weights` | Share of the total train crops represented by this run. |
| `val_weights` | Share of the total validation crops represented by this run. |

When no fine-tuning is performed, the training metrics are left empty (NaN)
and only split sizes, cosine accuracy and weights are meaningful.

### 4.8 `val_pred_early_stop.csv`

Validation predictions at the stop epoch, appended after each run.

| Column | Description |
|---|---|
| `run` | Run identifier. |
| `model_name` | Detection model folder (e.g. `head_ears_detect_09_08`). |
| `detection` | Detection class. |
| `image` | Crop filename. |
| `video` | Empty in practice: the selected-images table replaces `video` by `event`. Use `event` from the split files instead. |
| `indiv` | True individual. |
| `label` | True training label. |
| `pred_label` | Predicted label. |
| `margin` | Gap between the two highest class probabilities. |
| `logits` | Raw class scores (Python list). |
| `conf` | Probability of the predicted class. |

### 4.9 `val_cosine_similarity_pred.csv`

Cosine-similarity predictions, appended per run. The first columns are the same
as `val_pred_early_stop.csv`, with `conf` and `margin` computed from the
majority vote. Additional columns come from the selected-images row and the
neighbour search:

| Column | Description |
|---|---|
| `frame`, `track_id`, `x1`, `y1`, `x2`, `y2`, `area`, `ratio` | Crop metadata. |
| `event` | Event identifier. |
| `split` | `val` (validation crops only). |
| `label_1` ... `label_k` | Labels of the `k` nearest train crops. |
| `sim_1` ... `sim_k` | Cosine similarities of these neighbours. |

`k` is `similarity_research.num_points` (capped by the number of train crops).
`logits` here contains the per-class vote fractions.

### 4.10 `cosine_accuracy_summary.csv`

| Column | Description |
|---|---|
| `run` | Run identifier. |
| `cosine_acc` | Validation accuracy (%) of the cosine majority vote. |

### 4.11 `all_experiments.csv`

One row appended per pipeline execution, sorted by `version`.

| Column | Description |
|---|---|
| `version` | Result folder version (`v1`, `v2`, ...). |
| `dataset_date` | `dataset_date` from the YAML. |
| `detections` | List of selected detections (stored as a string). |
| `mode` | List of selected modes (stored as a string). |
| `architecture` | Feature extractor name. |
| `finetune` | `FINETUNE` switch value of the execution (`True` = backbone fine-tuned, `False` = head only). |
| `data_augmentation` | Summary string such as `RandomAffine+ColorJitter_from_epoch_2`, or `none`. |
| `split_event` | Split event unit. |
| `last_fozen_layer` | Last frozen layer (column name keeps the historical typo). |
| `last_layer` | Last feature layer. |
| `mean_train_acc_over_runs` | Aggregated train accuracy over runs. |
| `train_acc_std` | Standard deviation of train accuracy over runs. |
| `mean_val_acc_over_run` | Aggregated validation accuracy over runs. |
| `val_acc_std` | Standard deviation of validation accuracy over runs. |
| `mean_cosine_acc_over_run` | Aggregated cosine accuracy over runs. |
| `val_cosine_acc_std` | Standard deviation of cosine accuracy over runs. |

### 4.12 `models/model_run_<run>.pth`

Checkpoint saved only when the backbone was fine-tuned:

| Key | Content |
|---|---|
| `backbone_state_dict` | Fine-tuned backbone weights. |
| `head_state_dict` | Classification head weights. |
| `run` | Run identifier. |
| `class_names` | Ordered label list. |
| `detections` | Detection classes used. |
| `feature_extractor` | Backbone name. |
| `last_layer`, `last_frozen_layer` | Layer names. |
| `num_epoch`, `stop_epoch`, `early_stopping` | Training settings. |
| `is_finetuned` | Whether the backbone was fine-tuned. |
| `val_accuracy`, `train_accuracy` | Metrics at the stop epoch. |

---

## 5. File writing, appending and deletion rules

### Created

- A new results version folder `<results>/vN` on every pipeline execution.
- `<results>/vN/input_metadata/pipeline.yaml` (copy of the configuration).
- `<results>/vN/input_metadata/<split_version>_{splits,selected_images}.csv`.
- `<results>/all_experiments.csv` on the first execution.
- Crop folders and `unidentified` folders for each kept detection class.
- Split files `vN_splits.csv` and `vN_selected_images.csv` in `split_dir`.

### Appended / rewritten

- `<crops_dir>/<model>.csv`: new rows are prepended, then the whole file is
  rewritten after every processed video.
- `<crops_dir>/<model>/treated_videos.pkl`: rewritten after every processed
  video.
- `split_dir/splits_metadata.csv`: new split metadata rows are appended.
- `val_pred_early_stop.csv`: deleted at the start of each training execution,
  then appended after each run.
- `val_cosine_similarity_pred.csv`: appended after each run. Because a new
  `vN` folder is created for each pipeline execution, normal usage does not
  create duplicates.
- `runs_summary.csv`: written by training, then overwritten by `pipeline.py`
  when cosine results are merged.
- `cosine_accuracy_summary.csv`: overwritten.
- `all_experiments.csv`: one row appended per execution, then the file is
  rewritten sorted by `version`.

### Deleted

- `next_results_dir()` removes result folders `vN` that contain no file other
  than `.yaml` (empty/incomplete experiments).
- `detect_and_track.py` deletes the entire `split_dir` (`shutil.rmtree`) as
  soon as a new identified detection is written. Existing splits are therefore
  invalidated.
- `train_classifier.py` deletes `val_pred_early_stop.csv` before starting a new
  training execution.
- With `RERUN_SOME_DETECT = True`, crops that can newly be matched are moved
  (renamed, not deleted) from `unidentified/` to the identified folder.

No script deletes non-empty result version folders, source videos, crop images
outside the `unidentified` move, or `all_experiments.csv` rows.

---

## 6. Practical notes and limitations

- `--dry-run` is parsed but currently ignored.
- With `FINETUNE = False` the training metrics (`train_acc`, `val_acc`, ...) of
  `runs_summary.csv` refer to the classification head only; the backbone weights
  are never saved, so `cosine_similarity.py` uses the pretrained backbone.
- `ASSYMETRY = False` merges left/right ears into a single label. Set it to
  `True` if left and right ears must be treated as different classes.
- Detection labels must contain `detect` or `seg` so that the YOLO processing
  mode can be inferred.
- New detection models can be added by placing a compatible `.pt` file in
  `models/` and adding its key, confidence and labels to `DETECT_PARAMS` in
  `pipeline.py`. Other model families require a new `process_video_*` function
  and an entry in the processing-mode dispatch.
- Filter scripts must be stored in
  `scripts/identification/detection_filters/` and expose
  `main(df, crops_dir, model) -> pd.DataFrame`.
