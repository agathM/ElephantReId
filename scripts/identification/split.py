"""
Configurable "event" based split script.

Goal:
- Take a dict as input:
  {'event': 'day' | 'week' | 'month' | '5min' | '5h', 'max_runs': ..., 'images_df': ..., 'database': ...}
- Assign an event to each video of `database` based on `event` and the `time` column event is a duration groups video when succesive creation time is less than duration
- Link via `images_df` (columns `video`, `label`) all labels present per event
- Forcing rules by order of priority:
    -at least one event in train per label,
    - if total crops of on label < min_crop_per_label_for_val all events goes to train
    - if force_val at least one event in val for labels appearing in two events,

- Enumerate all possible train/val assignment combinations for non-forced events and export a CSV with columns: event, run, split without taking into account force_val
- If more events than max_run generate max_run splits
- Returns images_df with split column added
"""
import pdb
import itertools
import re
from pathlib import Path
from typing import Any, Dict, Iterable
import pdb
import numpy as np
import pandas as pd

# Greedy split configuration
DEFAULT_max_runs = 100

def next_split_version(split_versions: Path, row: dict) -> tuple[str, Path] | None:
    """Generate the next split version and append its metadata."""

    if not row:
        return None

    split_versions.parent.mkdir(parents=True, exist_ok=True)

    versions = [
        int(file.stem[1:])
        for file in split_versions.parent.glob("v*.csv")
        if file.stem[1:].isdigit()
    ]

    version = f"v{max(versions) + 1 if versions else 1}"

    split_file = split_versions

    row["version"] = version

    pd.DataFrame([row]).to_csv(
        split_file,
        mode="a",
        header=not split_file.exists(),
        index=False,
    )

    return version


def _assign_event_by_period(clips_dataframe: pd.DataFrame, period: str) -> pd.Series:
    """
    Assign events to dataframe
     one event is one calendar period (day, week or month) and one camera
    """
    df = clips_dataframe.copy()
    df = df.sort_values(["camera", "time"], na_position="last").reset_index(drop=True)

    event_period = np.full(len(df), np.nan)
    group = 0
    for camera in df["camera"].unique():
        mask = (df["camera"] == camera) & df["time"].notna()
        idx = df.index[mask].to_numpy()
        if len(idx) == 0:
            continue
        periods = df.loc[idx, "time"].dt.to_period(period).to_numpy()
        if len(periods) == 0:
            continue
        event_period[idx[0]] = group
        for i in range(1, len(periods)):
            if periods[i] != periods[i - 1]:
                group += 1
            event_period[idx[i]] = group
        group += 1

    return pd.array(event_period, dtype="Int64")


def _assign_event_by_timedelta(clips_dataframe: pd.DataFrame, delta: pd.Timedelta) -> pd.Series:
    """
    Assign events to dataframe
     one event is a camera and a group of videos ordered by chronological times that are less spaced in creation time than delta
    """
    df = clips_dataframe.copy()
    df = df.sort_values(["camera", "time"], na_position="last").reset_index(drop=True)

    entry = np.full(len(df), np.nan)
    group = 0
    for camera in df["camera"].unique():
        mask = (df["camera"] == camera) & df["time"].notna()
        idx = df.index[mask].to_numpy()
        if len(idx) == 0:
            continue
        times = df.loc[idx, "time"].to_numpy()
        if len(times) == 0:
            continue
        entry[idx[0]] = group
        for i in range(1, len(times)):
            if (times[i] - times[i - 1]) >= delta.to_numpy():
                group += 1
            entry[idx[i]] = group
    return pd.array(entry, dtype="Int64")


def assign_events_to_videos(database: pd.DataFrame, event: str) -> tuple[pd.DataFrame, str]:
    """
    Assign event to dataframe
    """
    df = database.copy()

    # Normalize time
    if "time" not in df.columns:
        raise ValueError("The `database` must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Normalize required columns
    for col in ["camera", "video"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'.")

    df = df.sort_values(["camera", "time"], na_position="last").reset_index(drop=True)

    event_key = str(event).strip().lower()
    if event_key in {"day", "week", "month"}:
        period_map = {"day": "D", "week": "W", "month": "M"}
        event_series = _assign_event_by_period(df, period_map[event_key])
    else:
        m = re.fullmatch(r"(\d+)\s*(min|h|minute|minutes|hour|hours)", event_key)
        if not m:
            raise ValueError("`event` must be 'day', 'week', 'month' or a duration like '5min' or '2h'")
        val = int(m.group(1))
        unit = m.group(2)
        if unit in ["min", "minute", "minutes"]:
            delta = pd.Timedelta(minutes=val)
        elif unit in ["h", "hour", "hours"]:
            delta = pd.Timedelta(hours=val)
        else:
            raise ValueError("Unsupported unit")
        event_series = _assign_event_by_timedelta(df, delta)

    df["event"] = event_series
    # Keep only rows with a valid event
    df = df[df["event"].notna()].copy()
    df["event"] = df["event"].astype("Int64")
    return df


def get_labels_per_event(videos_df: pd.DataFrame, img: str | Path):
    """
    Read images_df (columns video, label) and link all present labels to each event.
    """
    # merge to link label to event
    img.dropna(subset=['video',"label"], inplace=True)
    merged = videos_df[["video", "event"]].drop_duplicates().merge(img[["image","video","label"]], on="video", how="inner")
    merged = merged.dropna(subset=["event"])

    for col in ["video", "label"]:
        if col not in img.columns:
            raise ValueError(f"images_df must contain the 'video' and 'label' columns (found columns: {list(img.columns)})")

    img = img[["video", "label"]].dropna()
    img["label"] = img["label"].astype(str)



    # unique labels per event
    labels_per_event = (
        merged.groupby("event")["label"]
        .apply(lambda s: sorted(set(s)))
        .reset_index(name="labels_list")
    )
    # reverse mapping
    label_to_events = (
        merged.drop_duplicates(["event", "label"])[["label", "event"]]
        .groupby("label")["event"]
        .apply(lambda s: sorted(set(s)))
        .reset_index(name="events_list")
    )
    return labels_per_event, label_to_events, merged


def apply_manual_split_rules(
    videos_df: pd.DataFrame,
    label_to_events: pd.DataFrame,
    crops_count_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply EXACTLY the same rules as manual_split.py but to the chosen event:
    - label with nb_event == 1 -> all its events go to 'train'
    - then labels with nb_crop_per_label < min_crop_per_label_for_val -> all their remaining events go to 'train'
    """
    df = videos_df[["event", "video"]].drop_duplicates().copy()

    # number of events per label
    label_nb_events = label_to_events.copy()
    label_nb_events["nb_event"] = label_nb_events["events_list"].apply(len)
    label_1_event = label_nb_events[label_nb_events["nb_event"] == 1]["label"].tolist()

    # init empty split
    df["split"] = ""

    # rule 1: label with only one event
    events_for_label1 = set()
    for _, row in label_to_events.iterrows():
        label = row["label"]
        if label in label_1_event:
            for ev in row["events_list"]:
                events_for_label1.add(ev)
    df.loc[df["event"].isin(events_for_label1), "split"] = "train"

    # rule 2: number of crops per label < min_crop_per_label_for_val
    if crops_count_df is not None and not crops_count_df.empty:
        nb_crop_per_label = crops_count_df.copy()
        if "label" in nb_crop_per_label.columns and "nb_crop" in nb_crop_per_label.columns:
            label_under_thresh = nb_crop_per_label[nb_crop_per_label["nb_crop"] < min_crop_per_label_for_val]["label"].tolist()
            events_under_thresh = set()
            for _, row in label_to_events.iterrows():
                label = row["label"]
                if label in label_under_thresh:
                    for ev in row["events_list"]:
                        events_under_thresh.add(ev)
            mask = (df["split"] == "") & (df["event"].isin(events_under_thresh))
            df.loc[mask, "split"] = "train"
    else:
        print("[WARN] No 'crops' count provided. Only the 'label with a single event' rule is strictly applied here. "
              "If you have a crops df (video/label/crop_id) you can pass it via crops_count_df.")

    return df


def generate_combinations(event_split_df: pd.DataFrame, max_runs: int | None = None) -> pd.DataFrame:
    """
    Events already set to train in split are 'forced'.
    The other events can be assigned either to train or val.
    Compute all possible combinations and export with columns: event, run, split
    """
    df = event_split_df.copy()
    events_all = sorted(df["event"].dropna().unique().tolist())
    forced_train = set(df[df["split"] == "train"]["event"].tolist())
    free_events = [e for e in events_all if e not in forced_train]

    print(f"[INFO] Total events: {len(events_all)}")
    print(f"[INFO] Events forced to train: {len(forced_train)}")
    print(f"[INFO] Free events (train/val): {len(free_events)}")

    if len(free_events) == 0:
        print("[INFO] No free event, only one combination is possible.")
        combs = [[]]
    else:
        total_combs = 2 ** len(free_events)
        print(f"[INFO] Number of possible combinations (2^{len(free_events)}) = {total_combs}")
        if max_runs is not None and total_combs > int(max_runs):
            return None

        combs = list(itertools.product([True, False], repeat=len(free_events)))

    rows = []
    for run_id, comb in enumerate(combs):
        for ev in forced_train:
            rows.append({"event": ev, "run": run_id, "split": "train"})
        for is_train, ev in zip(comb, free_events):
            if is_train:
                rows.append({"event": ev, "run": run_id, "split": "train"})
            else:
                rows.append({"event": ev, "run": run_id, "split": "val"})
    out = pd.DataFrame(rows)
    out = out.sort_values(["run", "event"]).reset_index(drop=True)
    return out


def greedy_split(
    df: pd.DataFrame,
    max_runs: int | None = None,
    force_val: bool = True,
    forced_train_events: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """
    Generate greedy train/val splits from a dataframe with columns
    ``label``, ``event`` and ``image``.

    The procedure is:
    - labeliduals appearing in a single event -> all their events go to train;
    - labeliduals appearing in exactly two events are tracked;
    - for each seed from 1 to ``max_runs``, the non-forced events are shuffled;
    - events are processed one by one:
        * an event is put in train if it contains an labelidual with no image
          in train yet (this guarantees at least one event per labelidual);
        * an event is put in val if it contains an labelidual appearing in
          exactly two events that is already present in train;
        * otherwise, the event is assigned to train or val by comparing the
          resulting distance to ``train_ratio`` for every
          labelidual, using the number of images the event adds for each
          labelidual (``entry_n``).

    Returns a dataframe with columns ``event``, ``run`` and ``split``.
    """
    required_cols = {"label", "event", "image"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"greedy_split requires columns {sorted(required_cols)}; "
            f"missing: {sorted(missing_cols)}"
        )

    if max_runs is None:
        max_runs = DEFAULT_max_runs
    max_runs = int(max_runs)
    if max_runs < 1:
        raise ValueError("max_runs must be >= 1")

    data = df[["label", "event", "image"]].dropna(subset=["label", "event"]).copy()
    data["label"] = data["label"].astype(str)

    # Distinct images per (event, label) and per label
    event_label_counts = (
        data.groupby(["event", "label"])["image"]
        .nunique()
        .astype(int)
    )
    total_image_counts = (
        data.groupby("label")["image"]
        .nunique()
        .astype(int)
    )

    # Number of events per labelidual
    events_per_label = data.groupby("label")["event"].nunique()
    single_event_labels = set(events_per_label[events_per_label == 1].index)
    two_event_labels = set(events_per_label[events_per_label == 2].index)

    # Map each event to the image count of each labelidual
    event_to_label_counts: dict[Any, dict[str, int]] = {}
    for (event, label), n_images in event_label_counts.items():
        event_to_label_counts.setdefault(event, {})[label] = int(n_images)

    all_events = data["event"].drop_duplicates().tolist()

    # Events containing at least one single-event labelidual are forced to train
    forced_train = set(
        data.loc[data["label"].isin(single_event_labels), "event"].unique().tolist()
    )
    if forced_train_events is not None:
        forced_train.update(pd.Series(list(forced_train_events)).dropna().tolist())
    forced_train &= set(all_events)

    free_events = [event for event in all_events if event not in forced_train]

    print(f"[INFO] Total events: {len(all_events)}")
    print(f"[INFO] Events forced to train: {len(forced_train)}")
    print(f"[INFO] Free events (train/val): {len(free_events)}")
    print(f"[INFO] Target train proportion: {train_ratio:.0%}")

    rows: list[dict[str, Any]] = []
    free_events_series = pd.Series(free_events, dtype=object)

    for seed in range(1, max_runs + 1):
        shuffled_events = free_events_series.sample(
            frac=1,
            random_state=seed,
        ).tolist()

        split_by_event: dict[Any, str] = {}
        train_image_counts = {
            label: 0
            for label in total_image_counts.index
        }

        # Forced train events are processed before the shuffled free events
        for event in forced_train:
            split_by_event[event] = "train"
            for label, n_images in event_to_label_counts.get(event, {}).items():
                train_image_counts[label] = train_image_counts.get(label, 0) + n_images

        for event in shuffled_events:
            label_counts = event_to_label_counts.get(event, {})
            labeliduals = list(label_counts)

            # Priority 1: an labelidual with no image in train yet
            force_train = any(
                train_image_counts.get(label, 0) == 0
                for label in labeliduals
            )

            if force_train:
                split = "train"
            # Priority 2: a two-event labelidual already present in train
            elif force_val and any(
                label in two_event_labels
                and train_image_counts.get(label, 0) > 0
                for label in labeliduals
            ):
                split = "val"
            else:
                train_score = 0.0
                val_score = 0.0

                for label in labeliduals:
                    current_train = train_image_counts.get(label, 0)
                    total = int(total_image_counts.get(label, 0))
                    if total <= 0:
                        continue

                    entry_n = int(label_counts[label])

                    train_proportion = (current_train + entry_n) / total
                    val_proportion = current_train / total

                    train_score += abs(
                        train_proportion
                        - train_ratio
                    )
                    val_score += abs(
                        val_proportion
                        - train_ratio
                    )

                if train_score <= val_score:
                    split = "train"
                else:
                    split = "val"

            split_by_event[event] = split

            if split == "train":
                for label, n_images in label_counts.items():
                    train_image_counts[label] = (
                        train_image_counts.get(label, 0) + n_images
                    )

        for event in all_events:
            rows.append({
                "event": event,
                "run": seed,
                "split": split_by_event[event],
            })

    out = pd.DataFrame(rows)
    out = out.sort_values(["run", "event"]).reset_index(drop=True)
    return out


def main(params: Dict[str, Any]) -> None:
    """
    Main function taking a dictionary with:
    output_file, event, max_runs, images_df, database
    """
    global train_ratio, min_crop_per_label_for_val
    required = ["max_runs", "event", "images_df", "video_database","min_crop_per_label_for_val"]
    for k in required:
        if k not in params:
            raise ValueError(f"Missing parameter in dict: {k}")
    event = params["event"]
    max_runs = params.get("max_runs")
    images_df = params["images_df"]
    videos_with_detections = images_df.video.unique()
    train_ratio = params["train_ratio"]
    min_crop_per_label_for_val = params["min_crop_per_label_for_val"]
    split_versions = params["split_versions"]
    split_metadata = params["metadata"]
    version = next_split_version(split_versions,split_metadata)

    force_val = params.get("force_val")
    crops_count_df = images_df.groupby(['label'])['image'].nunique().reset_index(name='nb_crops')

    # Load database
    database = params["video_database"]
    database = database[database.video.isin(videos_with_detections)]

    print(f"[INFO] Database with detections loaded: {database.shape}")

    # 1) Assign an event to each video
    videos_with_event = assign_events_to_videos(database, event)
    print(f"[INFO] Number of videos with event: {len(videos_with_event)}")

    # 2) Link labels per event via images_df
    labels_per_event, label_to_events, merged_iev = get_labels_per_event(videos_with_event, images_df)
    print(f"[INFO] Number of events with at least one label: {len(labels_per_event)}")
    print(f"[INFO] Number of labels involved: {len(label_to_events)}")

    # 3) Apply manual_split rules
    event_split_forced = apply_manual_split_rules(videos_with_event, label_to_events, crops_count_df)
    print(f"[INFO] Forced split distribution:\n{event_split_forced['split'].value_counts(dropna=False)}")

    # 4) Generate all train/val combinations, or use greedy splits if there are too many
    splits_df = generate_combinations(event_split_forced, max_runs=max_runs)
    if splits_df is None:
        # greedy_split expects columns label, event, image
        greedy_df = (
            merged_iev[["image", "event", "label"]]
        )
        forced_events = event_split_forced.loc[
            event_split_forced["split"] == "train",
            "event",
        ]
        splits_df = greedy_split(
            greedy_df,
            max_runs=max_runs,
            forced_train_events=forced_events,
            force_val=force_val
        )

    images_df = images_df.merge(videos_with_event[["video","event"]],on="video", how="inner").drop(columns="video")
    images_df.to_csv(split_versions.parent / f"{version}_selected_images.csv", index=False)
    splits_df.to_csv(
        split_versions.parent / f"{version}_splits.csv",
        index=False)
    return splits_df, images_df, version
