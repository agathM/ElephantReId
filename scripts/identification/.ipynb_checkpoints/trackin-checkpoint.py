"""
TracIn-CP influence analysis for elephant individual identification.

This script:
- computes TracIn-CP scores using classifier checkpoints
- works with a frozen backbone + trainable FC classifier
- filters validation crops using confidence/margin/prediction statistics
- supports correct/wrong prediction selection
- optionally aggregates influence by video
- saves the full crop-level influence matrix:
      validation_selected_crop x training_crop
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = "tracin_config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


DEVICE = cfg.get("device", "cuda")

TOP_K = cfg.get("top_k", 3)



# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()



# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, checkpoint):

    model.fc.load_state_dict(
        checkpoint["model_state"]
    )



# ============================================================
# COMPUTE GRADIENT OF ONE SAMPLE
# ============================================================

def compute_gradient(model, feature, label):

    model.zero_grad()

    feature = feature.to(DEVICE)
    label = label.to(DEVICE)


    output = model.fc(
        feature.unsqueeze(0)
    )


    loss = criterion(
        output,
        label.unsqueeze(0)
    )


    loss.backward()


    gradients = []

    for param in model.fc.parameters():

        gradients.append(
            param.grad.detach().flatten()
        )


    return torch.cat(gradients).cpu()



# ============================================================
# CHECKPOINT SELECTION
# ============================================================

def select_checkpoints(checkpoints):

    ids = np.linspace(
        cfg["start_cp"],
        min(cfg["end_cp"], len(checkpoints)-1),
        cfg["n_cp"],
        dtype=int
    )


    return [
        checkpoints[i]
        for i in ids
    ]



# ============================================================
# TRAIN GRADIENTS
# ============================================================

def compute_train_gradients(
        model,
        train_dataset,
        checkpoint):


    load_checkpoint(
        model,
        checkpoint
    )


    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False
    )


    gradients = []


    for feat,label,_ in loader:


        g = compute_gradient(
            model,
            feat[0],
            label[0]
        )


        gradients.append(g)



    return torch.stack(gradients) #concatenate tensors



# ============================================================
# FILTER VALIDATION CROPS
# ============================================================

def select_validation_indices(
        predictions,
        labels,
        outputs,
        id_to_label):

    selected = []

    allowed_individuals = cfg.get("trackin_individuals", [])

    for i, (pred, gt, out) in enumerate(
            zip(predictions, labels, outputs)
    ):

        pred = pred.item()
        gt = gt.item()

        # --------------------------------------------------
        # Filter by ground-truth individual
        # --------------------------------------------------

        if allowed_individuals:
            gt_name = id_to_label[gt]

            if gt_name not in allowed_individuals:
                continue

        probs = torch.softmax(out, dim=0)

        confidence = probs[pred].item()

        top2 = torch.topk(probs, 2).values

        margin = (top2[0] - top2[1]).item()

        # correct / wrong filtering

        if cfg["correct_prediction_only"]:
            if pred != gt:
                continue

        if cfg["wrong_prediction_only"]:
            if pred == gt:
                continue

        # confidence filtering

        if confidence < cfg["conf_thresh"]:
            continue

        if margin < cfg["margin_thresh"]:
            continue

        selected.append(i)

    return selected



# ============================================================
# PREDICTION STATISTICS
# ============================================================

def compute_prediction_stats(
        predictions,
        labels,
        gt):


    mask = labels == gt


    preds = predictions[mask]


    counts = torch.bincount(
        preds
    )


    return counts



# ============================================================
# VIDEO AGGREGATION
# ============================================================

def aggregate_video_influence(
        influence,
        train_metadata):


    df=train_metadata.copy()


    df["score"]=(
        influence.numpy()
    )


    video_scores = (
        df.groupby("video_name")
        ["score"]
        .mean()
        .sort_values(
            ascending=False
        )
    )


    video_crops = (
        df.groupby("video_name")
        [["crop_id", "score"]]
        .apply(lambda x: x.to_dict("records"))
    )


    return video_scores, video_crops



# ============================================================
# MAIN TRACIN-CP
# ============================================================

def run_tracin_cp(
        model,
        train_dataset,
        val_dataset,
        checkpoints,
        train_metadata,
        predictions,
        labels,
        outputs):


    checkpoints = select_checkpoints(
        checkpoints
    )


    print(
        "Number of checkpoints:",
        len(checkpoints)
    )



    # --------------------------------------------------------
    # Compute training gradients
    # --------------------------------------------------------

    train_gradients=[]


    for cp in checkpoints:

        print(
            "Checkpoint:",
            cp["epoch"]
        )


        train_gradients.append(
            compute_train_gradients(
                model,
                train_dataset,
                cp
            )
        )



    # --------------------------------------------------------
    # Select validation crops
    # --------------------------------------------------------

    selected_val_indices = select_validation_indices(
        predictions,
        labels,
        outputs,
        id_to_label={v: k for k, v in label_map.items()}
    )


    print(
        "Selected validation crops:",
        len(selected_val_indices)
    )



    influence_matrix=[]

    results=[]



    # --------------------------------------------------------
    # Compute influence
    # --------------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )


    for val_idx in selected_val_indices:


        feat,gt,_ = val_dataset[val_idx]


        pred = predictions[val_idx].item()


        gt = gt.item()



        influence=torch.zeros(
            len(train_dataset)
        )



        for cp_grad,cp in zip(
            train_gradients,
            checkpoints):


            load_checkpoint(
                model,
                cp
            )


            val_grad = compute_gradient(
                model,
                feat,
                torch.tensor(gt)
            )



            influence += (
                cp["lr"]
                *
                (cp_grad @ val_grad)
            )



        # Save crop level matrix

        influence_matrix.append(
            influence
        )



        # ----------------------------------------------------
        # proponents / opponents
        # ----------------------------------------------------

        row={

            "val_index":val_idx,
            "gt":gt,
            "pred":pred

        }



        if cfg["aggregate_by_video"]:


            video_scores,video_crops = (
                aggregate_video_influence(
                    influence,
                    train_metadata
                )
            )


            best = video_scores.head(
                TOP_K
            )


            worst = video_scores.tail(
                TOP_K
            )


            for k,(video,score) in enumerate(best.items()):

                row[f"proponent_{k+1}_video"]=video

                row[f"proponent_{k+1}_score"]=score



            for k,(video,score) in enumerate(worst.items()):

                row[f"opponent_{k+1}_video"]=video

                row[f"opponent_{k+1}_score"]=score




        else:


            values,ids=torch.topk(
                influence,
                TOP_K
            )


            neg_values,neg_ids=torch.topk(
                -influence,
                TOP_K
            )


            for k in range(TOP_K):

                row[f"proponent_{k+1}_idx"]=ids[k].item()

                row[f"proponent_{k+1}_score"]=values[k].item()


                row[f"opponent_{k+1}_idx"]=neg_ids[k].item()

                row[f"opponent_{k+1}_score"]=-neg_values[k].item()



        results.append(row)



    # --------------------------------------------------------
    # Save influence matrix
    # --------------------------------------------------------

    influence_matrix=torch.stack(
        influence_matrix
    )


    if cfg["save_influence_matrix"]:


        torch.save(
            {
                "scores":influence_matrix,
                "val_indices":selected_val_indices,
                "train_size":len(train_dataset)

            },
            cfg["influence_matrix_path"]
        )


    return pd.DataFrame(results)



# ============================================================
# RUN
# ============================================================

tracin_df = run_tracin_cp(
    model=base_model,
    train_dataset=train_features_dataset,
    val_dataset=val_features_dataset,
    checkpoints=checkpoints,
    train_metadata=train_metadata,
    predictions=torch.cat(all_predicted),
    labels=torch.cat(all_labels),
    outputs=torch.cat(all_outputs)
)



tracin_df.to_csv(
    cfg["output_csv"],
    index=False
)


print(
    "Saved:",
    cfg["output_csv"]
)