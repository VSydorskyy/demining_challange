import os
import pandas as pd
import yaml
import json

from glob import glob
from shutil import copyfile

import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold

def convert_to_yolo_coco8_format(df: pd.DataFrame):
    """
    Converts a detection label dataframe into YOLO COCO8 format.
    
    Parameters:
    - df: DataFrame with columns [image_id,label,x,y,width,height,image_width,image_height]
    """
    label_dict = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting to COCO8 format"):
        if row['label'] == -1:
             label_dict[row['image_id']] = None
        else:
            x_center = (row["x"] + row["width"] / 2) / row["image_width"]
            y_center = (row["y"] + row["height"] / 2) / row["image_height"]
            width_norm = row["width"] / row["image_width"]
            height_norm = row["height"] / row["image_height"]
    
            label_line = f"{int(row['label'])} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            label_dict[row['image_id']].append(label_line)

    return label_dict

def list_to_files(input, filename):
    """
    Takes a list of filenames and makes a text file of filenames
    """
    with open(filename, "w") as the_file:
        for idx, f in enumerate(input):
            if idx == len(input) - 1:
                the_file.write(f)
            else:
                the_file.write(f + "\n")

def save_anotation_file(path, content):
    if content is None:
        Path(path).touch()
    else:
        list_to_files(content, path)


def main(image_root, annotation_root, save_root, n_folds, remove_empty_train):
    all_annotation_paths = glob(
        os.path.join(annotation_root, "*.csv")
    )
    all_image_paths = glob(
        os.path.join(image_root, "*.jpg")
    )
    full_annotation_df = pd.concat([pd.read_csv(path) for path in all_annotation_paths])
    empty_images = (
        set([os.path.splitext(os.path.basename(el))[0] for el in all_image_paths]) -
        set(full_annotation_df["image_id"])
    )
    full_annotation_df = pd.concat([
        full_annotation_df,
        pd.DataFrame({
            "image_id": list(empty_images),
            "label": [-1] * len(empty_images)
        })
    ]).reset_index(drop=True)

    coco8_df_format = convert_to_yolo_coco8_format(full_annotation_df)

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    split = []
    for train_idx, val_idx in skf.split(
        full_annotation_df, 
        full_annotation_df["label"].astype("category"), 
        full_annotation_df["image_id"].astype("category")
    ):
        split.append({
            "train": list(set(full_annotation_df["image_id"].iloc[train_idx])),
            "val": list(set(full_annotation_df["image_id"].iloc[val_idx]))
        })

    os.makedirs(save_root, exist_ok=True)

    for fold_id, fold_dict in enumerate(split):
        train_images = fold_dict["train"]
        val_images = fold_dict["val"]

        save_root_image_train = os.path.join(save_root, f"fold_{fold_id}", "images", "train")
        save_root_image_val = os.path.join(save_root, f"fold_{fold_id}", "images", "val")
        save_root_annotations_train = os.path.join(save_root, f"fold_{fold_id}", "labels", "train")
        save_root_annotations_val = os.path.join(save_root, f"fold_{fold_id}", "labels", "val")
        os.makedirs(save_root_image_train, exist_ok=True)
        os.makedirs(save_root_image_val, exist_ok=True)
        os.makedirs(save_root_annotations_train, exist_ok=True)
        os.makedirs(save_root_annotations_val, exist_ok=True)

        for image_id, label in tqdm(coco8_df_format.items(), desc=f"Processing fold {fold_id}"):
            if image_id in train_images:
                if remove_empty_train and label is None:
                    continue
                copyfile(
                    os.path.join(image_root, image_id + ".jpg"),
                    os.path.join(save_root_image_train, image_id + ".jpg")
                )
                save_anotation_file(
                    os.path.join(save_root_annotations_train, image_id + ".txt"),
                    label
                )
            elif image_id in val_images:
                copyfile(
                    os.path.join(image_root, image_id + ".jpg"),
                    os.path.join(save_root_image_val, image_id + ".jpg")
                )
                save_anotation_file(
                    os.path.join(save_root_annotations_val, image_id + ".txt"),
                    label
                )
            else:
                raise RuntimeError(f"{image_id} not in train and val sets!!!")

        with open(os.path.join(save_root, f"fold_{fold_id}", "coco8.yaml"), "w") as f:
            yaml.dump({
                "path": os.path.join(save_root, f"fold_{fold_id}"),
                "train": "images/train",
                "val": "images/val",
                "names": {
                    0: "Explosives",
                    1: "Anti-personnel_mine",
                    2: "Anti-vehicle_mine"
                }
            }, f, default_flow_style=False, sort_keys=False)

    # Save split at the very end to save_root in .json format
    with open(os.path.join(save_root, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data and convert into COCO8 format")
    parser.add_argument("--image_root", type=str, required=True, help="Path to images directory")
    parser.add_argument("--annotation_root", type=str, required=True, help="Path to annotations directory")
    parser.add_argument("--save_root", type=str, required=True, help="Path to save output directory")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--remove_empty_train", action="store_true", help="Remove images without instances from train split")
    args = parser.parse_args()
    main(args.image_root, args.annotation_root, args.save_root, args.n_folds, args.remove_empty_train)