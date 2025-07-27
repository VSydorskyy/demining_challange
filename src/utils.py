import os
import pandas as pd
from glob import glob

def create_gt_df(
    image_path: str,
    annotation_path: str,
):
    all_annotation_paths = glob(
        os.path.join(annotation_path, "*.csv")
    )
    all_image_paths = glob(
        os.path.join(image_path, "*.jpg")
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

    return full_annotation_df