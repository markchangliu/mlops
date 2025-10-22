import os
import shutil
from pathlib import Path
from typing import Dict, Literal

from cvproject.datasets.convert.labelme2yolo import labelme2yolo_batch


def make_ds_labelme_simple(
    raw_img_root: str,
    dataset_root: str,
    labelme_dirname: str
) -> None:
    dst_labelme_dataset_root = os.path.join(dataset_root, "dataset_labelme")
    dst_train_root = os.path.join(dst_labelme_dataset_root, "train_all")
    dst_test_root = os.path.join(dst_labelme_dataset_root, "test_all")
    dst_train_img_dir = os.path.join(dst_train_root, "images")
    dst_train_labelme_dir = os.path.join(dst_train_root, labelme_dirname)
    dst_test_img_dir = os.path.join(dst_test_root, "images")
    dst_test_labelme_dir = os.path.join(dst_test_root, labelme_dirname)

    os.makedirs(dst_train_img_dir, exist_ok=True)
    os.makedirs(dst_train_labelme_dir, exist_ok=True)
    os.makedirs(dst_test_img_dir, exist_ok=True)
    os.makedirs(dst_test_labelme_dir, exist_ok=True)

    raw_label_root = os.path.join(dataset_root, "raw_labels")
    dirnames = os.listdir(raw_label_root)
    dirnames.sort()

    data_id = 0

    for dirname in dirnames:
        raw_label_dir = os.path.join(raw_label_root, dirname, labelme_dirname)
        raw_img_dir = os.path.join(raw_img_root, dirname, "images")

        filenames = os.listdir(raw_img_dir)
        filenames.sort()

        if dirname.endswith("_test"):
            dst_img_dir = dst_test_img_dir
            dst_labelme_dir = dst_test_labelme_dir
        elif dirname.endswith("_train"):
            dst_img_dir = dst_train_img_dir
            dst_labelme_dir = dst_train_labelme_dir
        else:
            raise NotImplementedError(f"unrecognized split {dirname}")

        for filename in filenames:
            if not filename.endswith((".png", ".jpg", ".jpeg")):
                continue

            img_name = filename
            img_stem = Path(img_name).stem
            img_suffix = Path(img_name).suffix
            img_p = os.path.join(raw_img_dir, img_name)

            labelme_name = f"{img_stem}.json"
            labelme_p = os.path.join(raw_label_dir, labelme_name)

            if not os.path.exists(img_p):
                continue
            if not os.path.exists(labelme_p):
                continue
            
            dst_img_name = f"{data_id}{img_suffix}"
            dst_labelme_name = f"{data_id}.json"
            dst_img_p = os.path.join(dst_img_dir, dst_img_name)
            dst_labelme_p = os.path.join(dst_labelme_dir, dst_labelme_name)

            shutil.copy(img_p, dst_img_p)
            shutil.copy(labelme_p, dst_labelme_p)

            data_id += 1

def convert_ds_labelme2yolo(
    dataset_root: str,
    cat_name_id_dict: Dict[str, int],
    labelme_dirname: str,
    shape_type: Literal["bbox", "poly"]
) -> None:
    labelme_root = os.path.join(dataset_root, "dataset_labelme")
    yolo_root = os.path.join(dataset_root, "dataset_yolo")
    splits = os.listdir(labelme_root)
    splits = [s for s in splits if s.startswith(("train_", "test_"))]
    splits = [s for s in splits if os.path.isdir(os.path.join(labelme_root, s))]

    for split in splits:
        img_dir = os.path.join(labelme_root, split, "images")
        labelme_dir = os.path.join(labelme_root, split, labelme_dirname)

        labelme2yolo_batch(
            [img_dir], [labelme_dir], yolo_root,
            split, cat_name_id_dict, shape_type
        )