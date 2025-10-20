import os
from typing import Literal, Dict

from cvlabel.convert.batch.labelme2yolo import labelme2yolo_batch


def make_yolo_dataset(
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