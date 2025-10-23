import os
import json
import shutil
from pathlib import Path
from typing import Callable, Union

from cvproject.datasets.typedef.labelme import LabelmeDictType


def _default_name_mapf_img2ply(
    img_name: str,
) -> str:
    img_stem = Path(img_name)
    ply_name = f"{img_stem}.ply"
    return ply_name


def del_labelme_img_data(
    unprocessed_root: str
) -> None:
    for root, subdirs, files in os.walk(unprocessed_root):
        for file in files:
            if not file.endswith(".json"):
                continue
            
            labelme_p = os.path.join(root, file)

            with open(labelme_p, "r") as f:
                labelme_dict: LabelmeDictType = json.load(f)

            labelme_dict["imageData"] = None

            with open(labelme_p, "w") as f:
                json.dump(labelme_dict, f)

def organize_unprocessed_data(
    raw_root: str,
    dataset_root: str,
    name_mapf_img2ply: Union[Callable[[str], str], None],
    labelme_dirname: str,
    del_unprocessed_flag: bool,
) -> None:
    unprocessed_root = os.path.join(raw_root, "unprocessed")

    batchnames = os.listdir(unprocessed_root)
    batchnames.sort()

    if name_mapf_img2ply is None:
        name_mapf_img2ply = _default_name_mapf_img2ply

    raw_img_root = os.path.join(raw_root, "images")
    raw_ply_root = os.path.join(raw_root, "plys")

    data_id = 0

    for bn in batchnames:
        batch_root = os.path.join(unprocessed_root, bn)

        if not os.path.isdir(batch_root):
            continue

        dst_raw_img_dir = os.path.join(raw_img_root, bn, "images")
        dst_raw_ply_dir = os.path.join(raw_ply_root, bn, "plys")
        dst_ds_labelme_dir = os.path.join(dataset_root, "raw_labels", bn, labelme_dirname)

        for root, subdirs, files in os.walk(batch_root):
            for file in files:
                if not file.endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_name = file
                img_stem = Path(img_name).stem
                img_suffix = Path(img_name).suffix
                img_p = os.path.join(root, img_name)

                ply_name = name_mapf_img2ply(img_name)
                ply_p = os.path.join(root, ply_name)

                labelme_name = f"{img_stem}.json"
                labelme_p = os.path.join(root, labelme_name)

                os.makedirs(dst_raw_img_dir, exist_ok=True)
                if os.path.exists(ply_p) and not os.path.exists(dst_raw_ply_dir):
                    os.makedirs(dst_raw_ply_dir)
                if os.path.exists(labelme_p) and not os.path.exists(dst_ds_labelme_dir):
                    os.makedirs(dst_ds_labelme_dir)

                dst_img_name = f"{data_id}{img_suffix}"
                dst_img_p = os.path.join(dst_raw_img_dir, dst_img_name)
                shutil.copy(img_p, dst_img_p)

                dst_ply_name = f"{data_id}.ply"
                dst_ply_p = os.path.join(dst_raw_ply_dir, dst_ply_name)

                if os.path.exists(ply_p):
                    shutil.copy(ply_p, dst_ply_p)

                dst_labelme_name = f"{data_id}.json"
                dst_labelme_p = os.path.join(dst_ds_labelme_dir, dst_labelme_name)

                if os.path.exists(labelme_p):
                    shutil.copy(labelme_p, dst_labelme_p)

                data_id += 1

        if del_unprocessed_flag:
            shutil.rmtree(batch_root)
