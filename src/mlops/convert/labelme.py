import json
import os
from pathlib import Path
from typing import Literal, Union

import cv2
import numpy as np

import mlops.convert.shapes as shapes_lib
import mlops.core.instances as insts_lib
import mlops.core.schema as schema_lib


__all__ = [
    "load_labelme"
]


def _load_labelme_offline(
    img_root: str,
    labelme_root: str,
    cat_name_id_dict: dict[str, int],
    label_format: Literal["bbox", "poly"],
    add_empty_img_flag: bool
) -> schema_lib.OfflineDatasetType:
    dataset_dict: schema_lib.OfflineDatasetType = {
        "cat_id_name_dict": {v:k for k, v in cat_name_id_dict.items()},
        "cat_name_id_dict": cat_name_id_dict,
        "format": label_format,
        "img_p_list": [],
        "label_p_list": [],
    }

    for root, subdirs, files in os.walk(img_root):
        for file in files:
            if not file.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            img_p = os.path.join(root, file)
            stem = Path(file).stem
            rel_dir = str(Path(img_p).relative_to(img_root).parent)
            label_p = os.path.join(labelme_root, rel_dir, f"{stem}.json")

            if not os.path.exists(label_p):
                if not add_empty_img_flag:
                    continue

                img = cv2.imread(img_p)
                img_h, img_w = img.shape[:2]

                labelme_dict: schema_lib.LabelmeFileType = {
                    "flags": {},
                    "imageData": None,
                    "imageHeight": img_h,
                    "imageWidth": img_w,
                    "version": "4.5.6",
                    "shapes": []
                }

                with open(label_p, "w") as f:
                    json.dump(labelme_dict, f)
            
            dataset_dict["img_p_list"].append(img_p)
            dataset_dict["label_p_list"].append(None)
    
    return dataset_dict

def _labelmeFile2instances(
    label_p: str,
    cat_name_id_dict: dict[str, int],
    label_format: Literal["bbox", "poly"],
    add_empty_img_flag: bool
) -> Union[insts_lib.Instances, None]:
    if not os.path.exists(label_p):
        if not add_empty_img_flag:
            insts = None
        else:
            insts = insts_lib.Instances.from_values(
                confs = np.ones((0, ), dtype = np.float32),
                cat_ids = np.zeros((0, ), dtype = np.int32),
                bboxes = np.zeros((0, 4), dtype = np.int32),
                polys = [] if label_format == "poly" else None
            )

        return insts
    
    with open(label_p, "r") as f:
        labelme_dict: schema_lib.LabelmeFileType = json.load(f)

    cat_ids = []
    bboxes = []
    polys = []

    for shape_dict in labelme_dict["shapes"]:
        if label_format == "bbox":
            bbox = shape_dict["points"]
            bbox = shapes_lib.bboxLabelme2bboxSchema(bbox)
            bboxes.append(bbox)
        else:
            poly = shape_dict["points"]
            bbox = shapes_lib.polyLabelme2bboxSchema(poly)
            poly = shapes_lib.polyLabelme2polySchema(poly)
            bboxes.append(bbox)
            polys.append([polys])

        cat_name = shape_dict["label"]
        cat_id = cat_name_id_dict[cat_name]
        cat_ids.append(cat_id)

    if len(cat_ids) == 0:
        if add_empty_img_flag:
            confs = np.ones((0, ), dtype = np.float32),
            cat_ids = np.zeros((0, ), dtype = np.int32),
            bboxes = np.zeros((0, 4), dtype = np.int32),
            polys = [] if label_format == "poly" else None

            insts = insts_lib.Instances.from_values(
                confs, cat_ids, bboxes, polys
            )
        else:
            insts = None
    else:
        confs = np.ones(len(cat_ids))
        cat_ids = np.asarray(cat_ids, dtype = np.int32)
        bboxes = np.concat(bboxes, axis = 0)

        if label_format == "bbox":
            polys = None

        insts = insts_lib.Instances.from_values(
            confs = confs, 
            cat_ids = cat_ids,
            bboxes = bboxes,
            polys = polys
        )

    return insts

def _load_labelme_online(
    img_root: str,
    labelme_root: str,
    cat_name_id_dict: dict[str, int],
    label_format: Literal["bbox", "mask", "poly"],
    add_empty_img_flag: bool
) -> schema_lib.OnlineDatasetType:
    dataset_dict: schema_lib.OnlineDatasetType = {
        "cat_id_name_dict": {v:k for k, v in cat_name_id_dict.items()},
        "cat_name_id_dict": cat_name_id_dict,
        "format": label_format,
        "img_p_list": [],
        "insts_list": []
    }

    for root, subdirs, files in os.walk(img_root):
        for file in files:
            if not file.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            img_p = os.path.join(root, file)
            stem = Path(file).stem
            rel_dir = str(Path(img_p).relative_to(img_root).parent)
            label_p = os.path.join(labelme_root, rel_dir, f"{stem}.json")
            
            insts = _labelmeFile2instances(
                label_p, cat_name_id_dict, label_format,
                add_empty_img_flag
            )

            if insts is not None:
                dataset_dict["img_p_list"].append(img_p)
                dataset_dict["insts_list"].append(insts)
    
    return dataset_dict
            
def load_labelme(
    img_root: str,
    label_root: str,
    cat_name_id_dict: dict[str, int],
    label_format: Literal["bbox", "mask", "poly"],
    add_empty_img_flag: bool,
    dataset_mode: Literal["online", "offline"],
) -> Union[schema_lib.OnlineDatasetType, schema_lib.OfflineDatasetType]:
    if dataset_mode == "online":
        dataset_dict = _load_labelme_online(
            img_root, label_root, cat_name_id_dict, label_format,
            add_empty_img_flag
        )
    else:
        dataset_dict = _load_labelme_offline(
            img_root, label_root, cat_name_id_dict, label_format,
            add_empty_img_flag
        )
    
    return dataset_dict
            
            