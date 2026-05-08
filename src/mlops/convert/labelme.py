import json
import os
from pathlib import Path
from typing import Literal, Union

import numpy as np

import mlops.convert.shapes as shapes_lib
import mlops.core.instances as insts_lib
import mlops.core.schema as schema_lib


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
            
            dataset_dict["img_p_list"].append(img_p)
            dataset_dict["label_p_list"].append(None)
    
    return dataset_dict

def _labelmeFile2instances(
    label_p: str,
    cat_name_id_dict: dict[str, int],
    label_format: Literal["bbox", "poly"],
) -> Union[insts_lib.Instances, None]:
    if not os.path.exists(label_p):
        return None
    
    with open(label_p, "r") as f:
        labelme_dict: schema_lib.LabelmeFileType = json.load(f)
    
    if len(labelme_dict["shapes"]) == 0:
        return None

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

            if not os.path.exists(label_p):
                if not add_empty_img_flag:
                    continue

            dataset_dict["img_p_list"].append(img_p)

            if not os.path.exists(label_p):
                dataset_dict["insts_list"].append(None)
            
            with open(label_p, "r") as f:
                labelme_dict: LabelmeFileType = json.load(f)
                labelme_shape_dict: LabelmeShapeType = labelme_dict["shapes"]
                bboxes = []
                polys = []

                if label_format == "bbox":
                    bboxes
            
            