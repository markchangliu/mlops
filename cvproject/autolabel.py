import copy
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pycocotools.mask as pycocomask

import cvproject.datasets.typedef.labelme as labelme_type
from cvproject.models.base import BaseModel
from cvproject.shapes.insts import Insts
from cvproject.shapes.convert.mask2cnt import mask2cnt_merge
from cvproject.shapes.convert.cnt2rle import cnt2rle
from cvproject.shapes.convert.cnt2poly import cnt2poly_labelme


def bbox2mask_labelme(
    model: BaseModel,
    img_p: str,
    labelme_p: str,
    export_labelme_p: str,
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float]
) -> None:
    with open(labelme_p, "r") as f:
        labelme_dict: labelme_type.LabelmeDictType = json.load(f)
    
    shapes_dict = labelme_dict["shapes"]

    img = cv2.imread(img_p)

    bboxes = []

    for shape_dict in shapes_dict:
        if shape_dict["shape_type"] != "rectangle":
            continue
        if len(shape_dict["points"]) != 2:
            continue
        
        bboxes.append(shape_dict["points"])
    
    bboxes = np.asarray(bboxes, dtype = np.int32).reshape(-1, 4)
    masks = model.infer_masks_given_bboxes(img, bboxes)

    export_labelme_dict: labelme_type.LabelmeDictType = copy.deepcopy(labelme_dict)
    export_labelme_dict["shapes"] = []

    for mask, shape_dict in zip(masks, labelme_dict["shapes"]):
        if mask.max() == 0:
            print("Empty mask")
            continue

        shape_dict["shape_type"] = "polygon"
        cnt = mask2cnt_merge(mask, True)

        if len(cnt) < 2:
            print(f"Invalid poly, number of points {len(cnt)} less than 2")
            continue

        rle = cnt2rle(cnt, (img.shape[0], img.shape[1]))
        x1, y1, w, h = pycocomask.toBbox(rle)
        aspect_ratio = h / (w + 1e-8)
        area = pycocomask.area(rle)

        if not aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
            print(f"Abnormal mask, aspect ratio {aspect_ratio:.3f} not in range")
            continue

        if not area_range[0] < area < area_range[1]:
            print(f"Abnormal mask, area {area:.3f} not in range")
            continue

        poly_labelme = cnt2poly_labelme(cnt)
        shape_dict["points"] = poly_labelme
        export_labelme_dict["shapes"].append(shape_dict)
    
    with open(export_labelme_p, "w") as f:
        json.dump(export_labelme_dict, f)


def bbox2mask_labelme_batch(
    model: BaseModel,
    img_dirs: List[str],
    labelme_dirs: List[str],
    export_labelme_dirs: List[str],
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert isinstance(export_labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs) == len(export_labelme_dirs)

    for i_dir, l_dir, e_dir in zip(img_dirs, labelme_dirs, export_labelme_dirs):
        filenames = os.listdir(i_dir)
        filenames.sort()

        os.makedirs(e_dir, exist_ok = True)

        for filename in filenames:
            if not filename.endswith((".png", ".jpg", ".jpeg")):
                continue

            img_name = filename
            img_stem = Path(filename).stem
            img_suffix = Path(filename).suffix
            img_p = os.path.join(i_dir, filename)

            labelme_name = f"{img_stem}.json"
            labelme_p = os.path.join(l_dir, labelme_name)

            if not os.path.exists(labelme_p):
                continue

            export_labelme_p = os.path.join(e_dir, labelme_name)

            bbox2mask_labelme(
                model, img_p, labelme_p, export_labelme_p, 
                aspect_ratio_range, area_range
            )
    