import os
from pathlib import Path
from typing import Dict, Tuple, List, Union, Literal

import pycocotools.mask as pycocomask

import cvproject.datasets.typedef.labelme as labelme_type
import cvproject.datasets.typedef.coco as coco_type
import cvproject.shapes.convert.poly2poly as poly2poly
import cvproject.shapes.convert.poly2rle as poly2rle
import cvproject.datasets.utils.labelme as labelme_utils


def shape_groups_to_coco_rle(
    shape_groups: labelme_type.LabelmeShapeGroupsType,
    cat_name_id_dict: Dict[str, int],
    img_hw: Tuple[int, int],
    img_id: int,
    start_ann_id: int
) -> Tuple[coco_type.CocoAnnDictType, int]:
    coco_anns = []
    curr_ann_id = start_ann_id

    for i, (group_id, shape_group) in enumerate(shape_groups.items()):
        if len(shape_group) == 0:
            print("Empty shape group")
            continue
        
        polys_coco = []

        for shape in shape_group:
            poly_labelme = shape_group[0]["points"]

            if len(poly_labelme) < 2:
                print("Abnormal poly, num_points less than 2")
                continue

            if len(poly_labelme) == 2:
                poly_labelme += poly_labelme[-1]

            poly_coco = poly2poly.poly2poly_labelme2coco(poly_labelme)
            polys_coco.append(poly_coco)

        rle = poly2rle.polys2rle_merge_coco(polys_coco, img_hw)
        area = pycocomask.area(rle)
        bbox = pycocomask.toBbox(rle)

        coco_ann: coco_type.CocoAnnDictType = {}
        coco_ann["area"] = area
        coco_ann["bbox"] = bbox
        coco_ann["segmentation"] = rle
        coco_ann["id"] = curr_ann_id
        coco_ann["iscrowd"] = 0
        coco_ann["image_id"] = img_id

        coco_anns.append(coco_ann)

        curr_ann_id += 1
    
    return coco_anns, curr_ann_id

# def labelme2coco_batch(
#     img_dirs: List[Union[str, os.PathLike]],
#     labelme_dirs: List[Union[str, os.PathLike]],
#     export_root: Union[str, os.PathLike],
#     export_img_dirname: str,
#     export_coco_name: str,
#     cat_name_id_dict: Dict[str, int],
#     shape_type: Literal["bbox", "poly", "rle"]
# ) -> None:
#     assert isinstance(img_dirs, list)
#     assert isinstance(labelme_dirs, list)
#     assert len(img_dirs) == len(labelme_dirs)

#     assert shape_type in ["bbox", "poly", "rle"]

#     export_img_dir = os.path.join(export_root, "images", export_img_dirname)
#     export_coco_p = os.path.join(export_root, export_coco_name)

#     os.makedirs(export_img_dir, exist_ok = True)

#     coco_imgs: List[coco_type.CocoImgDictType] = []
#     coco_anns: List[coco_type.CocoAnnDictType] = []
#     coco_cats: List[coco_type.CocoCatDictType] = []

#     for cat_name, cat_id in cat_name_id_dict.items():
#         coco_cat: coco_type.CocoCatDictType = {}
#         coco_cat["id"] = cat_id
#         coco_cat["name"] = cat_name
#         coco_cats.append(coco_cat)

#     curr_img_id = 0

#     for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
#         p_generator = labelme_utils.img_labelme_p_generator(
#             img_dir, labelme_dir
#         )

#         for img_p, labelme_p in p_generator:
#             coco_img: coco_type.Co

#             img_suffix = Path(img_p).suffix
#             export_img_name = f"{curr_img_id}{img_suffix}"
#             export_img_p = os.path.join(export_img_dir, export_img_name)

#             export_label_name = f"{curr_img_id}.txt"
#             export_label_p = os.path.join(export_label_dir, export_label_name)