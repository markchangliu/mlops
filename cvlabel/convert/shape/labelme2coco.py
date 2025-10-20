from typing import Dict, Tuple

import pycocotools.mask as pycocomask

import cvlabel.typedef.labelme as labelme_type
import cvlabel.typedef.coco as coco_type
import cvstruct.convert.poly2poly as poly2poly
import cvstruct.convert.poly2rle as poly2rle
import cvstruct.convert.poly2bbox as poly2bbox


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

        rle = poly2rle.polys2rle_merge_coco(polys, img_hw)
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
        