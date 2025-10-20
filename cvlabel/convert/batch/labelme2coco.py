import os
from typing import Union, List, Dict

import cvlabel.typedef.coco as coco_type


def labelme2yolo_batch(
    img_dirs: List[Union[str, os.PathLike]],
    labelme_dirs: List[Union[str, os.PathLike]],
    export_root: Union[str, os.PathLike],
    export_img_dirname: str,
    export_coco_name: str,
    cat_name_id_dict: Dict[str, int],
    shape_type: Literal["bbox", "poly", "rle"]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs)

    assert shape_type in ["bbox", "poly", "rle"]

    export_img_dir = os.path.join(export_root, "images", export_img_dirname)
    export_coco_p = os.path.join(export_root, export_coco_name)

    os.makedirs(export_img_dir, exist_ok = True)

    coco_imgs: List[coco_type.CocoImgDictType] = []
    coco_anns: List[coco_type.CocoAnnDictType] = []
    coco_cats: List[coco_type.CocoCatDictType] = []

    for cat_name, cat_id in cat_name_id_dict.items():
        coco_cat: coco_type.CocoCatDictType = {}
        coco_cat["id"] = cat_id
        coco_cat["name"] = cat_name
        coco_cats.append(coco_cat)

    curr_img_id = 0

    for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
        p_generator = labelme_utils.img_labelme_p_generator(
            img_dir, labelme_dir
        )

        for img_p, labelme_p in p_generator:
            coco_img: coco_type.Co

            img_suffix = Path(img_p).suffix
            export_img_name = f"{curr_img_id}{img_suffix}"
            export_img_p = os.path.join(export_img_dir, export_img_name)

            export_label_name = f"{curr_img_id}.txt"
            export_label_p = os.path.join(export_label_dir, export_label_name)

