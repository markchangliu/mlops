import json
import os
import shutil
from typing import Union, Dict, Literal

import cvlabel.convert.shape.labelme2yolo as cvt_shape
import cvlabel.utils.labelme as labelme_utils
import cvlabel.typedef.labelme as labelme_type


def labelme2yolo_file(
    img_p: Union[str, os.PathLike],
    labelme_p: Union[str, os.PathLike],
    export_img_p: Union[str, os.PathLike],
    export_label_p: Union[str, os.PathLike],
    cat_name_id_dict: Dict[str, int],
    shape_type: Literal["bbox", "poly"]
) -> None:
    assert shape_type in ["bbox", "poly"]

    shutil.copy(img_p, export_img_p)

    with open(labelme_p, "r") as f:
        labelme_dict: labelme_type.LabelmeDictType = json.load(f)

    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]

    if shape_type == "poly":
        shape_groups = labelme_utils.get_shape_groups(labelme_dict)
        yolo_labels = cvt_shape.shape_groups_to_yolo_poly(
            shape_groups, cat_name_id_dict, (img_h, img_w)
        )
    elif shape_type == "bbox":
        shapes = labelme_dict["shapes"]
        yolo_labels = cvt_shape.shapes_to_yolo_bbox(
            shapes, cat_name_id_dict, (img_h, img_w)
        )

    with open(export_label_p, "w") as f:
        for l in yolo_labels:
            l = [str(e) for e in l]
            l_txt = " ".join(l)
            l_txt = f"{l_txt}\n"
            f.write(l_txt)