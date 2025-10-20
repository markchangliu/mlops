import os
from pathlib import Path
from typing import List, Union, Dict, Literal

import cvlabel.utils.labelme as labelme_utils
import cvlabel.typedef.labelme as labelme_type
import cvlabel.convert.file.labelme2yolo as cvt_file


def labelme2yolo_batch(
    img_dirs: List[Union[str, os.PathLike]],
    labelme_dirs: List[Union[str, os.PathLike]],
    export_root: Union[str, os.PathLike],
    export_foldername: str,
    cat_name_id_dict: Dict[str, int],
    shape_type: Literal["bbox", "poly"]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs)

    assert shape_type in ["bbox", "poly"]

    export_img_dir = os.path.join(export_root, "images", export_foldername)
    export_label_dir = os.path.join(export_root, "labels", export_foldername)

    os.makedirs(export_img_dir, exist_ok = True)
    os.makedirs(export_label_dir, exist_ok = True)

    curr_img_id = 0

    for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
        p_generator = labelme_utils.img_labelme_p_generator(
            img_dir, labelme_dir
        )

        for img_p, labelme_p in p_generator:
            img_suffix = Path(img_p).suffix
            export_img_name = f"{curr_img_id}{img_suffix}"
            export_img_p = os.path.join(export_img_dir, export_img_name)

            export_label_name = f"{curr_img_id}.txt"
            export_label_p = os.path.join(export_label_dir, export_label_name)

            cvt_file.labelme2yolo_file(
                img_p, labelme_p, export_img_p, export_label_p, 
                cat_name_id_dict, shape_type
            )

            curr_img_id += 1