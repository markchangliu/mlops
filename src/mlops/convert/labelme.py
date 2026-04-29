import json
import os
from pathlib import Path
from typing import Literal

from mlops.core.schema import LabelmeFileType, LabelmeShapeType
from mlops.core.schema import OnlineDatasetType, OfflineDatasetType


def _load_labelme_offline(
    img_root: str,
    labelme_root: str,
    cat_name_id_dict: dict[str, int],
    label_format: Literal["bbox", "mask", "poly"],
    add_empty_img_flag: bool
) -> OfflineDatasetType:
    dataset_dict: OfflineDatasetType = {
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


            