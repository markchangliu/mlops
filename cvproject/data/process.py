import os
import json

from cvproject.datasets.typedef.labelme import LabelmeDictType


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

