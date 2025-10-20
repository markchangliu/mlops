import json
import os
from typing import Union, List

from cvlabel.typedef.labelme import LabelmeDictType


def tag_img_from_flags(
    labelme_p: Union[str, os.PathLike],
    export_p: Union[str, os.PathLike]
) -> None:
    with open(labelme_p, "r") as f:
        labelme_dict: LabelmeDictType = json.load(f)

    flags = labelme_dict.get("flags", {})
    tags = []

    if len(flags) == 0:
        labelme_dict["imageLabels"] = []
        return
    
    for k, v in flags.items():
        if v:
            tags.append(k)

    with open(export_p, "w") as f:
        json.dump(tags)
    
def tag_img_from_dirname(
    export_p: Union[str, os.PathLike],
    dirname: str,
    exclude_tags: List[str]
) -> None:
    dirname = dirname.replace("/", "_")
    tags = dirname.split("_")
    tags = set([t.strip() for t in tags if len(t) > 0])
    exclude_tags = set(exclude_tags)

    tags = list(tags - exclude_tags)

    with open(export_p, "w") as f:
        json.dump(tags, f)
    