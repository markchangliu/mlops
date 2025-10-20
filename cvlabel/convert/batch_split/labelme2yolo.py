import os
from typing import Union, List, Dict, Literal

import cvlabel.convert.batch.labelme2yolo as cvt_batch


def labelme2yolo_split(
    img_dirs: Union[str, os.PathLike],
    labelme_dirs: Union[str, os.PathLike],
    export_root: Union[str, os.PathLike],
    export_foldernames: List[str],
    splits_kwords: List[List[str]],
    cat_name_id_dict: Dict[str, int],
    shape_type: Literal["bbox", "poly"]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs)

    assert shape_type in ["bbox", "poly"]

    splits_convert_kwargs = []

    for kws, foldername in zip(splits_kwords, export_foldernames):
        split_convert_kwargs = {
            "img_dirs": [],
            "labelme_dirs": [],
            "export_root": export_root,
            "export_foldername": foldername,
            "cat_name_id_dict": cat_name_id_dict,
            "shape_type": shape_type
        }

        for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
            include_flags = []
            
            for kw in kws:
                include_flag = False

                if "!" in kw:
                    tokens_exclude = kw.split("!")
                    tokens_exclude = [t.strip() for t in tokens_exclude if len(t) > 0]
                else:
                    tokens_exclude = []

                if "|" in kw:
                    tokens = kw.split("|")
                    tokens = [t.strip() for t in tokens if len(t) > 0]
                else:
                    tokens = [kw]

                for token in tokens:
                    if token in img_dir:
                        include_flag = True
                        break
                
                for token in tokens_exclude:
                    if token in img_dir:
                        include_flag = False
                        break
                
                include_flags.append(include_flag)
            
            if not all(include_flags):
                continue

            split_convert_kwargs["img_dirs"].append(img_dir)
            split_convert_kwargs["labelme_dirs"].append(labelme_dir)

            splits_convert_kwargs.append(split_convert_kwargs)
    
    for kwargs in splits_convert_kwargs:
        cvt_batch.labelme2yolo_batch(**kwargs)
        
