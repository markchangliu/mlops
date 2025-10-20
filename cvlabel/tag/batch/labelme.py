import os
from pathlib import Path
from typing import Union, List

from cvlabel.tag.file.labelme import tag_img_from_dirname, tag_img_from_flags


def tag_imgs_from_flags(
    labelme_dirs: List[Union[str, os.PathLike]],
    export_dirs: List[Union[str, os.PathLike]]
) -> None:
    assert isinstance(labelme_dirs, list)
    assert isinstance(export_dirs, list)

    for labelme_dir, export_dir in zip(labelme_dirs, export_dirs):
        filenames = os.listdir(labelme_dir)
        filenames.sort()

        os.makedirs(export_dir, exist_ok = True)

        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            
            img_stem = Path(filename).stem
            labelme_p = os.path.join(labelme_dir, filename)
            export_p = os.path.join(export_dir, f"{img_stem}.json")

            tag_img_from_flags(labelme_p, export_p)

def tag_imgs_from_dirname(
    labelme_dirs: List[Union[str, os.PathLike]],
    export_dirs: List[Union[str, os.PathLike]],
    dir_prefix: Union[str, None],
    dir_suffix: Union[str, None],
    exclude_tags: List[str],
) -> None:
    assert isinstance(labelme_dirs, list)
    assert isinstance(export_dirs, list)

    for labelme_dir, export_dir in zip(labelme_dirs, export_dirs):
        filenames = os.listdir(labelme_dir)
        filenames.sort()

        os.makedirs(export_dir, exist_ok = True)

        if dir_prefix is not None:
            dirname_tag = str(Path(labelme_dir).relative_to(dir_prefix))
        else:
            dirname_tag = str(labelme_dir)
        
        if dir_suffix is not None:
            dirname_tag = dirname_tag.replace(dir_suffix, "")

        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            
            img_stem = Path(filename).stem
            export_p = os.path.join(export_dir, f"{img_stem}.json")

            tag_img_from_dirname(export_p, dirname_tag, exclude_tags)