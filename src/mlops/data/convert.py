import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Literal, Union, TypedDict

import src.mlops.data.typing as dataTP


_IMG_EXTS = [".jpg", ".png", ".jpeg", ".bmp", ".JPG", ".PNG"]


def labelme2coco_bbox_shape(
    points: dataTP.BBoxLabelmeT
) -> dataTP.BBoxCocoT:
    """
    Task:
        把一个 labelme bbox 转化为 coco bbox。
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # 计算左上角和右下角坐标
    x1 = float(min(x_coords))
    y1 = float(min(y_coords))
    x2 = float(max(x_coords))
    y2 = float(max(y_coords))
    
    w = x2 - x1
    h = y2 - y1
    
    # COCO bbox 格式: (x1, y1, w, h)
    bbox: dataTP.BBoxCocoT = (x1, y1, w, h)

    return bbox

def labelme2coco_poly_shape(
    points: dataTP.PolyLabelmeT
) -> dataTP.PolyCocoT:
    """
    Task:
        把一个 labelme poly 转化为 coco poly
    """
    poly = []
    for x, y in points:
        poly.append(x)
        poly.append(y)
    return poly

def _polygon_area(poly: dataTP.PolyCocoT) -> float:
    """
    Task:
        使用鞋带公式计算一个 coco poly 的多边形面积。
    """
    xs = poly[0::2]
    ys = poly[1::2]
    n = len(xs)

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]

    return abs(area) / 2.0

class LabelFile2CocoReturnT(TypedDict):
    """
    ```
    一个将单一的 Labelme json 或者 Yolo txt 标注文件转化为 Coco
    格式的标注列表的函数所返回的数据类型。

    keys & values:
        anns: list[dataTP.CocoAnnT]
            Coco 格式的标注列表

        end_ann_id: int
            anns 列表中最后一个 annotation 的 id
    ```
    """
    anns: list[dataTP.CocoAnnT]
    end_ann_id: int

def labelme2coco_file(
    labelme_file: Union[str, dataTP.LabelmeFileT],
    img_id: int,
    start_ann_id: int,
    cat_name_id_dict: dict[str, int],
    shape_type: Literal["bbox", "poly"]
) -> "LabelFile2CocoReturnT":
    """
    ```
    Task:
        将 Labelme bbox/poly json 标注文件转换为 COCO 格式的标注列表。

    Args:
        labelme_file: Union[str, dataTP.LabelmeFileT]
            Labelme json 文件路径或已解析的字典数据。
        
        img_id: int,
            该 Labelme json 文件对应的图片 ID。
            也是输出的 Coco 标注列表里的 "image_id"。
        
        start_ann_id: int
            起始的 Coco annotation id。

            Labelme JSON 文件中的第一个 shape 对应的
            Coco 标注列表里的 "id" 为 start_ann_id。

            后续 shape 的 "id" 则在前一个 shape 的 "id" 
            基础上增加1

        cat_name_id_dict: dict[str, int]
            类别名称和 id 的映射。
            labelme 名称不在这个字典中的 labelme shape 将会忽略跳过

        shape_type: Literal["bbox", "poly"]
            指定 Labelme json 中的 shape 类型是 bbox 还是 poly。
    
    Returns:
        LabelFile2CocoReturnT: dict
            详见 LabelFile2CocoReturnT 定义
    ```
    """

    # 若传入的是文件路径则解析为 dict
    if isinstance(labelme_file, str):
        with open(labelme_file, "r") as f:
            labelme_data: dataTP.LabelmeFileT = json.load(f)
    else:
        labelme_data = labelme_file

    anns: list[dataTP.CocoAnnT] = []
    ann_id = start_ann_id

    for shape in labelme_data["shapes"]:
        label = shape["label"]
        if label not in cat_name_id_dict:
            continue

        points = shape["points"]

        if shape_type == "bbox":
            bbox = labelme2coco_bbox_shape(points)
            area = int(bbox[2] * bbox[3])
            segmentation = [labelme2coco_poly_shape(points)]
        else:
            poly = labelme2coco_poly_shape(points)
            xs = poly[0::2]
            ys = poly[1::2]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            bbox = (x1, y1, x2 - x1, y2 - y1)
            area = int(_polygon_area(poly))
            segmentation = [poly]

        ann: dataTP.CocoAnnT = {
            "id": ann_id,
            "iscrowd": 0,
            "image_id": img_id,
            "category_id": cat_name_id_dict[label],
            "area": area,
            "bbox": bbox,
            "segmentation": segmentation,
        }
        anns.append(ann)
        ann_id += 1

    return {"anns": anns, "end_ann_id": ann_id}

def labelme2coco_dataset(
    labelme_roots: list[str],
    coco_root: str,
    cat_name_id_dict: dict[str, int],
    shape_type: Literal["bbox", "poly"],
    add_background_img: bool,
    merge_shape_group: bool,
    flatten_reindex_img: bool,
    export_img_mode: Literal["copy", "symbolic", "paths"],
) -> None:
    """
    ```
    Task:
        将 Labelme 数据集转化为 Coco 数据集。

    Args:
        labelme_roots: list[str]
            要转换的 labelme 数据集根路径列表，列表的元素个数
            可以大于1，也就是说可以将多个 labelme 数据集转换成一份
            coco 数据集。

            每一个元素是一个 labelme_root。
            
            一个 labelme_root 内的路径结构：
            ```
            # <...> 表示代指，实际中的文件夹或文件不是真的叫这个名字
            
            <labelme_root>:
            | - <nested_folder_with_arbitrary_depth>
            | - | - XXX.png/jpg/jpeg/bmp/..
            | - | - XXX.json
            | - <another_folder_with_arbitrary_depth>
            | - | - XXX.png/jpg/jpeg/bmp/..
            | - | - XXX.json
            | - ...
            ```

            其中 `XXX.png/jpg/jpeg/bmp/..` 是一个图片，
            `XXX.json` 是该图片对应的 labelme json 标注文件。

            有些图片可能没有对应的 labelme json 文件，或者 
            labelme json 文件里的 shape 列表为空，这些图片是 
            背景图片，对它们的处理由参数 add_background_img 决定
        
        coco_root: str
            导出的 Coco 数据集根路径。

            coco_root 路径结构:
            ```
            # <...> 表示代指，实际中的文件夹或文件不是真的叫这个名字

            <coco_root>
            | - images
            | - | - <flatten_images_or_same_relative_directory_architecture_as_in_labelme_root>
            | - | - | - XXX.png/jpg/jpeg/bmp/..
            | - paths
            | - | - <same_directory_architecture_as_images_subfolder>
            | - | - | - XXX.path
            | - coco.json
            ```
            
            其中 coco_root/images 是图片根路径，里面的图片可以
            平铺放置 (当参数 flatten_reindex_img 为 True)，也可以
            保持和原始 labelme_root 一样的文件夹结构 (当参数 
            flatten_reindex_img 为 False)。

            coco_root/paths 用来记录图片的原始相对路径，里面的
            路径结构和 coco_root/images 相同，每一张图片都有一个
            同名的但是后缀为 ".path" 的文件。

            ".path" 文件实际上和 ".txt" 文件相同，可以被编辑器或者
            记事本打开路演编辑。

            ".path" 内写的是对应图片在 labelme_root 中的原始图片
            的相对路径。

            比如 coco_root/images/a.jpg，
            它来自于 labelme_root/folderA/x.jpg,
            那么 coco_root/paths/a.path 里写的就是 "folderA/x.jpg"

            ".path" 文件的意义是，在 flatten_reindex_img 为 True，
            图片经过重命名时，保留一份来源的记录。

            coco_root/coco.json 是 coco 标注文件，里面的图片路径 
            (也就是图片列表里的 file_name) 是相对于 coco_root/images
            的相对路径。
        
        cat_name_id_dict: dict[str, int]
            类别名称和 id 的映射。
            labelme 名称不在这个字典中的 labelme shape 将会忽略跳过
        
        shape_type: Literal["bbox", "poly"]
            指定 Labelme json 中的 shape 类型是 bbox 还是 poly。

        add_background_img: bool
            对于没有标注的背景图片的处理。

            背景图片既包括缺省 labelme json 的图片，也包括有
            labelme json 但是里面的 shape 列表为空的图片

            设为 True 时，会把这些图片加入 coco 图片列表。

            设为 False 时，会忽略这些图片，不把它们加入 coco 图片列表。
        
        merge_shape_group: bool
            是否融合相同 group_id 的 labelme shape。

            当一个 labelme shape 的 group_id 缺省时，视为一个
            独立的 shape。该参数没有影响。

            当一个 labelme shape 的 group_id 不缺省时，但没有
            其他 shape 的 group_id 和它相同，也将该 shape 视为一个
            独立的 shape。该参数没有影响。

            当一个 labelme shape 的 group_id 不缺省时，且有
            其他 shape 的 group_id 和它相同，那么 merge_shape_group
            参数的设定不同，这些 shape 转换的 coco bbox 或 coco poly
            结果会不同。

            设为 True, 若 labelme shape 的 group_id 相同，
            这回融合这些 shape。coco bbox 会重新计算
            为一个能包裹所有单个 shape bbox 的大 bbox，
            coco segmentation 则会包括所有单个 shape 的 poly。

            设为 False，则每一个 labelme shape 都为一个独立的个体，
            coco bbox 就是单独 shape 的 bbox，coco segmentation
            的元素的个数为一 (仅包含一个 shape 的 poly)。
        
        flatten_reindex_img: bool
            是否将 labelme_root 中的图片重新编号并平铺放置。

            设为 True 时，labelme_root 和 coco_root 的路径结构保持一致。

            设为 False 时，coco_root 内的图片会按序号重命名并平铺放置。

            举例：
            ```
            # <...> 表示代指，实际中的文件夹或文件不是真的叫这个名字

            <labelme_roots>
            | - <labelme_root1>
            | - | - <subfolderA_with_arbitrary_depth>
            | - | - | - a.jpg
            | - | - | - a.json
            | - | - | - ...
            | - | - <subfolderB_with_arbitrary_depth>
            | - | - | - b.jpg
            | - | - | - b.json
            | - | - | - ...
            | - | - ...
            | - <labelme_root2>
            | - ...

            <coco_root> # flatten_reindex_img is True
            | - images
            | - | - <labelme_root1>
            | - | - | - <subfolderA_with_arbitrary_depth>
            | - | - | - | - a.jpg
            | - | - | - | - ...
            | - | - | - <subfolderB_with_arbitrary_depth>
            | - | - | - | - b.jpg
            | - | - | - | - ...
            | - | - | - ...
            | - | - <labelme_root2>
            | - | - ...
            | - paths
            | - | - <labelme_root1>
            | - | - | - <subfolderA_with_arbitrary_depth>
            | - | - | - | - a.path
            | - | - | - | - ...
            | - | - | - <subfolderB_with_arbitrary_depth>
            | - | - | - | - b.path
            | - | - | - | - ...
            | - | - | - ...
            | - | - <labelme_root2>
            | - | - ...
            | - coco.json

            <coco_root> # flatten_reindex_img is False
            | - images
            | - | - 0.jpg # the first image in <labelme_roots>
            | - | - ...
            | - paths
            | - | - 0.path
            | - | - ...
            | - coco.json
            ```
        
        export_img_mode: Literal["copy", "symbolic", "paths"],
            输出图片的方式。

            设为 "copy"，<coco_root>/images 中的图片
            从原始图片拷贝。

            设为 "symbolic"，<coco_root>/images 中的图片
            从原始图片创建软链接。

            设为 "paths"，则只有 <coco_root>/paths 和 
            <coco_root>/coco.json，不创建 <coco_root>/images
    
    Returns: 
        None
    ```
    """

    # your code here
