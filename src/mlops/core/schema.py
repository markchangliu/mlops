from typing import TypeAlias, TypedDict, Union, Literal

import numpy as np
from numpy.typing import NDArray


##### BBox #####

BBoxSchemaType: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.integer], (4, ), [x1, y1, x2, y2]`
"""

BBoxesSchemaType: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.integer], (N, 4), [[x1, y1, x2, y2], ...]`
"""

BBoxLabelmeType: TypeAlias = tuple[tuple[float, float], tuple[float, float]]
"""
`tuple[tuple[float, float], tuple[float, float]], (2, (2, )), [[x1, y1], [x2, y2]]`
"""

BBoxCocoType: TypeAlias = tuple[float, float, float, float]
"""
`Tuple[float, float, float, float], (4, ), [x1, y1, w, h]`
"""

BBoxYoloType: TypeAlias = tuple[float, float, float, float]
"""
`Tuple[float, float, float, float], (4, ), [x_ctr_norm, y_ctr_norm, w_norm, h_norm]`
"""

BBoxType: TypeAlias = Union[BBoxSchemaType, BBoxLabelmeType, BBoxCocoType, BBoxYoloType]
"""
`Union[BBoxSchemaType, BBoxLabelmeType, BBoxCocoType, BBoxYoloType]`
"""

BBoxesType: TypeAlias = Union[BBoxesSchemaType, list[BBoxLabelmeType], list[BBoxCocoType], list[BBoxYoloType]]
"""
`Union[BBoxesSchemaType, list[BBoxLabelmeType], list[BBoxCocoType], list[BBoxYoloType]]`
"""

BBoxFormatType: TypeAlias = Literal["schema", "labelme", "coco", "yolo"]
"""
`Literal["schema", "labelme", "coco", "yolo"]`
"""

##### Mask #####

MaskSchemaType: TypeAlias = NDArray[np.bool_]
"""
`NDArray[np.bool_], (img_h, img_w)`
"""

MasksSchemaType: TypeAlias = NDArray[np.bool_]
"""
`NDArray[np.bool_], (N, img_h, img_w)`
"""

MaskImgType: TypeAlias = NDArray[np.uint8]
"""
`NDArray[np.uint8], (img_h, img_w), 0 or 255`
"""

MasksImgType: TypeAlias = NDArray[np.uint8]
"""
`NDArray[np.uint8], (N, img_h, img_w), 0 or 255`
"""

MaskType: TypeAlias = Union[MaskSchemaType, MaskImgType]
"""
`Union[MaskSchemaType, MaskImgType]`
"""

MasksType: TypeAlias = Union[MasksSchemaType, MasksImgType]
"""
`Union[MasksSchemaType, MasksImgType]`
"""

MaskFormatType: TypeAlias = Literal["schema", "img"]
"""
`Literal["schema", "img"]`
"""

##### Poly #####

PolySchemaType: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.integer], (num_points, 2), [[x, y], ...]`
"""

PolyLabelmeType: TypeAlias = list[tuple[float, float]]
"""
`list[tuple[float, float]], (num_points, (2, )), [[x1, y1], [x2, y2], ...]`
"""

PolyCocoType: TypeAlias = list[float]
"""
`list[float], (num_points * 2, ), [x1, y1, x2, y2, x3, y3, ...]`
"""

PolyYoloType: TypeAlias = list[float]
"""
`list[float], (num_points * 2, ), [x1_norm, y1_norm, x2_norm, y2_norm, ...]`
"""

PolyType: TypeAlias = Union[PolySchemaType, PolyLabelmeType, PolyCocoType, PolyYoloType]
"""
`Union[PolySchemaType, PolyLabelmeType, PolyCocoType, PolyYoloType]`
"""

PolyFormatType: TypeAlias = Literal["schema", "labelme", "coco", "yolo"]
"""
`Literal["schema", "labelme", "coco", "yolo"]`
"""

##### Rle #####

class RleType(TypedDict):
    """
    `"size": Tuple[int, int], [img_h, img_w]`
    `"counts": str`
    """
    size: tuple[int, int]
    counts: str

##### Labelme Dataset #####

class LabelmeShapeType(TypedDict):
    """
    `points: Union[PolyLabelmeType, BBoxLabelmeType]`
    `label: str`
    `shape_type: Literal["polygon", "rectangle"]`
    `group_id: Optional[str]`
    `flags: Dict[Any, Any]`
    """
    points: Union[PolyLabelmeType, BBoxLabelmeType]
    label: str
    shape_type: Literal["polygon", "rectangle"]
    group_id: Union[str, None]
    flags: dict[str, bool]

class LabelmeFileType(TypedDict):
    """
    `version: str`
    `flags: Dict[str, bool]`
    `shapes: List[LabelmeShapeDictType]`
    `imagePath: str`
    `imageData: Optional[str]`
    `imageHeight: int`
    `imageWidth: int`
    """
    version: str
    flags: dict[str, bool]
    shapes: list["LabelmeShapeType"]
    imagePath: str
    imageData: Union[str, None]
    imageHeight: int
    imageWidth: int

class LabelmeShapeGroupType(TypedDict):
    """
    `group_id: int`
    `shapes: List[LabelmeShapeType]`
    """
    group_id: int
    shapes: list[LabelmeShapeType]

##### Coco Dataset #####

class CocoImgType(TypedDict):
    """
    `height: int`
    `width: int`
    `id: int`
    `file_name: str`
    """
    height: int
    width: int
    id: int
    file_name: str

class CocoCatType(TypedDict):
    """
    `id: int`
    `name: str`
    """
    id: int
    name: str

class CocoAnnType(TypedDict):
    """
    `id: int`
    `iscrowd: Literal[0, 1]`
    `image_id: int`
    `category_id: int`
    `area: int`,
    `bbox: BBoxCocoType`,
    `segmentation: PolysCocoType`
    """
    id: int
    iscrowd: Literal[0, 1]
    image_id: int
    category_id: int
    area: int
    bbox: BBoxCocoType
    segmentation: list[PolyCocoType]

class CocoFileType(TypedDict):
    """
    `images: List[CocoImgDict]`
    `categories: List[CocoCatDict]`
    `annotations: List[CocoAnnDict]`
    """
    images: list[CocoImgType]
    categories: list[CocoCatType]
    annotations: list[CocoAnnType]

##### Yolo Dataset #####

YoloDetAnnType: TypeAlias = tuple[int, float, float, float, float]
"""
`Tuple[int, float, float, float, float], (5, ), `
`[cat_id, x_ctr_norm, y_ctr_norm, w_norm, h_norm]`
"""

YoloSegAnnType: TypeAlias = list[float]
"""
`Tuple[int, float, ...], (1 + num_points * 2, ),`
`[cat_id, x1_norm, y1_norm, x2_norm, y2_norm, ...]`
"""

##### Instances #####

class InstancesType(TypedDict):
    """
    `cat_ids: NDArray[np.integer]`
    `confs: NDArray[np.floating]`
    `bboxes: BBoxesSchemaType`
    `polys: Union[list[PolySchemaType], None]`
    """
    cat_ids: NDArray[np.integer]
    confs: NDArray[np.floating]
    bboxes: BBoxesSchemaType
    polys: Union[list[PolySchemaType], None]

##### Schema Dataset #####

class OnlineDatasetType(TypedDict):
    """
    `img_p_list: list[str]`
    `insts_list: list[Union[InstancesType, None]]`
    `cat_id_name_dict: dict[int, str]`
    `cat_name_id_dict: dict[str, int]`
    `format: Literal["bbox", "poly"]`
    """
    img_p_list: list[str]
    insts_list: list[Union[InstancesType, None]]
    cat_id_name_dict: dict[int, str]
    cat_name_id_dict: dict[str, int]
    format: Literal["bbox", "poly"]

class OfflineDatasetType(TypedDict):
    """
    `img_p_list: list[str]`
    `label_p_list: list[Union[str, None]]`
    `cat_id_name_dict: dict[int, str]`
    `cat_name_id_dict: dict[str, int]`
    `format: Literal["bbox", "poly"]`
    """
    img_p_list: list[str]
    label_p_list: list[Union[str, None]]
    cat_id_name_dict: dict[int, str]
    cat_name_id_dict: dict[str, int]
    format: Literal["bbox", "poly"]

##### Evaluation Result #####

class DatasetEvalResult(TypedDict):
    """
    `img_p_list: list[str]`
    `gt_p_list: list[str]`
    `gt_inst_list: Union[list[InstancesType], None]`
    `pred_p_list: list[str]`
    `pred_inst_list: Union[list[InstancesType], None]`
    `tp_pred_id_list: list[list[int]]`
    `fp_pred_id_list: list[list[int]]`
    `fn_gt_id_list: list[list[int]]`
    """
    img_p_list: list[str]
    gt_p_list: list[str]
    gt_inst_list: Union[list[InstancesType], None]
    pred_p_list: list[str]
    pred_inst_list: Union[list[InstancesType], None]
    tp_pred_id_list: list[list[int]]
    fp_pred_id_list: list[list[int]]
    fn_gt_id_list: list[list[int]]