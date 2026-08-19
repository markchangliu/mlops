from typing import TypeAlias, TypedDict, Union, Literal

import numpy as np
from numpy.typing import NDArray


##### BBox #####

BBoxArrT: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.integer], (4, ), [x1, y1, x2, y2]`
"""

BBoxesArrT: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.integer], (N, 4), [[x1, y1, x2, y2], ...]`
"""

BBoxLabelmeT: TypeAlias = tuple[tuple[float, float], tuple[float, float]]
"""
`tuple[tuple[float, float], tuple[float, float]], (2, (2, )), [[x1, y1], [x2, y2]]`
"""

BBoxCocoT: TypeAlias = tuple[float, float, float, float]
"""
`Tuple[float, float, float, float], (4, ), [x1, y1, w, h]`
"""

BBoxYoloT: TypeAlias = tuple[float, float, float, float]
"""
`Tuple[float, float, float, float], (4, ), [x_ctr_norm, y_ctr_norm, w_norm, h_norm]`
"""

BBoxFormatT: TypeAlias = Literal["array", "labelme", "coco", "yolo"]
"""
`Literal["array", "labelme", "coco", "yolo"]`
"""

##### Mask #####

MaskArrT: TypeAlias = NDArray[np.bool_]
"""
`NDArray[np.bool_], (img_h, img_w)`
"""

MasksArrT: TypeAlias = NDArray[np.bool_]
"""
`NDArray[np.bool_], (N, img_h, img_w)`
"""

MaskImgT: TypeAlias = NDArray[np.uint8]
"""
`NDArray[np.uint8], (img_h, img_w), 0 or 255`
"""

MasksImgT: TypeAlias = NDArray[np.uint8]
"""
`NDArray[np.uint8], (N, img_h, img_w), 0 or 255`
"""

MaskFormatT: TypeAlias = Literal["array", "img"]
"""
`Literal["array", "img"]`
"""

##### Poly #####

PolyArrT: TypeAlias = NDArray[np.integer]
"""
`NDArray[np.integer], (num_points, 2), [[x, y], ...]`
"""

PolysArrT: TypeAlias = list["PolyArrT"]
"""
`list[PolySchemaType], (num_polys, (num_points, 2))`
"""

PolyLabelmeT: TypeAlias = list[tuple[float, float]]
"""
`list[tuple[float, float]], (num_points, (2, )), [[x1, y1], [x2, y2], ...]`
"""

PolyCocoT: TypeAlias = list[float]
"""
`list[float], (num_points * 2, ), [x1, y1, x2, y2, x3, y3, ...]`
"""

PolyYoloT: TypeAlias = list[float]
"""
`list[float], (num_points * 2, ), [x1_norm, y1_norm, x2_norm, y2_norm, ...]`
"""

PolyFormatT: TypeAlias = Literal["array", "labelme", "coco", "yolo"]
"""
`Literal["array", "labelme", "coco", "yolo"]`
"""

##### Rle #####

class RleT(TypedDict):
    """
    ```
    "size": Tuple[int, int], [img_h, img_w]
    "counts": str
    ```
    """
    size: tuple[int, int]
    counts: str

##### Labelme Dataset #####

class LabelmeShapeT(TypedDict):
    """
    `points: Union[PolyLabelmeType, BBoxLabelmeType]`
    `label: str`
    `shape_type: Literal["polygon", "rectangle"]`
    `group_id: Optional[str]`
    `flags: Dict[Any, Any]`
    """
    points: Union[PolyLabelmeT, BBoxLabelmeT]
    label: str
    shape_type: Literal["polygon", "rectangle"]
    group_id: Union[str, None]
    flags: dict[str, bool]

class LabelmeFileT(TypedDict):
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
    shapes: list["LabelmeShapeT"]
    imagePath: str
    imageData: Union[str, None]
    imageHeight: int
    imageWidth: int

class LabelmeShapeGroupT(TypedDict):
    """
    `group_id: int`
    `shapes: List[LabelmeShapeType]`
    """
    group_id: int
    shapes: list[LabelmeShapeT]

##### Coco Dataset #####

class CocoImgT(TypedDict):
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

class CocoCatT(TypedDict):
    """
    `id: int`
    `name: str`
    """
    id: int
    name: str

class CocoAnnT(TypedDict):
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
    bbox: BBoxCocoT
    segmentation: list[PolyCocoT]

class CocoFileT(TypedDict):
    """
    `images: List[CocoImgDict]`
    `categories: List[CocoCatDict]`
    `annotations: List[CocoAnnDict]`
    """
    images: list[CocoImgT]
    categories: list[CocoCatT]
    annotations: list[CocoAnnT]

##### Yolo Dataset #####

YoloDetAnnT: TypeAlias = tuple[int, float, float, float, float]
"""
`Tuple[int, float, float, float, float], (5, ), `
`[cat_id, x_ctr_norm, y_ctr_norm, w_norm, h_norm]`
"""

YoloSegAnnT: TypeAlias = list[float]
"""
`Tuple[int, float, ...], (1 + num_points * 2, ),`
`[cat_id, x1_norm, y1_norm, x2_norm, y2_norm, ...]`
"""

