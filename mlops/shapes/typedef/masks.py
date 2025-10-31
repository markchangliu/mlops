from typing import Tuple, Dict, Any, List

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt


MaskType: TypeAlias = npt.NDArray[np.bool_]
"""
`MaskType`
    `NDArray[np.bool_]`, `(img_h, img_w)`
"""

MaskShapeType: TypeAlias = Tuple[int, int]
"""
`MaskShapeType`
    `Tuple[int, int]`, `[img_h, img_w]`
"""

MaskImgType: TypeAlias = npt.NDArray[np.uint8]
"""
`MaskImgType`
    `NDArray[np.uint8]`, `(img_h, img_w)`, 0 ~ 255
"""

MasksType: TypeAlias = npt.NDArray[np.bool_]
"""
`MasksType`
    `NDArray[np.bool_]`, `(num_masks, img_h, img_w)`
"""

MasksShapeType: TypeAlias = Tuple[int, int, int]
"""
`MasksShapeType`
    `Tuple[int, int, int]`, `[num_masks, img_h, img_w]`
"""