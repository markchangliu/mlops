import os
from typing import Union

import numpy as np

from cvstruct.typedef.insts import InstsType


def insts2npz_mask(
    insts: InstsType,
    export_npz_p: Union[os.PathLike, str],
) -> None:
    bboxes = insts.bboxes
    scores = insts.confs
    masks = insts.masks.astype(np.int32)
    np.savez_compressed(
        export_npz_p, bboxes = bboxes, scores = scores, masks = masks
    )