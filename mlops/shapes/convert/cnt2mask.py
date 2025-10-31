import os
from typing import Union, Tuple

import numpy as np
import pycocotools.mask as pycocomask

import mlops.shapes.convert.cnt2rle as cnt2rle
import mlops.shapes.typedef.contours as cnt_type
import mlops.shapes.typedef.masks as mask_type


def cnt2mask(
    cnt: cnt_type.ContourType,
    img_hw: Tuple[int, int]
) -> mask_type.MaskType:
    rle = cnt2rle.cnt2rle(cnt, img_hw)
    mask = pycocomask.decode(rle).astype(np.bool_)
    return mask

def cnts2masks(
    cnts: cnt_type.ContoursType,
    img_hw: Tuple[int, int]
) -> mask_type.MasksType:
    rles = cnt2rle.cnts2rles(cnts, img_hw)
    masks = pycocomask.decode(rles).astype(np.bool_)
    masks = np.transpose(masks, (2, 0, 1))
    return masks

def cnts2mask_merge(
    cnts: cnt_type.ContoursType,
    img_hw: Tuple[int, int]
) -> mask_type.MaskType:
    rle_merge = cnt2rle.cnts2rle_merge(cnts, img_hw)
    mask_merge = pycocomask.decode(rle_merge).astype(np.bool_)
    return mask_merge