import pycocotools.mask as pycocomask

import cvstruct.typedef.rles as rle_type
import cvstruct.typedef.masks as mask_type


def rle2mask(
    rle: rle_type.RLEType,
) -> mask_type.MaskType:
    mask = pycocomask.decode(rle)
    return mask