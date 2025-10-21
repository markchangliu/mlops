
from cvproject.shapes.convert.cnt2poly import cnt2poly_labelme
from cvproject.shapes.convert.mask2cnt import mask2cnt_merge
from cvproject.shapes.typedef.masks import MaskType
from cvproject.shapes.typedef.polys import PolyLabelmeType


def mask2poly_labelme(
    mask: MaskType,
    approx_flag: bool
) -> PolyLabelmeType:
    cnt = mask2cnt_merge(mask, approx_flag)
    poly = cnt2poly_labelme(cnt)
    return poly