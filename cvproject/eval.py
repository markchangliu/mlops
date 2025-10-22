from typing import TypedDict, Dict, List

import numpy as np
from numpy.typing import NDArray
import pycocotools
import pycocotools.mask as pycocomask

from cvproject.shapes.typedef.masks import MasksType


class MatchResultType(TypedDict):
    # dt: Dict[int, List[int]]
    # gt: Dict[int, List[int]]
    tp_dt_indice: List[int]
    tp_gt_indice: List[int]
    fp_dt_indice: List[int]
    fn_gt_indice: List[int]


def iou_mask(
    masks1: MasksType,
    masks2: MasksType,
) -> NDArray[np.number]:
    """
    Params
    ------
    - `masks1`: shape (n, h, w)
    - `masks2`: shape (m, h, w)

    Returns
    -----
    - `ious`: (n, m)
    """
    is_crowd = [False] * len(masks2)
    masks1 = np.transpose(masks1, (1, 2, 0)).astype(np.uint8)
    masks2 = np.transpose(masks2, (1, 2, 0)).astype(np.uint8)
    rles1 = pycocomask.encode(np.asfortranarray(masks1))
    rles2 = pycocomask.encode(np.asfortranarray(masks2))
    ious = pycocomask.iou(rles1, rles2, is_crowd)
    return ious

def match_dt_gt(
    masks_dt: MasksType,
    masks_gt: MasksType,
    iou_thres: float,
    gt_match_1dt_atmost_flag: bool,
    allow_suboptim_match: bool,
) -> NDArray[np.bool_]:
    ious = iou_mask(masks_dt, masks_gt)
    match_res = ious > iou_thres

    match_res_dt = 


# def match_dt_gt(
#     masks_dt: MasksType,
#     masks_gt: MasksType,
#     iou_thres: float,
#     gt_match_1dt_atmost_flag: bool,
#     dt_match_1gt_atmost_flag: bool,
#     match_1vs1_only_flag: bool,
#     allow_nonbest_match_flag: bool,
# ) -> NDArray[np.integer]:
#     """
#     Params
#     -----
#     - `masks_dt`: `MasksType`, shape `(n, h, w)`
#     - `masks_dt`: `MasksType`, shape `(m, h, w)`
#     - `iou_thres`: `float`
#     - `match_1vs1_only_flag`: `bool`
#         - `True`: 1 gt can only have 1 dt match at most
#         - `False`: 1 gt can have multiple dt match
#     - `allow_nonbest_match_flag`: `bool`, used only when `match_1vs1_only_flag=True`;
#     decides whether to adjust the match results for dts that have mul
#         - `True`: if a dt match

#     Returns
#     -----
#     - `tp_flags`: `NDArray[bool]`, shape `(n, )`
#     - `tp_gt_indice`: `NDArray[int]`, shape `(n, )`, -1 means no match
#     - `fn_flags`: `NDArray[bool]`, shape `(m, )`
#     """
#     ious = iou_mask(masks_dt, masks_gt)
#     num_dts = len(masks_dt)
#     num_gts = len(masks_gt)

#     # shape (num_dts, )
#     match_gt_indice = np.argmax(ious, axis = 1)
#     best_ious_dt = ious[range(num_dts), match_gt_indice]

#     # shape (num_gts, )
#     match_dt_indice = np.argmax(ious, axis = 0)
#     best_ious_gt = ious[match_dt_indice, range(num_gts)]

#     if not match_1vs1_only_flag:
#         tp_flags = best_ious_dt > iou_thres
#         tp_gt_indice = match_gt_indice
#         tp_gt_indice[~tp_flags] = -1
#         fn_flags = ~np.isin(list(range(num_gts)), tp_gt_indice)

#     else:
#         fn_flags = best_ious_gt < iou_thres
#         match_dt_indice[fn_flags] = -1

        

#         tp_flags = best_ious_dt > iou_thres
#         tp_gt_indice = match_gt_indice

#         # set 


        

