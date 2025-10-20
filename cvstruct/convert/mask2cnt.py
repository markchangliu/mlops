import numpy as np
import cv2

import cvstruct.merge.cnts as cnt_merge
import cvstruct.typedef.masks as mask_type
import cvstruct.typedef.contours as cnt_type


def mask2cnt(
    mask: mask_type.MaskType,
    approx_flag: bool
) -> cnt_type.ContoursType:
    mask_img = mask.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    
    if not approx_flag:
        return cnts
    
    cnts_approx = []
    for cnt in cnts:
        eps = 0.001 * cv2.arcLength(contour, True)
        cnt_approx = cv2.approxPolyDP(cnt, eps, True)
        cnts_approx.append(cnt_approx)
    
    return cnts_approx

def mask2cnt_merge(
    mask: mask_type.MaskType,
    approx_flag: bool
) -> cnt_type.ContourType:
    mask_img = mask.astype(np.uint8) * 255
    cnts, hierarchies = cv2.findContours(mask_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    
    if approx_flag:
        cnts_approx = []
        for cnt in cnts:
            eps = 0.001 * cv2.arcLength(cnt, True)
            cnt_approx = cv2.approxPolyDP(cnt, eps, True)
            cnts_approx.append(cnt_approx)
        
        cnts = cnts_approx
    
    if len(cnts) == 0:
        cnt = np.zeros((0, 1, 2))
    elif len(cnts) == 1:
        cnt = cnts[0]
    else:
        cnt = cnt_merge.merge_contours(cnts, hierarchies)

    return cnt