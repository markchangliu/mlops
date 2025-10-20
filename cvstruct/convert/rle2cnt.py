import cv2
import numpy as np

import pycocotools.mask as pycocomask

import cvstruct.merge.cnts as cnt_merge
import cvstruct.typedef.rles as rle_type
import cvstruct.typedef.contours as cnt_type


def rle2cnts(
    rle: rle_type.RLEType,
    approx_flag: bool
) -> cnt_type.ContoursType:
    mask = pycocomask.decode(rle).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    
    if not approx_flag:
        return cnts
    
    cnts_approx = []
    for cnt in cnts:
        eps = 0.001 * cv2.arcLength(contour, True)
        cnt_approx = cv2.approxPolyDP(cnt, eps, True)
        cnts_approx.append(cnt_approx)
    
    return cnts_approx

def rle2cnt_merge(
    rle: rle_type.RLEType,
    approx_flag: bool
) -> cnt_type.ContourType:
    mask = pycocomask.decode(rle).astype(np.uint8) * 255
    cnts, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    
    if approx_flag:
        cnts_approx = []
        for cnt in cnts:
            eps = 0.001 * cv2.arcLength(contour, True)
            cnt_approx = cv2.approxPolyDP(cnt, eps, True)
            cnts_approx.append(cnt_approx)
        
        cnts = cnts_approx
    
    if len(cnts) == 0:
        return cnts[0]
    
    cnt = cnt_merge.merge_contours(cnts, hierarchies)

    return cnt
    
