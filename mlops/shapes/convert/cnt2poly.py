import os
from typing import Union, Tuple

import numpy as np

import mlops.shapes.typedef.contours as cnt_type
import mlops.shapes.typedef.polys as poly_type


def cnt2poly_labelme(
    cnt: cnt_type.ContourType
) -> poly_type.PolyLabelmeType:
    poly_labelme = np.squeeze(cnt).tolist()
    return poly_labelme

def cnt2poly_coco(
    cnt: cnt_type.ContourType
) -> poly_type.PolyCocoType:
    poly_coco = cnt.flatten().tolist()
    return poly_coco

def cnt2poly_yolo(
    cnt: cnt_type.ContourType,
    img_hw: Tuple[int, int]
) -> poly_type.PolyYoloType:
    poly_yolo = np.squeeze(cnt)
    poly_yolo = poly_yolo / np.asarray((img_hw[1], img_hw[0])).reshape(-1, 2)
    poly_yolo = poly_yolo.flatten().tolist()
    return poly_yolo

def cnts2polys_labelme(
    cnts: cnt_type.ContoursType
) -> poly_type.PolysLabelmeType:
    polys_labelme = []

    for cnt in cnts:
        polys_labelme = cnt2poly_labelme(cnt)
        polys_labelme.append(polys_labelme)

    return polys_labelme

def cnts2polys_coco(
    cnts: cnt_type.ContoursType
) -> poly_type.PolysCocoType:
    polys_coco = []

    for cnt in cnts:
        poly_coco = cnt2poly_coco(cnt)
        polys_coco.append(poly_coco)
    
    return polys_coco

def cnts2polys_yolo(
    cnts: cnt_type.ContoursType,
    img_hw: Tuple[int, int]
) -> poly_type.PolysYoloType:
    polys_yolo = []

    for cnt in cnts:
        poly_yolo = cnt2poly_yolo(cnt, img_hw)
        polys_yolo.append(poly_yolo)
    
    return polys_yolo
    