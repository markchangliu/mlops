import copy
from typing import Tuple, List, Dict, Union

import numpy as np

import cvstruct.typedef.contours as cnt_type


def is_clockwise(contour: cnt_type.ContourType) -> bool:
    contour_next = np.empty_like(contour)
    point_next_idx = list(range(1, len(contour) + 1))
    point_next_idx[-1] = 0
    contour_next[:, ...] = contour_next[point_next_idx, ...]

    shoelace_value = contour_next - contour
    shoelace_value = shoelace_value[:, :, 0] * shoelace_value[:, :, 1]
    shoelace_value = np.sum(shoelace_value).item()

    clockwise_flag = shoelace_value < 0

    return clockwise_flag

def get_closest_point_idx(
    cnt1: cnt_type.ContourType,
    cnt2: cnt_type.ContourType
) -> Tuple[int, int]:
    # cnt1 shape: (n, 1, 2)
    # cnt2 shape: (1, m, 2)
    # dist_mat shape: (n, m)
    cnt2 = np.transpose(cnt2, (1, 0, 2))

    dist_mat = np.square(cnt1 - cnt2)
    dist_mat = dist_mat[..., 0] + dist_mat[..., 1]

    idx = np.argmin(dist_mat)
    idx1, idx2 = np.unravel_index(idx, dist_mat.shape)

    return idx1, idx2

def get_contour_groups(
    cnts: cnt_type.ContoursType,
    hierarchies: cnt_type.HierarchiesType
) -> cnt_type.ContourGroupsType:
    # if cnts[0][0][0][0] == 1043 and cnts[0][0][0][1] == 570:
    #     print("hi")

    cnt_group_temp = {"parent": None, "children": []}
    cnt_groups = [copy.deepcopy(cnt_group_temp) for _ in range(len(cnts))]
    for i, (cnt, hierarchy) in enumerate(zip(cnts, hierarchies[0])):
        if hierarchy[3] == -1:
            cnt_groups[i]["parent"] = cnt
        else:
            parent_cnt_idx = hierarchy[2]
            cnt_groups[parent_cnt_idx]["children"].append(cnt)
    
    cnt_groups_ = []
    for cnt_group in cnt_groups:
        if cnt_group["parent"] is not None:
            cnt_groups_.append(cnt_group)
    
    cnt_groups = cnt_groups_
    del cnt_groups_

    return cnt_groups

