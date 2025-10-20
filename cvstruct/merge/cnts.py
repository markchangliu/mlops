
import numpy as np

import cvstruct.merge.cnts_utils as cnts_utils
import cvstruct.merge.cnts_subs as cnts_subs
import cvstruct.typedef.contours as cnt_type


def merge_contours(
    cnts: cnt_type.ContoursType,
    hierarchies: cnt_type.HierarchiesType
) -> cnt_type.ContourType:
    cnt_groups = cnts_utils.get_contour_groups(cnts, hierarchies)

    cnts_group_merge = []
    for cnt_group in cnt_groups:
        cnt_group_merge = cnts_subs.merge_contour_group(cnt_group)
        cnts_group_merge.append(cnt_group_merge)
    
    cnt_merge = cnts_group_merge[0]
    for cnt in cnts_group_merge[1:]:
        cnt_merge = cnts_subs.merge_two_contours(cnt_merge, cnt)
    
    return cnt_merge

def merge_contours_sibling(
    cnts: cnt_type.ContoursType
) -> cnt_type.ContoursType:
    cnt_merge = cnts[0]

    for cnt in cnts[1:]:
        cnt_merge = cnts_subs.merge_two_contours(cnt_merge, cnt)
    
    return cnt_merge