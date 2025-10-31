from typing import Optional, Union, List

import numpy as np
import numpy.typing as npt

from mlops.shapes.typedef.bboxes import BBoxesXYXYArrType
from mlops.shapes.typedef.others import CatIDsType, ConfsType
from mlops.shapes.typedef.masks import MasksType


class Insts:
    def __init__(
        self,
        confs: ConfsType,
        cat_ids: CatIDsType,
        bboxes: BBoxesXYXYArrType,
        masks: Optional[MasksType],
    ) -> None:
        assert len(confs) == len(cat_ids) == len(bboxes)

        if masks is not None:
            assert len(confs) == len(masks)
        
        self.confs = confs.astype(np.float32)
        self.cat_ids = cat_ids.astype(np.int32)
        self.bboxes = bboxes.astype(np.int32)

        if masks is not None:
            self.masks = masks.astype(np.bool_)
        else:
            self.masks = None
    
    def __len__(self) -> int:
        return len(self.confs)

    def __getitem__(
        self, 
        item: Union[int, List[int], slice, npt.NDArray[np.integer]]
    ) -> "Insts":
        if isinstance(item, int):
            item = [item]
        
        new_confs = self.confs[item]
        new_cat_ids = self.cat_ids[item]
        new_bboxes = self.bboxes[item, :]

        if self.masks is not None:
            new_masks = self.masks[item, :]
        else:
            new_masks = None

        new_insts = Insts(
            new_confs, 
            new_cat_ids, 
            new_bboxes, 
            new_masks
        )

        return new_insts