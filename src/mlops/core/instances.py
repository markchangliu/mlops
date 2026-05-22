import copy
from typing import Union

import numpy as np
from numpy.typing import NDArray

import mlops.core.schema as schema_lib


__all__ = [
    "Instances",
    "concat"
]


class Instances:
    def __init__(
        self,
        data: schema_lib.InstancesType
    ) -> None:
        self.data = copy.deepcopy(data)
    
    @classmethod
    def from_values(
        cls,
        confs: schema_lib.ConfsSchemaType,
        cat_ids: schema_lib.CatIDsSchemaType,
        bboxes: schema_lib.BBoxesSchemaType,
        polys: Union[None, list[schema_lib.PolysSchemaType]]
    ) -> "Instances":
        data: schema_lib.InstancesType = {
            "confs": confs,
            "cat_ids": cat_ids,
            "bboxes": bboxes,
            "polys": polys
        }
        insts = cls(data)
        return insts
    
    def __len__(self) -> int:
        return len(self.data['confs'])
    
    def __getitem__(
        self,
        item: Union[int, list[int], NDArray[np.integer], NDArray[np.bool_]],
    ) -> "Instances":
        new_confs = self.data["confs"][item]
        new_cat_ids = self.data["cat_ids"][item]
        new_bboxes = self.data["bboxes"][item]

        if self.data["polys"] is None:
            new_polys = None
        else:
            if isinstance(item, np.ndarray):
                if item.dtype == np.bool_:
                    indice = np.arange(len(self))[item].tolist()
                else:
                    indice = item.tolist()
            
            new_polys = []
            for i, poly in self.data["polys"]:
                if i in indice:
                    new_polys.append(poly)

        new_data: schema_lib.InstancesType = {
            "bboxes": new_bboxes,
            "polys": new_polys,
            "cat_ids": new_cat_ids,
            "confs": new_confs
        }

        new_insts = Instances(new_data)

        return new_insts
    
    def concat(
        self, 
        others: list["Instances"]
    ) -> "Instances":
        insts_list = [self] + others
        poly_flag = True if self.data["polys"] is not None else False
        new_insts = concat(insts_list, poly_flag)
        return new_insts


def concat(
    insts_list: list[Union["Instances", schema_lib.InstancesType]],
    poly_flag: bool
) -> "Instances":
    new_confs = []
    new_cat_ids = []
    new_bboxes = []
    new_polys = [] if poly_flag else None

    for insts in insts_list:
        if isinstance(insts, dict):
            insts = Instances(insts)

        new_confs.append(insts.data["confs"])
        new_cat_ids.append(insts.data["cat_ids"])
        new_bboxes.append(insts.data["bboxes"])

        if poly_flag:
            new_polys += insts.data["polys"]
    
    new_confs = np.concat(new_confs)
    new_cat_ids = np.concat(new_cat_ids)
    new_bboxes = np.concat(new_bboxes, axis = 0)

    new_data: schema_lib.InstancesType = {
        "confs": new_confs,
        "cat_ids": new_cat_ids,
        "bboxes": new_bboxes,
        "polys": new_polys
    }

    new_insts = Instances(new_data)
    return new_insts
    