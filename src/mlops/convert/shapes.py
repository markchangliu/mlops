
import numpy as np

import mlops.core.schema as schema


__all__ = [
    "bboxLabelme2bboxSchema",
    "polyLabelme2polySchema"
]


##### BBox #####

def bboxLabelme2bboxSchema(
    bboxLabelme: schema.BBoxLabelmeType
) -> schema.BBoxSchemaType:
    bboxSchema = np.asarray(bboxLabelme).flatten()
    return bboxSchema

##### Poly #####

def polyLabelme2bboxSchema(
    polylabelme: schema.PolyLabelmeType
) -> schema.BBoxSchemaType:
    polySchema = np.asarray(polylabelme)
    x1 = np.min(polySchema[:, 0]).item()
    x2 = np.max(polySchema[:, 0]).item()
    y1 = np.min(polySchema[:, 1]).item()
    y2 = np.max(polySchema[:, 1]).item()
    bboxSchema = np.asarray([x1, y1, x2, y2])
    return bboxSchema

def polyLabelme2polySchema(
    polylabelme: schema.PolyLabelmeType
) -> schema.PolySchemaType:
    polySchema = np.asarray(polylabelme)
    return polySchema