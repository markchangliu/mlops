import os

from mlops.data.convert import labelme2coco_dataset


DATA_ROOT = "/home/cliu/projects/mlops/tests/data/assets"
LABELME_DIR1 = os.path.join(DATA_ROOT, "data_labelme1")
LABELME_DIR2 = os.path.join(DATA_ROOT, "data_labelme2")
LABELME_DIR3 = os.path.join(DATA_ROOT, "data_labelme3")
LABELME_ROOTS = [LABELME_DIR1, LABELME_DIR2, LABELME_DIR3]
COCO_DIR = os.path.join(DATA_ROOT, "data_coco")

LABEL_P = os.path.join(DATA_ROOT, "labels.txt")
CAT_NAME_ID_DICT = {}
with open(LABEL_P, "r") as f:
    for cat_id, line in enumerate(f):
        cat_name = line.strip()
        CAT_NAME_ID_DICT[cat_name] = cat_id


def test_labelme2coco_dataset1() -> None:
    """
    ```
    Task
        测试以下行为是否符合预期
            add_background_img = True
            merge_shape_group = True
            flatten_reindex_img = False
            export_img_mode = "symbolic"

    Expected Result (Must Meet ALL to Pass)

        1. "data_labelme1" 和 "data_labelme2" 内的图片，
        转化后的标注列表、图片列表、类别列表
        和 "data_coco/expected1/coco_expected.json" 相同

        2. "data_labelme3" 内的图片在 "data_coco/expected1/coco_expected.json"
        内的图片列表中

        3. 转化后的 "data_coco/expected1/images" 内部路径结构和
        "data_labelme1"、"data_labelme2"、"data_labelme3" 原本的
        路径结构相同

        4. 
    ```
    """
    labelme2coco_dataset(
        labelme_roots = LABELME_ROOTS,
        coco_root = COCO_DIR,
        cat_name_id_dict = CAT_NAME_ID_DICT,
        shape_type = "poly",
        add_background_img = True,
        merge_shape_group = True,
        flatten_reindex_img = False,
        export_img_mode = "symbolic"
    )
    
