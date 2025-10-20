import os
from pathlib import Path
import shutil

from cvproject.dataset import make_yolo_dataset


def main2() -> None:
    raw_img_root = "/mnt/c/Users/cliu/Desktop/d2l/large_files/projects/blueberry/data/raw/images"
    raw_label_root = "/mnt/c/Users/cliu/Desktop/d2l/large_files/projects/blueberry/data/datasets/v.demo-s.0.0_futian_phase03_20251016/raw_labels"

    raw_dirnames = [
        "futian_20251014_indoor_virtual_train",
        "futian_20251014_indoor_virtual_test"
    ]

    dst_data_root = "/mnt/c/Users/cliu/Desktop/d2l/large_files/projects/blueberry/data/datasets/v.demo-s.0.0_futian_phase03_20251016/dataset_labelme"
    dst_train_img_dir = os.path.join(dst_data_root, "train_all", "images")
    dst_train_labelme_dir = os.path.join(dst_data_root, "train_all", "labelmes_mask")
    dst_test_img_dir = os.path.join(dst_data_root, "test_all", "images")
    dst_test_labelme_dir = os.path.join(dst_data_root, "test_all", "labelmes_mask")

    os.makedirs(dst_train_img_dir, exist_ok=True)
    os.makedirs(dst_train_labelme_dir, exist_ok=True)
    os.makedirs(dst_test_img_dir, exist_ok=True)
    os.makedirs(dst_test_labelme_dir, exist_ok=True)

    data_id = 0

    for raw_dn in raw_dirnames:
        raw_img_dir = os.path.join(raw_img_root, raw_dn, "images")
        raw_labelme_dir = os.path.join(raw_label_root, raw_dn, "labelmes_mask")

        if "train" in raw_dn:
            dst_img_dir = dst_train_img_dir
            dst_labelme_dir = dst_train_labelme_dir
        else:
            dst_img_dir = dst_test_img_dir
            dst_labelme_dir = dst_test_labelme_dir
        
        filenames = os.listdir(raw_img_dir)
        filenames.sort()

        for fn in filenames:
            if not fn.endswith((".png", ".jpg", ".jpeg")):
                continue

            img_name = fn
            img_stem = Path(img_name).stem
            img_suffix = Path(img_name).suffix
            labelme_name = f"{img_stem}.json"
            img_p = os.path.join(raw_img_dir, img_name)
            labelme_p = os.path.join(raw_labelme_dir, labelme_name)

            if not os.path.exists(labelme_p):
                continue

            dst_img_p = os.path.join(dst_img_dir, f"{data_id}{img_suffix}")
            dst_labelme_p = os.path.join(dst_labelme_dir, f"{data_id}.json")

            shutil.copy(img_p, dst_img_p)
            shutil.copy(labelme_p, dst_labelme_p)

            data_id += 1

def main() -> None:
    make_yolo_dataset(
        "/mnt/c/Users/cliu/Desktop/d2l/large_files/projects/blueberry/data/datasets/v.demo-s.0.0_futian_phase03_20251016",
        {"mature": 0},
        "labelmes_mask",
        "poly"
    )


if __name__ == "__main__":
    main()