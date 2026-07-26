import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Literal, Union

import src.mlops.dataset.typing as tp_lib


_IMG_EXTS = [".jpg", ".png", ".jpeg", ".bmp", ".JPG", ".PNG"]


def labelme2coco_bbox_shape(
    points: tp_lib.BBoxLabelmeT
) -> tp_lib.BBoxCocoT:
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # 计算左上角和右下角坐标
    x1 = float(min(x_coords))
    y1 = float(min(y_coords))
    x2 = float(max(x_coords))
    y2 = float(max(y_coords))
    
    w = x2 - x1
    h = y2 - y1
    
    # COCO bbox 格式: (x1, y1, w, h)
    bbox: tp_lib.BBoxCocoT = (x1, y1, w, h)

    return bbox


def labelme2coco_bbox_file(
    labelme_img_p: str,
    labelme_file: Union[str, tp_lib.LabelmeFileT],
    cat_name_id_dict: dict[str, int]
) -> list[tp_lib.CocoAnnT]:
    """
    ```
    将 Labelme 的 BBox JSON 标注文件转换为 COCO 格式的标注列表。
    
    参数:
    - labelme_img_p: 图片路径，用于生成唯一的 image_id
    - labelme_file: Labelme JSON 文件路径或已解析的字典数据
        
    返回:
    - list[CocoAnnT]: 包含所有转换后 bbox 标注的列表

    注意：
    - 将忽略 polygon 标注
    - coco segmentation 为 bbox 的四个角点
    ```
    """
    # 1. 加载 Labelme 数据
    if isinstance(labelme_file, str):
        with open(labelme_file, 'r', encoding='utf-8') as f:
            data: tp_lib.LabelmeFileT = json.load(f)
    else:
        data = labelme_file
        
    label2id = cat_name_id_dict
            
    coco_anns: list[tp_lib.CocoAnnT] = []
    
    # 使用图片路径的哈希值作为 image_id (取模防止过大)
    image_id = abs(hash(labelme_img_p)) % (10**8)
    
    # 3. 遍历 shapes 进行转换
    for idx, shape in enumerate(data['shapes']):
        # 仅处理矩形框 (rectangle)
        if shape['shape_type'] != 'rectangle':
            continue
            
        points = shape['points']
        bbox = labelme2coco_bbox_shape(points)

        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        area = int(w * h)
        
        # COCO segmentation 对于 bbox 通常用四个角点表示的多边形
        # 格式: [x1, y1, x2, y1, x2, y2, x1, y2]
        segmentation: list[tp_lib.PolyCocoT] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
        
        label = shape['label']
        
        # 4. 组装 CocoAnnT
        ann: tp_lib.CocoAnnT = {
            "id": idx + 1,  # annotation 的唯一 id
            "iscrowd": 0,   # Labelme 中无此概念，默认为 0
            "image_id": image_id,
            "category_id": label2id[label],
            "area": area,
            "bbox": bbox,
            "segmentation": segmentation
        }
        coco_anns.append(ann)
        
    return coco_anns


def labelme2coco_bbox_dataset(
    labelme_img_root: str,
    labelme_file_root: str,
    coco_dir: str,
    # coco_file_p: str,
    # coco_img_root: str,
    img_dir_mode: Literal["flat", "org_dir"],
    img_copy_mode: Literal["copy", "symbolic", "link_file"],
    cat_name_id_dict: dict[str, int]
) -> None:
    """
    ```
    将指定目录下的所有 Labelme BBox 标注及其对应图片转换为 COCO 数据集格式。
    
    参数:
    - labelme_file_root: Labelme JSON 文件根路径
    - labelme_img_root: 对应的图片根目录，与 JSON 目录结构相同
    - coco_file_p: 导出的 COCO JSON 文件路径
    - coco_img_root: 导出的 COCO 图片根路径
    - img_dir_mode: 图片目录模式 ("flat" 平铺重命名, "org_dir" 保持原目录结构)
    - img_copy_mode: 图片拷贝模式 ("copy" 硬拷贝, "symbolic" 软链接, "link_file" 生成路径txt)
    
    注意:
    - 将会忽略不在 cat_name_id_dict 里的 category
    ```
    """
    labelme_file_root = Path(labelme_file_root)
    labelme_img_root = Path(labelme_img_root)
    coco_dir = Path(coco_dir)
    coco_file_p = Path(coco_dir) / "coco.json"
    coco_img_root = Path(coco_dir) / "images"
    
    # 创建输出目录
    if os.path.exists(str(coco_dir)):
        shutil.rmtree(coco_dir)
    
    os.makedirs(coco_dir, exist_ok = True)

    # coco_img_root.mkdir(parents=True, exist_ok=True)
    # coco_file_p.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化 COCO 数据集结构
    coco_images: list[tp_lib.CocoImgT] = []
    coco_annotations: list[tp_lib.CocoAnnT] = []
    coco_categories: list[tp_lib.CocoCatT] = [
        {v: k} for k, v in cat_name_id_dict.items()
    ]
    
    img_idx = 0
    ann_idx = 0
    
    # 递归遍历所有 json 文件
    json_files = sorted(labelme_file_root.rglob("*.json"))
    
    for json_p in json_files:
        # 1. 读取 Labelme JSON
        with open(json_p, 'r', encoding='utf-8') as f:
            data: tp_lib.LabelmeFileT = json.load(f)
            
        # 2. 匹配对应的图片
        rel_json_p = json_p.relative_to(labelme_file_root)
        rel_dir = str(rel_json_p.parent)
        src_img_p = None
        
        # 尝试常见的图片后缀
        for ext in _IMG_EXTS:
            p = labelme_img_root / rel_json_p.with_suffix(ext)
            if p.exists():
                src_img_p = p
                break
                
        if src_img_p is None:
            # Fallback: 使用 JSON 中的 imagePath
            img_name = data.get('imagePath', json_p.stem)
            src_img_p = labelme_img_root / rel_json_p.parent / img_name
            if not src_img_p.exists():
                print(f"Warning: Image not found for {json_p}")
                continue
                
        # 3. 调用转换函数获取 annotations
        anns = labelme2coco_bbox_file(str(src_img_p), data, cat_name_id_dict)
        
        # 提取所有的 rectangle shapes 用于同步获取 label
        rect_shapes = [s for s in data['shapes'] if s['shape_type'] == 'rectangle']
        
        # 4. 处理图片导出
        rel_img_p = src_img_p.relative_to(labelme_img_root)
        
        if img_dir_mode == "flat":
            new_img_name = f"{img_idx}{src_img_p.suffix}"
            dst_img_p = coco_img_root / new_img_name
            coco_file_name = new_img_name
        elif img_dir_mode == "org_dir":
            dst_img_p = coco_img_root / rel_img_p
            coco_file_name = rel_img_p.as_posix()  # COCO 规范使用 '/'
        else:
            raise ValueError(f"Unknown img_dir_mode: {img_dir_mode}")
            
        dst_img_p.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理图片拷贝/链接逻辑
        if img_copy_mode == "copy":
            shutil.copy2(src_img_p, dst_img_p)
        elif img_copy_mode == "symbolic":
            if dst_img_p.exists() or dst_img_p.is_symlink():
                dst_img_p.unlink()
            # 使用绝对路径创建软链接，避免相对路径计算错误
            os.symlink(os.path.abspath(src_img_p), dst_img_p)
        elif img_copy_mode == "link_file":
            # 生成和图片名字相同的 txt 文件 (例如: img.jpg -> img.txt)
            txt_p = dst_img_p.with_suffix(".txt")
            with open(txt_p, 'w', encoding='utf-8') as f_txt:
                # 写入原始图片相对于 labelme_img_root 的相对路径
                f_txt.write(rel_img_p.as_posix())
        else:
            raise ValueError(f"Unknown img_copy_mode: {img_copy_mode}")
            
        # 5. 构建 CocoImgT
        img_h = data.get('imageHeight', 0)
        img_w = data.get('imageWidth', 0)
        
        coco_img: tp_lib.CocoImgT = {
            "id": img_idx + 1,
            "file_name": coco_file_name,
            "height": img_h,
            "width": img_w
        }
        coco_images.append(coco_img)
        
        # 6. 更新 annotations 的 ID 和 category_id，并收集 categories
        for ann, shape in zip(anns, rect_shapes):
            label = shape['label']
                
            ann_idx += 1
            # 覆盖局部 ID 为全局唯一 ID
            ann['id'] = ann_idx
            ann['image_id'] = img_idx + 1
            ann['category_id'] = cat_name_id_dict[label]
            
            coco_annotations.append(ann)
            
        img_idx += 1

    # 7. 组装最终的 COCO JSON 并保存
    coco_dataset: tp_lib.CocoFileT = {
        "images": coco_images,
        "categories": coco_categories,
        "annotations": coco_annotations
    }
    
    with open(coco_file_p, 'w', encoding='utf-8') as f:
        json.dump(coco_dataset, f, ensure_ascii=False)
    
    msg = f"""
    Conversion completed.
    Total images: {len(coco_images)}
    Total annotations: {len(coco_annotations)}
    Latest image id: {img_idx}
    Latest ann id: {ann_idx}
    """
    msg = textwrap.dedent(msg)
    print(msg)