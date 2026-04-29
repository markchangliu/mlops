# Arch-01

## 代码结构

```
src/mlops
| - __init__.py         # 公共 API 入口
| - core
| - | - __init__.py
| - | - schema.py       # 数据契约
| - transforms
| - | - base.py         # 变换基类
| - | - resize.py       # resize 变换
| - | - crop.py         # crop 变换
| - | - pipeline.py     # 变换流水线
| - convert
| - | - shapes.py       # bbox、mask、poly 格式转化
| - | - labelme.py      # labelme <-> 内部契约
| - | - coco.py         # coco <-> 内部契约
| - | - yolo.py         # yolo <-> 内部契约
| - | - cli.py          # 通用转换 API
| - eval
| - | - iou.py          # 计算 iou
| - | - mAP.py          # 计算 mAP
| - | - precision.py    # 计算 precision
| - | - recall.py       # 计算 recall
| - visualize
| - | - __init__.py
| - | - shapes.py       # bbox、mask、polygon 可视化
| - | - datasets.py     # 数据集可视化
| - | - metrics.py      # TP、FP、FN 可视化
```

## 依赖方向

```
transforms <- core
convert <- core, transform
eval <- core
visualize <- core
```
