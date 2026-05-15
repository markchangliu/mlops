# Architecture-01

## 代码结构

```
src/mlops
| - __init__.py         # 公共 API 入口
| - core
| - | - __init__.py
| - | - schema.py       # 数据定义
| - | - instances.py    # 检测/分割实例类
| - convert
| - | - __init__.py
| - | - shapes.py       # bbox、mask、poly 等单个标注数据之间的转化
| - | - labelme.py      # labelme <-> 内部数据集
| - | - coco.py         # coco <-> 内部数据集
| - | - yolo.py         # yolo <-> 内部数据集
| - | - cli.py          # 通用转换 API
| - eval
| - | - __init__.py
| - | - iou.py          # iou 计算
| - | - match.py        # GT 和 PRED 匹配，判断 TP、FP、FN
| - | - metrics.py      # mAP、precision、recall 指标计算
| - | - test.py         # 推理结果评估
| - visualize
| - | - __init__.py
| - | - shapes.py       # bbox、mask、poly 等单个标注数据的可视化
| - | - datasets.py     # 数据集可视化
| - | - results.py      # TP、FP、FN 等评估结果的可视化
```

## 依赖方向

### 模块间的依赖方向

```
convert, eval, visualize <- core
```

### 模块内部的依赖方向

```
# core 模块
instances.py <- schema.py

# convert 模块
labelme.py, coco.py, yolo.py <- shapes.py
cli.py <- labelme.py, coco.py, yolo.py

# eval 模块
match.py <- iou.py
metrics.py <- match.py
test.py <- metrics.py, match.py

# visualize 模块
datasets.py, results.py <- shapes.py
```
