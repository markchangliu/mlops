#  MLOPS

## Introduction

一个用于开发视觉算法的标准和辅助工具。

辅助工具有以下功能：
- Data Management: 标注转化、数据预处理、数据迭代、数据集生成

## Directory Architecture

### 总览

```
large_files
| - data
| - | - {YYYYMMDD_XXX}
| - | - | - {XX}.png/jpg/jpeg/bmp
| - ckpts
| - | - Qwen3-VL-2B
| - | - | - {files}
| - solutions
| - | - v0.0.0
| - | - | - seg_blister
| - | - | - | - v0.0.0
| - | - | - | - | - datasets
| - | - | - | - | - | - raw
| - | - | - | - | - | - | - {YYYYMMDD_XXX}
| - | - | - | - | - | - | - | - {XX}.png/jpg/jpeg/bmp
| - | - | - | - | - | - | - | - {XX}.json
| - | - | - | - | - | - | - | - {XX}.path
| - | - | - | - | - | - | - | - raw_meta.csv
| - | - | - | - | - | - coco
| - | - | - | - | - | - | - {split}
| - | - | - | - | - | - | - | - images
| - | - | - | - | - | - | - | - | - {XX}.png/jpg/jpeg/bmp
| - | - | - | - | - | - | - | - paths
| - | - | - | - | - | - | - | - | - {XX}.path
| - | - | - | - | - | - | - | - coco.json
| - | - | - | - | - | - yolo
| - | - | - | - | - | - | - {split}
| - | - | - | - | - | - | - | - images
| - | - | - | - | - | - | - | - | - {XX}.png/jpg/jpeg/bmp
| - | - | - | - | - | - | - | - paths
| - | - | - | - | - | - | - | - | - {XX}.path
| - | - | - | - | - | - | - | - labels
| - | - | - | - | - | - | - | - | - {XX}.txt
| - | - | - | - | - | - split_cfg.json
| - | - | - | - | - | - label_rules.md
| - | - | - | - | - runs
| - | - | - | - | - | - cfg1
| - | - | - | - | - | - | - train
| - | - | - | - | - | - | - test1
| - | - | - | - | - | - | - test2
| - | - | - | - | - | - | - description.md
| - | - | - | - | - evolve_log.md
| - | - | - | - | - best_manifest.json
```

### `split_cfg.json`

``` json
# split.json

{
    "split_name": {

        "data_list": [
            "batch1",
            "batch2",
            "..."
        ],

        "seed": 13,

        "shuffle": true,

        "shuffle_mode": "mix_folder" or "separate_folder",

        "include_ratio": 0.8,

        "filter": "all",

        "filter": {
            "tag_name1": "all",
            "tag_name2": ["val1_*", "val2_*"],
            "mode": "and" or "or"
        },

        "export_name_mode": "reindex" or "original"
    }
}
```

