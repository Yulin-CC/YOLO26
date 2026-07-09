# 🚀 YOLO26 / 使用指南

基于 Ultralytics 上游源码的 **YOLO26 检测与分割** 项目，通过 `task` 变量切换任务，日常入口统一在 `0-QuickStart/`。

## 更新日志

- [x] 2026-07-09
  - 统一 `task=detect|segment` 切换，训练 / 推理 / 评估脚本共用同一套入口
  - `1-create_dataset.sh` 支持 `task` + `platform`（labelimg / labelme）标签转换

---

- [1 环境搭建](#1-环境搭建)
- [2 快速流程](#2-快速流程)
- [3 进阶说明](#3-进阶说明)

---

## 1 环境搭建

```bash
conda create -n yolo python=3.10 -y
conda activate yolo

cd /path/to/ultralytics/
ln -sf z_others/pyproject.toml pyproject.toml
ln -sf z_others/LICENSE LICENSE
pip install -e .
```

**验证安装：**

```bash
python -c "from ultralytics import YOLO; m=YOLO('yolo26n.pt'); print('OK')"
```

---

## 2 快速流程

所有脚本顶部均有 `task=detect` 或 `task=segment`，改这一处即可切换任务；`model`、`dataset`、输出路径会随之自动适配。

### 2.1 推理 / 验证

```bash
# 编辑 0-QuickStart/*.sh，设置 task = detect | segment; weight = yolo26s.pt | yolo26s-seg.pt
bash 0-QuickStart/1-inference.sh
bash 0-QuickStart/2-eval.sh
```

### 2.2 训练

### 2.2.1 数据集准备

数据集目录约定：

```
YOUR_DATA_DIR/
├── images/
├── labels/              # 转换输出
├── annotations/         # labelimg：VOC XML
└── jsons-labelme/       # labelme：JSON 标注
```

编辑 `1-data-process/1-create_dataset.sh` 中的 `Path`、`task`、`platform`、`yaml` 后运行：

```bash
bash 1-data-process/1-create_dataset.sh
```


| task    | platform | 转换脚本                                 | 标签格式                            |
| ------- | -------- | ------------------------------------ | ------------------------------- |
| detect  | labelimg | `convert_xml2txt.py`                 | VOC XML → 检测 bbox txt           |
| detect  | labelme  | `convert_json2txt.py --task detect`  | LabelMe rectangle → 检测 bbox txt |
| segment | labelme  | `convert_json2txt.py --task segment` | LabelMe polygon → 分割 txt        |


训练输出：

```
runs/{task}/0-train/{project}/
├── configs/          # 配置快照（default.yaml + data/*.yaml）
├── weights/
│   ├── best.pt
│   └── last.pt
└── results.csv
```



### 2.4 模型导出（可选）

```bash
bash 0-QuickStart/3-export.sh         # 导出 ONNX
```

---



## 3 进阶说明



### 3.1 项目结构

```
ultralytics/
├── 0-QuickStart/              # 训练 / 推理 / 评估 / 导出入口
├── 1-data-process/            # 数据集整理与格式转换
│   ├── 1-create_dataset.sh            # 数据集一键整理（task + platform）
│   └── utils/
│       ├── 0-organize_dataset.py    # 格式转换 → 校验 → 自动整理
│       ├── validate.py              # 数据集一致性校验
│       ├── convert_xml2txt.py       # labelimg → 检测 txt
│       ├── convert_json2txt.py      # labelme → 检测 bbox / 分割 polygon txt
│       ├── yolo_merge.py            # 对齐 images/labels
│       ├── yolo_rename.py           # 批量重命名
│       └── yolo_split.py            # 划分 train.txt / val.txt
├── config/default.yaml        # 含 task 与默认超参
├── data/                      # 数据集 yaml
├── weights/                   # yolo26*.pt / yolo26*-seg.pt
├── runs/
│   ├── detect/                # 检测任务输出
│   └── segment/               # 分割任务输出
├── ultralytics/               # 上游核心代码包
└── z_others/                  # 官方文档与配置归档
```

---



## 参考

- 上游仓库：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- YOLO26 文档：[docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26)
- 官方原始 README：`z_others/README.md`、`z_others/README.zh-CN.md`
- 业务参考：`../yolo26/yolo26-det/`（检测）、`../yolo26/yolo26-seg/`（分割）

