# 📦 数据集构建规范（Dataset Building）

- **概述**: 当**为新项目设计或实现数据集 pipeline**（单数据集格式转换、多数据集汇总 yaml、对接 QuickStart 训练脚本）时，按本 Skill 的两步抽象流程执行；具体目录命名与输出格式须与用户确认后再落地。
- **日期**: 2026-06-16

---

## 0 核心抽象（Two-Step Pipeline）

所有视觉训练项目的数据集构建，统一拆为 **两步、两层目录**：

```markdown
Step 1  单数据集适配     1-data-process/              一个源文件夹 → 一种训练任务可读格式
Step 2  多数据集汇总     data/<task>/create_data.py   多个已适配文件夹 → 一个训练 yaml
```

```mermaid
flowchart LR
  A[images + 原始标签] --> B[1-data-process 脚本]
  B --> C[单数据集标准目录]
  C --> D[data/create_data.py]
  D --> E[0-xxx.yaml]
  E --> F[0-QuickStart 训练脚本]
```

- **Step 1 粒度**：始终针对**单个数据集根目录**操作（一对 `images/` + 标签子目录）。
- **Step 2 粒度**：扫描**数据根目录下多个子文件夹**，按命名规则筛选，生成**统一 yaml** 供 `0-QuickStart/*.sh` 读取。
- **注意**：不同训练任务（检测 / 分割 / grounding / 分类等）可各有 Step 1 脚本与 Step 2 输出 yaml，互不混用。

---

## 1 Step 1：单数据集适配（1-data-process/）

### 1.1 输入约定（源数据最小结构）

用户通常只提供「图像 + 已清洗标签」，标签目录取决于上游标注工具，**必须与 `images/` 内文件一一对应**（同名 stem 或可追溯映射）。

常见输入形态（任选其一，或组合）：

```markdown
├── path/to/single-dataset/       # 单数据集根目录（Step 1 的 --path / Path）
│   ├── images/                   # 图像 [.jpg/.png/...]
│   │   ├── xx01.jpg
│   │   └── xx02.jpg
│   └── labels/                   # 标签 [.txt / 其他]
│       ├── xx01.txt
│       └── xx02.txt
```

```markdown
├── path/to/single-dataset/
│   ├── images/
│   └── annotations/              # VOC [.xml] 等
│       ├── xx01.xml
│       └── xx02.xml
```

```markdown
├── path/to/single-dataset/
│   ├── images/
│   ├── annotations/              # 中间格式（可选）
│   └── labels/                   # 已转换格式（可选，可跳过部分 Step 1 逻辑）
```

- **(1)** 实施前先确认：**标签目录名**、**扩展名**、**与图像的配对规则**（stem 一致 / 映射表 / 子目录规则）。
- **(2)** 若标签已是目标格式，Step 1 可退化为「校验 + 生成 train/val 索引」，不必重复转换。

### 1.2 输出约定（单数据集标准目录）

Step 1 的输出必须满足：**当前项目训练代码能直接消费**，且**产物写回该数据集目录或用户指定目录**（便于 Step 2 扫描）。

输出内容因任务而异，但通常包含：

| 产物 | 用途 |
|------|------|
| 项目格式标签 | 训练器读取的 annotation |
| `train.txt` / `val.txt` | YOLO 类任务样本索引（可选） |
| `*_segm.json` + `.cache` | Grounding 类中间产物（可选） |
| 校验日志 | 缺失图像、空标签、stem 不匹配 |

示例（多任务并存于同一仓库时，用**不同 Step 1 脚本**分别产出）：

```markdown
# 任务 A：YOLO 分割
├── GEOAI-<name>/                 # 命名规则见 §3，与用户确认
│   ├── images/
│   ├── labels/                   # YOLO seg txt
│   ├── train.txt
│   └── val.txt

# 任务 B：Grounding
├── GEOAI-<name>-GD/
│   ├── images/
│   ├── jsons/                    # 源 COCO json（可选保留）
│   ├── <project>_segm.json
│   └── <project>_segm.cache
```

### 1.3 目录与脚本规范

```
1-data-process/
├── 4-create_<task_a>.sh          # Shell 入口：变量区 + conda + 调 Python
├── 5-create_<task_b>.sh
└── util/
    ├── create_<task_a>.py        # 单数据集转换逻辑
    └── create_<task_b>.py
```

- **Shell 脚本**：顶部声明 `Path`（建议绝对路径）、`split_ratio`、`project` 等；风格对齐 `standard-quickstart` / 现有 `4-create_yolodata.sh`。
- **Python 脚本**：`argparse` + `if __name__ == "__main__"`；文件头对齐 `standart-create_script` Skill。
- **编号**：按任务递增 `4-`、`5-`…，与 QuickStart 序号独立。

### 1.4 Step 1 实施检查清单

- [ ] 与用户确认：源标签格式、目标训练任务、输出目录树
- [ ] 图像与标签 stem 一一对应；无法配对的样本有明确跳过/报错策略
- [ ] 需要 train/val 时，划分方式与比例可配置（`split_ratio` 或固定列表）
- [ ] 路径写入产物时用**绝对路径**（或文档说明相对路径的 cwd 要求）
- [ ] 脚本 idempotent：重复运行结果一致，不破坏原始 `images/` 与源标签

---

## 2 Step 2：多数据集汇总（data/）

### 2.1 输入约定

Step 2 扫描的是**数据根目录**（如 `testdir/`、`/data/0-data/xxx/`），而非单个数据集内部：

```markdown
├── path/to/trainvalset/          # Step 2 的 path 列表项
│   ├── GEOAI-<name_a>/           # Step 1 已适配的数据集 A
│   ├── GEOAI-<name_b>/           # Step 1 已适配的数据集 B
│   └── GEOAI-<name_c>-GD/        # 另一种任务后缀
```

- 通过**文件夹命名规则**筛选（前缀 / 后缀 / 必需文件），而不是手动列举每个路径。
- **注意**：同一任务类型共用一套命名规则；不同任务类型用不同 `create_data.py` 与不同 yaml 前缀。

### 2.2 输出约定（训练 yaml）

```
data/
├── yolo/
│   ├── create_data.py            # 扫描 + 生成 0-{project}.yaml
│   └── 0-Person.yaml             # 输出：train / val / names / nc
└── grounding/
    ├── create_data.py
    └── 0-mixed.yaml              # 输出：train.grounding_data 列表
```

yaml 必须对齐 **QuickStart 训练脚本** 的读参方式，例如：

```yaml
# YOLO 类（示例）
names: [...]
nc: 1
train: [/abs/path/to/train.txt, ...]
val:   [/abs/path/to/val.txt, ...]
```

```yaml
# Grounding 类（示例）
train:
  grounding_data:
    - img_path: /abs/path/to/images
      json_file: /abs/path/to/xxx_segm.json
```

### 2.3 create_data.py 逻辑模板

**(1)** 配置区（脚本底部或 argparse）：

- **path**：数据根目录列表
- **project**：输出 yaml 前缀 `0-{project}.yaml`
- **筛选函数**：如 `startswith("GEOAI") and endswith("GD")`

**(2)** 扫描逻辑：

- 遍历 `path` 下子目录 → 命名规则过滤 → 检查必需文件（yaml / cache / train.txt 等）→ 写入条目
- 无有效条目时 **raise** 并打印清晰错误

**(3)** 输出：

- 写入 `data/<task>/0-{project}.yaml`
- 可选：统计 yaml（样本数、数据集个数）、版本信息 yaml（参考现有 `1-Dataset-stat.yaml` 模式）

### 2.4 Step 2 实施检查清单

- [ ] 命名规则与用户文档（README §数据集）一致
- [ ] yaml 内路径为绝对路径，或明确训练时 cwd
- [ ] 与 `0-QuickStart/0-train*.sh` 中 `--data` / `--grounding-data` 字段一一对应
- [ ] 重复数据集（同内容不同目录名）在文档或脚本注释中提醒用户去重

---

## 3 与用户确认的问题清单（实施前必问）

实施具体项目前，用以下问题锁定 Step 1 / Step 2 形态（可逐条 AskQuestion 或 README 中让用户填写）：

| # | 问题 | 影响 |
|---|------|------|
| 1 | 源标签格式是什么？（txt / xml / json / jsonl） | Step 1 解析器 |
| 2 | 训练任务类型？（YOLO det/seg / grounding / 其他） | Step 1 脚本个数与产物 |
| 3 | 单数据集目录命名规则？（如 `GEOAI-<name>` / `-GD` 后缀） | Step 2 扫描 |
| 4 | 是否需要 train/val 划分？比例或固定列表？ | Step 1 |
| 5 | 多数据集汇总 yaml 名称？（如 `0-Person.yaml` / `0-mixed.yaml`） | Step 2 输出 |
| 6 | 训练入口读哪个 yaml？（PE 仅 yolo / open 需 yolo + grounding） | QuickStart 变量 |
| 7 | 验证集用什么？（自有 val / 官方 lvis 等） | scratch 模式 dict |

- **注意**：未确认前只写通用骨架，**不要**擅自假定目录名或 yaml 字段。

---

## 4 与 QuickStart / README 的衔接

### 4.1 训练脚本变量映射

| QuickStart 变量 | 来源 | 对应 Step |
|-----------------|------|-----------|
| `dataset` / `yolo_dataset` | `data/yolo/0-{project}.yaml` | Step 2 |
| `grounding_dataset` | `data/grounding/0-{project}.yaml` | Step 2 |
| `Path` in `4-create_*.sh` | 用户单数据集根目录 | Step 1 |

### 4.2 README 文档结构

数据集章节按 **standard-create_readme** 规范，且**按任务拆分**（PE §2 / Grounding §3 等），每节包含：

- 源数据结构（§1.1 类图）
- Step 1 命令与产物树
- Step 2 命令与 yaml 说明
- 「有这个就可以训练了✅」总结构图

---

## 5 参考实例（YOLOE 仓库，非通用约束）

本仓库已落地示例，抽象时可对照，但不要硬编码到其他项目：

| 步骤 | 任务 | 脚本 | 输出 |
|------|------|------|------|
| Step 1 | YOLO seg / PE | `4-create_yolodata.sh` | `labels/`, `train.txt`, `val.txt` |
| Step 1 | Grounding | `5-create_grounding.sh` | `*_segm.json`, `*.cache` |
| Step 2 | YOLO | `data/yolo/create_data.py` | `0-Person.yaml` |
| Step 2 | Grounding | `data/grounding/create_data.py` | `0-mixed.yaml` |

---

## 6 Agent 执行顺序（Checklist）

新建或改造某项目的数据 pipeline 时，按序执行：

1. **读**本 Skill + 用户训练入口（`0-QuickStart/0-train*.sh`）+ 训练器读 data 的代码路径
2. **问**§3 确认问题（结构/命名/任务类型）
3. **实现 Step 1**：`1-data-process/util/*.py` + `N-create_*.sh`，单数据集跑通
4. **实现 Step 2**：`data/<task>/create_data.py`，多文件夹扫描 + yaml
5. **接 QuickStart**：shell 变量指向 yaml；必要时在 `train_*.py` 合并多 yaml
6. **写 README** 数据集章节（standard-create_readme + GroundingDINO 风格）
7. **冒烟**：至少 1 个样本完成 Step 1 → Step 2 → 训练脚本 `--help` 或 dry-run

---

## 7 反模式（避免）

- ❌ 在 Step 2 里做标签格式转换（应放在 Step 1）
- ❌ 一个 `create_data.py` 混扫 PE 与 Grounding 目录且无命名区分
- ❌ yaml 仅写相对路径但未文档化训练 cwd
- ❌ 未确认命名规则就写死 `GEOAI-*-GD` 到无关项目
- ❌ Step 1 输出 jsonl 等中间格式，但训练器实际只读 cache / yaml（中间格式需注明「仅人工排查」）

---

# 🎯 Done
