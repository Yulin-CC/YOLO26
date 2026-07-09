"""
# @Author: AI产品研发组
# @Date: 2026-07-09
# @Description: LabelMe JSON -> YOLO txt（detect=bbox / segment=polygon）
# @Command: python 1-data-process/utils/convert_json2txt.py --dataset /path/to/data --yaml classes.yaml --task detect
"""

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

SUPPORTED_TASKS = ("detect", "segment")


def read_labelme(json_path: str | Path) -> dict:
    """读取 LabelMe JSON，返回 imagePath / 尺寸 / shapes。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "imagePath": data.get("imagePath", ""),
        "imageHeight": data["imageHeight"],
        "imageWidth": data["imageWidth"],
        "shapes": [
            {
                "label": s["label"],
                "points": s["points"],
                "shape_type": s.get("shape_type", "polygon"),
            }
            for s in data.get("shapes", [])
            if s.get("points")
        ],
    }


def write_txt(txt_path: str | Path, instances: list[list[float]]) -> None:
    """写入 YOLO txt，每行一个实例。"""
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for obj in instances:
            f.write(" ".join(f"{x:.6g}" for x in obj) + "\n")


def _load_names(yaml_path: str | Path) -> list:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def _shape_to_detect(shape: dict, w: int, h: int, name2id: dict) -> list[float] | None:
    """LabelMe rectangle/4-point polygon -> YOLO detect: class_id cx cy bw bh（归一化）。"""
    label = shape["label"]
    if label not in name2id:
        return None

    shape_type = shape.get("shape_type", "")
    points = shape["points"]
    # 兼容 shape_type="polygon" 但只有 4 个点的矩形框
    if shape_type not in ("rectangle", "", "polygon"):
        return None
    if len(points) not in (2, 4):
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin or ymax <= ymin:
        return None

    return [
        name2id[label],
        ((xmin + xmax) / 2) / w,
        ((ymin + ymax) / 2) / h,
        (xmax - xmin) / w,
        (ymax - ymin) / h,
    ]


def _shape_to_segment(shape: dict, w: int, h: int, name2id: dict) -> list[float] | None:
    """LabelMe polygon -> YOLO segment: class_id x1 y1 x2 y2 ...（归一化）。"""
    label = shape["label"]
    if label not in name2id:
        return None

    shape_type = shape.get("shape_type", "polygon")
    points = shape["points"]
    # rectangle 和 4 点 polygon（本质是 bbox）都跳过，留给 detect 模式处理
    if shape_type == "rectangle" or len(points) < 3 or len(points) == 4:
        return None

    norm = [name2id[label]]
    for x, y in points:
        norm.extend([x / w, y / h])
    return norm


def labelme2txt(
    json_dir: str | Path,
    txt_dir: str | Path,
    yaml_path: str | Path,
    task: str = "segment",
) -> dict:
    """
    LabelMe 目录 -> YOLO txt 目录。
    detect: 仅 rectangle（bbox）；segment: 仅 polygon（分割）。
    """
    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}', choose from {SUPPORTED_TASKS}")

    json_dir = Path(json_dir)
    txt_dir = Path(txt_dir)
    name2id = {n: i for i, n in enumerate(_load_names(yaml_path))}
    txt_dir.mkdir(parents=True, exist_ok=True)
    shape_fn = _shape_to_detect if task == "detect" else _shape_to_segment

    stats = {"total": 0, "ok": 0, "skip_unknown_label": 0, "skip_shape": 0}
    for jp in tqdm(list(json_dir.glob("*.json")), desc=f"labelme2txt-{task}"):
        stats["total"] += 1
        data = read_labelme(jp)
        w, h = data["imageWidth"], data["imageHeight"]
        instances = []
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in name2id:
                stats["skip_unknown_label"] += 1
                continue
            row = shape_fn(shape, w, h, name2id)
            if row is None:
                stats["skip_shape"] += 1
                continue
            instances.append(row)
        write_txt(txt_dir / f"{jp.stem}.txt", instances)
        stats["ok"] += 1
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LabelMe JSON -> YOLO txt")
    parser.add_argument("--dataset", required=True, help="数据集根目录")
    parser.add_argument("--yaml", required=True, help="类别 yaml（含 names 字段）")
    parser.add_argument("--task", default="segment", choices=SUPPORTED_TASKS, help="detect=bbox, segment=polygon")
    parser.add_argument("--txt-dir", default=None, help="YOLO txt 输出目录，默认 {dataset}/labels")
    parser.add_argument("--json-dir", default=None, help="LabelMe json 目录，默认 {dataset}/jsons-labelme")
    return parser


def resolve_json_dir(dataset: Path, json_dir: str | Path | None) -> Path:
    if json_dir:
        return Path(json_dir)
    for dir_name in ("jsons-segment", "jsons-detect", "jsons-labelme"):
        candidate = dataset / dir_name
        if candidate.is_dir() and list(candidate.glob("*.json")):
            return candidate
    return dataset / "jsons-labelme"


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = Path(args.dataset)
    txt_dir = Path(args.txt_dir) if args.txt_dir else dataset / "labels"
    json_dir = resolve_json_dir(dataset, args.json_dir)

    stats = labelme2txt(json_dir, txt_dir, args.yaml, task=args.task)
    print(
        f"labelme2txt ({args.task}) 完成: {stats['ok']}/{stats['total']}，"
        f"未知类别 {stats['skip_unknown_label']}，形状跳过 {stats['skip_shape']}"
    )


if __name__ == "__main__":
    main()
