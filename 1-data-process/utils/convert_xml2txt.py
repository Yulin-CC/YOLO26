#!/usr/bin/env python3
"""
# @Author: 算法组
# @Date: 2026-06-30
# @Description: VOC XML → YOLO label 批量转换。
#   读取 annotations/ 下 XML 文件，生成 labels/ 下的 YOLO .txt 标注。
#
# @Command: python 1-data-process/utils/convert_xml2txt.py /path/to/dataset --classes classes.yaml
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

# ==============================
# 接口配置
# ==============================
DATA_DIR = "/path/to/dataset"
CLASSES_PATH = None  # 类别 yaml 路径，如 "../../data/0-Person.yaml"
# ==============================


def list_xml(folder: str) -> dict:
    """stem -> xml_path"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        if fn.lower().endswith(".xml"):
            out[os.path.splitext(fn)[0]] = os.path.join(folder, fn)
    return out


def load_classes(classes_path: str) -> list:
    """加载类别列表（支持 yaml names 字典/列表或纯列表）。"""
    if not classes_path or not os.path.isfile(classes_path):
        return []
    import yaml

    with open(classes_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", data)
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def xml_to_yolo_line(xml_path: str, label_mapping: list) -> str:
    """将单个 VOC XML 转为 YOLO label 文本。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    if size is None:
        return ""
    img_width = float(size.find('width').text)
    img_height = float(size.find('height').text)

    lines = []
    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is None:
            continue
        label = name_el.text
        if label not in label_mapping:
            continue
        label_id = label_mapping.index(label)

        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue

        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height

        lines.append(f"{label_id} {x_center} {y_center} {width} {height}")

    return "\n".join(lines) if lines else ""


def convert_all(data_dir: str, classes_path: str) -> dict:
    """批量 XML → YOLO。返回 {ok, total, skipped, skipped_no_match}。"""
    annot_dir = os.path.join(data_dir, "annotations")
    label_dir = os.path.join(data_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    xml_map = list_xml(annot_dir)
    img_stems = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_dir, "images"))} if os.path.isdir(os.path.join(data_dir, "images")) else set()
    existing_labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".txt")} if os.path.isdir(label_dir) else set()

    classes = load_classes(classes_path)

    stats = {"ok": 0, "total": len(xml_map), "skipped": 0, "skipped_no_match": 0}

    for stem, xml_path in sorted(xml_map.items()):
        if stem in existing_labels:
            stats["skipped"] += 1
            continue
        if stem not in img_stems:
            stats["skipped"] += 1
            continue

        content = xml_to_yolo_line(xml_path, classes)
        label_path = os.path.join(label_dir, stem + ".txt")
        with open(label_path, 'w', encoding='utf-8') as f:
            if content:
                f.write(content + "\n")
        stats["ok"] += 1

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VOC XML → YOLO label 批量转换")
    parser.add_argument("dataset", help="YOLO 数据目录路径")
    parser.add_argument("--Path", "--path", dest="data_dir", default=DATA_DIR, help="YOLO 数据目录（覆盖 dataset 参数）")
    parser.add_argument("--classes", default=CLASSES_PATH, help="类别 yaml 路径")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际转换")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # dataset 位置参数或 --path 均可指定路径
    data_dir = args.data_dir if args.data_dir != DATA_DIR else args.dataset
    classes_path = args.classes
    dry_run = args.dry_run

    print(f"数据目录：{data_dir}")
    print(f"类别文件：{classes_path or '(自动推断)'}")
    print("-" * 50)

    step, total = 1, 3

    print(f"[{step}/{total}] 扫描 XML 文件...")
    step += 1
    annot_dir = os.path.join(data_dir, "annotations")
    if not os.path.isdir(annot_dir):
        print(f"  [错误] 目录不存在: {annot_dir}")
        sys.exit(1)
    xml_count = len([f for f in os.listdir(annot_dir) if f.lower().endswith(".xml")])
    print(f"  发现 {xml_count} 个 XML 文件")

    label_dir = os.path.join(data_dir, "labels")
    if os.path.isdir(label_dir):
        existing = len([f for f in os.listdir(label_dir) if f.endswith(".txt")])
        print(f"  labels/ 已有 {existing} 个")

    print(f"[{step}/{total}] 转换 XML → YOLO...")
    step += 1
    stats = convert_all(data_dir, classes_path)

    if dry_run:
        print(f"  [预览] 将转换 {stats['ok']} / {stats['total']}")
    else:
        print(f"  成功: {stats['ok']} / {stats['total']}")
        if stats['skipped']:
            print(f"  跳过: {stats['skipped']}（已存在或无对应图像）")

    print(f"[{step}/{total}] 完成")
    print("-" * 50)
    print("完成！")


if __name__ == "__main__":
    main()
