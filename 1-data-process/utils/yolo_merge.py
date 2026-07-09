#!/usr/bin/env python3
"""
# @Author: 算法组
# @Date: 2026-06-30
# @Description: YOLO 数据集整理：
#   1) 删除 images/labels 中未配对的文件；
#   2) 若有 annotations/ 无 labels/，自动调用 xml2yolo 转换；
#   3) 同步 train.txt / val.txt，仅保留 images/labels 均存在的样本。
#
# @Command: python 1-data-process/utils/yolo_merge.py /path/to/dataset
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

# ==============================
# 接口配置
# ==============================
DATA_DIR = "/path/to/dataset"
# ==============================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_EXT = {".txt"}


def list_images(folder: str) -> dict:
    """stem -> filename"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in IMAGE_EXTS:
            out[stem] = fn
    return out


def list_labels(folder: str) -> dict:
    """stem -> filename"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in LABEL_EXT:
            out[stem] = fn
    return out


def list_xml(folder: str) -> dict:
    """stem -> xml_path"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        if fn.lower().endswith(".xml"):
            out[os.path.splitext(fn)[0]] = os.path.join(folder, fn)
    return out


def read_split(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def sync_split_list(path: str, valid_names: set) -> tuple:
    """重写 train.txt / val.txt，仅保留 valid_names 中的条目。"""
    if not os.path.isfile(path):
        return 0, 0
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    kept = [ln for ln in lines if os.path.basename(ln) in valid_names]
    removed = len(lines) - len(kept)
    with open(path, "w", encoding="utf-8") as f:
        if kept:
            f.write("\n".join(kept) + "\n")
    return len(kept), removed


def clean_unpaired(data_dir: str) -> tuple:
    """删除未配对的 images/labels，返回 (removed_img, removed_label)。"""
    img_map = list_images(os.path.join(data_dir, "images"))
    label_map = list_labels(os.path.join(data_dir, "labels"))

    keep_stems = set(img_map.keys()) & set(label_map.keys())
    img_only = set(img_map.keys()) - keep_stems
    label_only = set(label_map.keys()) - keep_stems

    removed_img = 0
    for stem in img_only:
        os.remove(os.path.join(data_dir, "images", img_map[stem]))
        removed_img += 1

    removed_label = 0
    for stem in label_only:
        os.remove(os.path.join(data_dir, "labels", label_map[stem]))
        removed_label += 1

    return removed_img, removed_label


def xml2yolo_single(xml_path: str, label_path: str, label_mapping: dict) -> int:
    """将单个 VOC XML 转为 YOLO label。返回生成的 object 数量。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    if size is None:
        return 0
    img_width = float(size.find('width').text)
    img_height = float(size.find('height').text)

    cls_names = label_mapping.get('names', [])
    if not cls_names:
        cls_names = label_mapping  # 兼容纯列表

    objects = []
    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is None:
            continue
        label = name_el.text
        if label not in cls_names:
            continue
        label_id = cls_names.index(label)

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

        objects.append(f"{label_id} {x_center} {y_center} {width} {height}")

    with open(label_path, 'w', encoding='utf-8') as f:
        if objects:
            f.write("\n".join(objects) + "\n")

    return len(objects)


def convert_xml_to_yolo(data_dir: str, label_mapping: dict) -> tuple:
    """批量将 annotations/ XML 转为 labels/ YOLO。返回 (ok, total)。"""
    annot_dir = os.path.join(data_dir, "annotations")
    label_dir = os.path.join(data_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    xml_map = list_xml(annot_dir)
    img_map = list_images(os.path.join(data_dir, "images"))
    label_map = list_labels(label_dir)

    existing_stems = set(label_map.keys())
    to_convert = {s: p for s, p in xml_map.items() if s not in existing_stems and s in img_map}

    if not to_convert:
        return 0, len(xml_map)

    ok = 0
    for stem, xml_path in to_convert.items():
        label_path = os.path.join(label_dir, stem + ".txt")
        n = xml2yolo_single(xml_path, label_path, label_mapping)
        if n >= 0:  # 即使 0 个对象，空文件也是有效的
            ok += 1

    return ok, len(xml_map)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO 数据集整理：对齐 + XML 转换 + 同步列表")
    parser.add_argument("dataset", help="YOLO 数据目录路径")
    parser.add_argument("--Path", "--path", dest="data_dir", default=DATA_DIR, help="YOLO 数据目录（覆盖 dataset 参数）")
    parser.add_argument("--classes", default=None, help="类别 yaml 路径，用于 XML 标签映射")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # dataset 位置参数或 --path 均可指定路径
    data_dir = args.data_dir if args.data_dir != DATA_DIR else args.dataset
    classes_path = args.classes

    print(f"数据目录：{data_dir}")
    print("-" * 50)

    step, total = 1, 4

    # Step 1: 加载类别映射
    print(f"[{step}/{total}] 加载类别映射...")
    step += 1
    label_mapping = {}
    if classes_path and os.path.isfile(classes_path):
        import yaml
        with open(classes_path, 'r') as f:
            label_mapping = yaml.safe_load(f)
        cls_names = label_mapping.get('names', label_mapping)
        print(f"  类别: {cls_names} ({len(cls_names)} 类)")
    else:
        print("  无类别文件，XML 转换时尝试从 XML 中提取类别")

    # Step 2: XML → YOLO 转换
    print(f"[{step}/{total}] XML → YOLO 转换...")
    step += 1
    annot_dir = os.path.join(data_dir, "annotations")
    if os.path.isdir(annot_dir):
        xml_count = len([f for f in os.listdir(annot_dir) if f.endswith(".xml")])
        label_dir = os.path.join(data_dir, "labels")
        label_count = len(os.listdir(label_dir)) if os.path.isdir(label_dir) else 0
        # 如果 labels 完全缺失，或 xml 数量明显多于 label，需要转换
        if label_count == 0 or xml_count > label_count:
            print(f"  发现 {xml_count} 个 XML，labels/ 仅 {label_count} 个，开始转换...")
            ok, total_xml = convert_xml_to_yolo(data_dir, label_mapping)
            print(f"  转换完成: {ok} / {total_xml}")
        else:
            print(f"  跳过: XML({xml_count}) 不多于 labels({label_count})")
    else:
        print("  跳过: 无 annotations/ 目录")

    # Step 3: 清理未配对 images/labels（在 xml2yolo 之后执行）
    print(f"[{step}/{total}] 清理未配对 images/labels...")
    step += 1
    removed_img, removed_label = clean_unpaired(data_dir)
    print(f"  删除: images={removed_img}, labels={removed_label}")

    # Step 4: 同步 train/val
    print(f"[{step}/{total}] 同步 train/val 列表...")
    step += 1
    img_map = list_images(os.path.join(data_dir, "images"))
    label_map = list_labels(os.path.join(data_dir, "labels"))
    valid_filenames = {img_map[s] for s in set(img_map.keys()) & set(label_map.keys()) if s in img_map}

    for name in ("train.txt", "val.txt"):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            n_kept, n_rm = sync_split_list(path, valid_filenames)
            print(f"  {name}: 保留 {n_kept} 条，移除 {n_rm} 条")

    print("-" * 50)
    print("完成！")


if __name__ == "__main__":
    main()
