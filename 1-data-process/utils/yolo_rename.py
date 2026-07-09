#!/usr/bin/env python3
"""
# @Author: 算法组
# @Date: 2026-06-30
# @Description: YOLO images/labels 成对重命名。
#   格式：{prefix}-{dateYYMMDD}-{seq:03d}.{ext}
#   images/ 和 labels/ 同步重命名，保持 stem 一致。
#
# @Command: python 1-data-process/utils/yolo_rename.py /path/to/dataset --prefix Person --date 260630
"""

import argparse
import os
import re
import glob
from PIL import Image

# ==============================
# 接口配置
# ==============================
DATA_DIR = "/path/to/dataset"
PREFIX = "yolo"
DATE_TAG = "260630"
START_INDEX = 1
DIGITS = 3
# ==============================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_EXT = {".txt"}


def list_files(folder: str, exts: set) -> dict:
    """stem -> filename"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts:
            out[stem] = fn
    return out


def list_paired_stems(data_dir: str) -> list:
    """返回 images/labels 均存在的 stem，按 stem 排序。"""
    img_map = list_files(os.path.join(data_dir, "images"), IMAGE_EXTS)
    label_map = list_files(os.path.join(data_dir, "labels"), LABEL_EXT)
    stems = set(img_map.keys()) & set(label_map.keys())
    return sorted(stems)


def get_image_size(images_dir: str, stem: str) -> tuple:
    """获取图像实际尺寸。"""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        fpath = os.path.join(images_dir, stem + ext)
        if os.path.isfile(fpath):
            try:
                with Image.open(fpath) as img:
                    return img.size  # (width, height)
            except Exception:
                break
    return (None, None)


def target_name(stem: str, index: int) -> tuple:
    """生成新文件名 (image_name, label_name)。"""
    new_stem = f"{PREFIX}-{DATE_TAG}-{index:0{DIGITS}d}"
    return f"{new_stem}.jpg", f"{new_stem}.txt"


def update_split_files(data_dir: str, stem_map: dict[str, str]) -> None:
    """重命名后同步更新 train.txt / val.txt 中的图像路径。"""
    for split_name in ("train.txt", "val.txt"):
        split_path = os.path.join(data_dir, split_name)
        if not os.path.isfile(split_path):
            continue
        with open(split_path, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        updated = []
        for line in lines:
            normalized = line.replace("\\", "/")
            dirname = os.path.dirname(normalized)
            basename = os.path.basename(normalized)
            stem, ext = os.path.splitext(basename)
            if stem in stem_map:
                new_base = stem_map[stem] + ext
                if dirname:
                    updated.append(f"{dirname}/{new_base}")
                else:
                    updated.append(new_base)
            else:
                updated.append(line)
        with open(split_path, "w", encoding="utf-8") as f:
            f.write("\n".join(updated))
            if updated:
                f.write("\n")


def rename_pairs(data_dir: str, dry_run: bool = False) -> None:
    stems = list_paired_stems(data_dir)
    if not stems:
        print("未找到完整成对样本（images/labels 必填）。")
        return

    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    # 收集图像信息（尺寸 + 文件）
    img_info = {}
    for stem in stems:
        img_size = get_image_size(images_dir, stem)
        img_ext = None
        for ext in IMAGE_EXTS:
            fpath = os.path.join(images_dir, stem + ext)
            if os.path.isfile(fpath):
                img_ext = ext
                break
        if img_ext:
            img_info[stem] = {"size": img_size, "ext": img_ext}

    print(f"数据目录：{data_dir}")
    print(f"命名格式：{PREFIX}-{DATE_TAG}-{'0' * DIGITS}.jpg / .txt")
    print(f"成对样本：{len(stems)}")
    print("-" * 50)

    stem_map = {stem: f"{PREFIX}-{DATE_TAG}-{i:0{DIGITS}d}" for i, stem in enumerate(stems, start=START_INDEX)}

    if not dry_run:
        # 第一阶段：先重命名为临时名（避免冲突）
        for i, stem in enumerate(stems, start=START_INDEX):
            old_img_name = None
            for ext in IMAGE_EXTS:
                fpath = os.path.join(images_dir, stem + ext)
                if os.path.isfile(fpath):
                    old_img_name = stem + ext
                    break
            old_label_name = stem + ".txt"

            if old_img_name is None or not os.path.isfile(os.path.join(images_dir, old_img_name)):
                continue
            if not os.path.isfile(os.path.join(labels_dir, old_label_name)):
                continue

            new_img_name, new_label_name = target_name(stem, i)

            temp_img = os.path.join(images_dir, "__tmp_" + new_img_name)
            temp_label = os.path.join(labels_dir, "__tmp_" + new_label_name)

            os.rename(os.path.join(images_dir, old_img_name), temp_img)
            os.rename(os.path.join(labels_dir, old_label_name), temp_label)

        # 第二阶段：从临时名改为最终名
        for i, stem in enumerate(stems, start=START_INDEX):
            new_img_name, new_label_name = target_name(stem, i)

            temp_img = os.path.join(images_dir, "__tmp_" + new_img_name)
            temp_label = os.path.join(labels_dir, "__tmp_" + new_label_name)

            final_img = os.path.join(images_dir, new_img_name)
            final_label = os.path.join(labels_dir, new_label_name)

            os.rename(temp_img, final_img)
            os.rename(temp_label, final_label)

        update_split_files(data_dir, stem_map)

    shown = min(5, len(stems))
    print(f"\n示例（前 {shown} 组）：")
    for i, stem in enumerate(stems[:shown], start=START_INDEX):
        new_img, new_label = target_name(stem, i)
        print(f"  {stem}")
        print(f"    images/{new_img}")
        print(f"    labels/{new_label}")
    if len(stems) > shown:
        print(f"  ... 共 {len(stems)} 组")

    action = "预览完成" if dry_run else "重命名完成"
    print(f"\n{action}。")


def auto_extract_prefix(data_dir: str) -> str:
    """从数据集目录名或 train.txt 中的文件名自动提取前缀。"""
    dir_name = os.path.basename(data_dir)
    # 尝试从目录名提取：PCI-((Person_VOC))-Public-2312-(D)(R) → person
    match = re.search(r'\(\((.+?)\)\)', dir_name)
    if match:
        inner = match.group(1)
        # 取括号内第一个单词
        return inner.split('_')[0].lower()

    # 从文件名提取
    stems = list_paired_stems(data_dir)
    if stems:
        # 取第一个 stem，提取其前缀部分
        parts = re.split(r'[_\-]', stems[0])
        if parts:
            return parts[0].lower()

    return "yolo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO images/labels 成对重命名")
    parser.add_argument("dataset", help="YOLO 数据目录路径")
    parser.add_argument("--path", dest="data_dir", default=DATA_DIR, help="YOLO 数据目录（覆盖 dataset 参数）")
    parser.add_argument("--prefix", default=None, help="文件名前缀（自动提取如果未指定）")
    parser.add_argument("--date", dest="date_tag", default=DATE_TAG, help="日期标识 YYMMDD")
    parser.add_argument("--start", dest="start_index", type=int, default=START_INDEX, help="起始序号")
    parser.add_argument("--digits", type=int, default=DIGITS, help="序号位数")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际重命名")
    return parser


def main(argv=None):
    global PREFIX, DATE_TAG, START_INDEX, DIGITS
    args = build_parser().parse_args(argv)

    # dataset 位置参数或 --path 均可指定路径
    data_dir = args.data_dir if args.data_dir != DATA_DIR else args.dataset
    dry_run = args.dry_run

    # 自动提取前缀
    if args.prefix is None:
        PREFIX = auto_extract_prefix(data_dir)
        print(f"自动提取前缀: {PREFIX}")
    else:
        PREFIX = args.prefix

    DATE_TAG = args.date_tag
    START_INDEX = args.start_index
    DIGITS = args.digits

    rename_pairs(data_dir, dry_run=dry_run)


if __name__ == "__main__":
    main()
