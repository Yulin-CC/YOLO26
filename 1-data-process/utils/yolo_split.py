#!/usr/bin/env python3
"""
# @Author: 算法组
# @Date: 2026-06-30
# @Description: YOLO 数据集划分：images/labels 按随机比例生成 train.txt / val.txt。
#
# @Mode  : txt=仅写列表（默认）；dir=复制到 for_training/{train,val}/{images,labels}
# @Split : 默认 9:1（SPLIT_RATIO=0.9），PART_RATIO=1.0 表示使用全部成对样本。
# @Command: python 1-data-process/utils/yolo_split.py /path/to/dataset --mode txt --split-ratio 0.9 --seed 42
"""

import argparse
import os
import random
import shutil
from os.path import join

# ==============================
# 接口配置
# ==============================
DATA_DIR = "/path/to/dataset"
MODE = "txt"           # txt 或 dir
SPLIT_RATIO = 0.9      # 训练集占比
PART_RATIO = 1.0       # 使用样本比例（1.0=全部）
RANDOM_SEED = 42       # 固定随机种子，便于复现
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


def check_yolo_layout(path: str) -> tuple:
    """检查 images/labels 目录，返回 (ok, img_map, label_map)。"""
    img_map = list_files(join(path, "images"), IMAGE_EXTS)
    label_map = list_files(join(path, "labels"), LABEL_EXT)
    return True, img_map, label_map


def paired_stems(img_map: dict, label_map: dict) -> list:
    """返回 images/labels 均存在的 stem 列表。"""
    stems = set(img_map.keys()) & set(label_map.keys())
    return sorted(stems)


def split_train_val(stems: list, split_ratio: float, part_ratio: float, seed: int) -> tuple:
    """打乱后按比例截取，再划分 train / val。"""
    stems = list(stems)
    if seed is not None:
        random.seed(seed)
    random.shuffle(stems)

    n = int(len(stems) * part_ratio)
    stems = stems[:n] if n > 0 else stems

    if len(stems) <= 1:
        return stems, []

    k = int(len(stems) * split_ratio)
    k = max(1, min(k, len(stems) - 1))
    return stems[:k], stems[k:]


def _img_path(img_map: dict, stem: str) -> str:
    """找到 stem 对应的图像文件名。"""
    for ext in IMAGE_EXTS:
        fpath = os.path.join("images", stem + ext)
        if os.path.isfile(fpath):
            return fpath
    return None


def _label_path(label_map: dict, stem: str) -> str:
    """找到 stem 对应的标签文件名。"""
    fpath = os.path.join("labels", stem + ".txt")
    if os.path.isfile(fpath):
        return fpath
    return None


def write_split_txt(data_dir: str, img_map: dict, train_set: list, val_set: list):
    """写入 train.txt / val.txt（相对路径 ./images/xxx.jpg）。"""
    def image_entry(stem: str) -> str:
        fn = img_map.get(stem)
        return f"./images/{fn}" if fn else f"./images/{stem}.jpg"

    with open(join(data_dir, "train.txt"), "w", encoding="utf-8", newline="\n") as f:
        for stem in train_set:
            f.write(image_entry(stem) + "\n")
    with open(join(data_dir, "val.txt"), "w", encoding="utf-8", newline="\n") as f:
        for stem in val_set:
            f.write(image_entry(stem) + "\n")


def create_txt(data_dir: str, split_ratio: float, part_ratio: float, seed: int) -> dict:
    """划分并写入 train.txt / val.txt。"""
    _, img_map, label_map = check_yolo_layout(data_dir)
    stems = paired_stems(img_map, label_map)
    train_set, val_set = split_train_val(stems, split_ratio, part_ratio, seed)
    write_split_txt(data_dir, img_map, train_set, val_set)
    return {"total": len(stems), "train": len(train_set), "val": len(val_set)}


def _copy_sample(data_dir: str, stem: str, dst_dir: str):
    """复制 image + label 到目标目录。"""
    img_rel = _img_path({"stem": stem}, stem)
    if img_rel:
        shutil.copy2(join(data_dir, img_rel), join(dst_dir, "images", os.path.basename(img_rel)))

    label_rel = _label_path({"stem": stem}, stem)
    if label_rel:
        shutil.copy2(join(data_dir, label_rel), join(dst_dir, "labels", os.path.basename(label_rel)))


def create_dir(data_dir: str, split_ratio: float, part_ratio: float, seed: int) -> dict:
    """划分并复制到 for_training/train、for_training/val。"""
    _, img_map, label_map = check_yolo_layout(data_dir)
    stems = paired_stems(img_map, label_map)
    train_set, val_set = split_train_val(stems, split_ratio, part_ratio, seed)

    base = join(data_dir, "for_training")
    for split_name in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(join(base, split_name, sub), exist_ok=True)

    from tqdm import tqdm

    for stem in tqdm(train_set, desc="  复制 train"):
        _copy_sample(data_dir, stem, join(base, "train"))
    for stem in tqdm(val_set, desc="  复制 val"):
        _copy_sample(data_dir, stem, join(base, "val"))

    return {"total": len(stems), "train": len(train_set), "val": len(val_set)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO 数据集划分 train/val")
    parser.add_argument("dataset", help="YOLO 数据目录路径")
    parser.add_argument("--Path", "--path", dest="data_dir", default=DATA_DIR, help="YOLO 数据目录（覆盖 dataset 参数）")
    parser.add_argument("--mode", default=MODE, choices=("txt", "dir"), help="txt=写列表；dir=复制子目录")
    parser.add_argument("--split-ratio", type=float, default=SPLIT_RATIO, help="训练集占比")
    parser.add_argument("--part-ratio", type=float, default=PART_RATIO, help="使用样本比例")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # dataset 位置参数或 --path 均可指定路径
    data_dir = args.data_dir if args.data_dir != DATA_DIR else args.dataset
    mode = args.mode
    split_ratio = args.split_ratio
    part_ratio = args.part_ratio
    seed = args.seed

    print(f"数据目录：{data_dir}")
    print(f"划分模式：{mode}  |  train 占比 {split_ratio}  |  样本比例 {part_ratio}  |  seed={seed}")
    print("-" * 50)

    step, total = 1, 3

    print(f"[{step}/{total}] 检查目录结构...")
    step += 1
    ok, img_map, label_map = check_yolo_layout(data_dir)
    stems = paired_stems(img_map, label_map)
    print(f"  成对样本：{len(stems)}")

    print(f"[{step}/{total}] 划分数据集...")
    step += 1
    if mode == "txt":
        stats = create_txt(data_dir, split_ratio, part_ratio, seed)
    elif mode == "dir":
        stats = create_dir(data_dir, split_ratio, part_ratio, seed)
    else:
        print(f'  [错误] MODE 应为 "txt" 或 "dir"，当前: {mode}')
        return
    print(f"  train / val: {stats['train']} / {stats['val']}  (共 {stats['total']})")

    print(f"[{step}/{total}] 完成")
    if mode == "txt":
        print(f"  train.txt → {join(data_dir, 'train.txt')}")
        print(f"  val.txt   → {join(data_dir, 'val.txt')}")
    else:
        print(f"  for_training → {join(data_dir, 'for_training')}")

    print("-" * 50)
    print("完成！")


if __name__ == "__main__":
    main()
