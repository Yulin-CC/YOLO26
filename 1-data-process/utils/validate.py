#!/usr/bin/env python3
"""
# @Author: 算法组
# @Date: 2026-06-30
# @Description: YOLO 数据集一致性校验器 — 输出结构化 JSON 报告，供 Agent/CLI 使用。
#
# 检查项：
#   1. 目录结构：images/ labels/ 是否存在
#   2. 文件计数：images/ labels/ 文件数
#   3. image/label 对齐：同名同文件判断
#   4. 标签格式：.txt 内容是否合法（class_id, x, y, w, h）
#   5. 训练集划分：train.txt / val.txt 是否存在、覆盖率、重叠
#   6. 类别有效性：class_id 是否在范围内
#   7. 图像可读性：JPEG/PNG 能否正常读取
#
# 输出格式：
#   {
#     "status": "pass" | "fail",
#     "dataset": "dataset_name",
#     "path": "/abs/path",
#     "counts": {"images": N, "labels": N},
#     "checks": {
#       "dir_structure": {"pass": bool, "missing": [...], "detail": "..."},
#       "image_label_match": {"pass": bool, "matching": N, "img_only": N, "label_only": N, "detail": "..."},
#       "label_format": {"pass": bool, "total": N, "bad_format": N, "empty": N, "detail": "..."},
#       "split": {"pass": bool, "has_train": bool, "has_val": bool, "train_count": N, "val_count": N, "overlap": N, "coverage": float, "detail": "..."},
#       "image_readable": {"pass": bool, "total": N, "unreadable": N, "detail": "..."}
#     },
#     "actions_needed": [
#       {"action": "align", "priority": 1, "reason": "描述"},
#       {"action": "xml2yolo", "priority": 2, "reason": "描述"},
#       {"action": "fix_label_format", "priority": 2, "reason": "描述"},
#       {"action": "rename", "priority": 3, "reason": "描述"},
#       {"action": "split", "priority": 4, "reason": "描述"}
#     ]
#   }
#
# Usage:
#   python 1-data-process/utils/validate.py /path/to/dataset
#   python 1-data-process/utils/validate.py /path/to/dataset --json   # 仅输出 JSON
"""

import argparse
import json
import os
import re
import sys


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# yolo_rename.py 目标格式：{prefix}-{YYMMDD}-{seq}.{ext}
RENAMED_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9_\-]+-\d{6}-\d{3,}\.(jpg|jpeg|png|bmp|tif|tiff)$",
    re.IGNORECASE,
)


def renamed_name_pattern(prefix: str | None = None) -> re.Pattern:
    """返回重命名后的文件名匹配规则。"""
    if prefix:
        return re.compile(
            rf"^{re.escape(prefix)}-\d{{6}}-\d{{3,}}\.(jpg|jpeg|png|bmp|tif|tiff)$",
            re.IGNORECASE,
        )
    return RENAMED_NAME_PATTERN


def list_files(folder: str, exts: set) -> dict:
    """返回目录 {stem: filename}。"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in {e.lower() for e in exts}:
            out[stem] = fn
    return out


def read_split(path: str) -> set:
    """读取 train.txt / val.txt，返回条目集合（保留原始路径）。"""
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def split_to_filenames(entries: set) -> set:
    """将 train.txt / val.txt 条目统一为 images/ 下的文件名（basename）。"""
    names = set()
    for entry in entries:
        name = entry.replace("\\", "/").strip()
        if name.startswith("./"):
            name = name[2:]
        names.add(os.path.basename(name))
    return names


def labels_match_task(data_dir: str, task: str = "detect") -> bool:
    """检查 labels/ 下现有 txt 是否与当前 task 格式一致。"""
    label_dir = os.path.join(data_dir, "labels")
    if not os.path.isdir(label_dir):
        return False
    checked = 0
    for fn in os.listdir(label_dir):
        if not fn.lower().endswith(".txt"):
            continue
        checked += 1
        ok, _, is_empty = check_label_format(os.path.join(label_dir, fn), task=task)
        if not is_empty and not ok:
            return False
    return checked > 0


def check_label_format(label_file: str, max_classes: int = 1000, task: str = "detect") -> tuple:
    """检查单个 YOLO label 文件格式。返回 (ok, bad_lines_count, is_empty)。"""
    try:
        with open(label_file, encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            return True, 0, True  # 空文件是正常的（无目标）
        lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
        bad = 0
        for line in lines:
            parts = line.split()
            if task == "segment":
                if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                    bad += 1
                    continue
            elif len(parts) != 5:
                bad += 1
                continue
            try:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if cls_id < 0 or cls_id >= max_classes:
                    bad += 1
                    continue
                for c in coords:
                    if c < 0.0 or c > 1.0:
                        bad += 1
                        break
            except (ValueError, IndexError):
                bad += 1
        return bad == 0, bad, False
    except Exception:
        return False, 1, False


def validate(data_dir: str, task: str = "detect", platform: str = "labelimg", prefix: str | None = None) -> dict:
    """校验 YOLO 数据集一致性，返回结构化结果。"""
    data_dir = os.path.abspath(data_dir)
    name = os.path.basename(data_dir)

    result = {
        "status": "pass",
        "dataset": name,
        "path": data_dir,
        "counts": {},
        "checks": {},
        "actions_needed": []
    }

    # 1. 目录结构
    dir_missing = []
    for sub in ("images", "labels"):
        if not os.path.isdir(os.path.join(data_dir, sub)):
            dir_missing.append(sub)
    result["checks"]["dir_structure"] = {
        "pass": len(dir_missing) == 0,
        "missing": dir_missing,
        "detail": f"缺失目录: {', '.join(dir_missing)}" if dir_missing else "所有目录存在"
    }
    if dir_missing:
        result["status"] = "fail"

    # 2. 文件计数
    img_files = list_files(os.path.join(data_dir, "images"), IMAGE_EXTS)
    label_files = list_files(os.path.join(data_dir, "labels"), {".txt"})
    n_img, n_label = len(img_files), len(label_files)
    result["counts"] = {"images": n_img, "labels": n_label}

    # 3. image/label 对齐
    il_match = {s for s in img_files if s in label_files}
    img_only = set(img_files.keys()) - il_match
    label_only = set(label_files.keys()) - il_match
    il_ok = len(img_only) == 0 and len(label_only) == 0

    # 如果有 annotations/ XML 且 labels 完全缺失，align 优先级降低（先做 xml2yolo）
    annot_dir = os.path.join(data_dir, "annotations")
    has_xml_and_no_labels = os.path.isdir(annot_dir) and n_label == 0

    result["checks"]["image_label_match"] = {
        "pass": il_ok,
        "matching": len(il_match),
        "img_only": len(img_only),
        "label_only": len(label_only),
        "detail": f"images: {n_img}, labels: {n_label}, 匹配: {len(il_match)}"
    }
    if not il_ok and not has_xml_and_no_labels:
        result["status"] = "fail"
        result["actions_needed"].append({
            "action": "align",
            "priority": 1,
            "reason": f"images/labels 未对齐 — images-only: {len(img_only)}, labels-only: {len(label_only)}"
        })

    # 4. 标签格式检查
    total_bad = 0
    total_empty = 0
    total_checked = 0
    label_dir = os.path.join(data_dir, "labels")
    if os.path.isdir(label_dir):
        for fn in os.listdir(label_dir):
            if not fn.lower().endswith(".txt"):
                continue
            total_checked += 1
            fpath = os.path.join(label_dir, fn)
            fmt_ok, bad_lines, is_empty = check_label_format(fpath, task=task)
            if not fmt_ok:
                total_bad += 1
            if is_empty:
                total_empty += 1

    fmt_ok = total_bad == 0
    result["checks"]["label_format"] = {
        "pass": fmt_ok,
        "total": total_checked,
        "bad_format": total_bad,
        "empty": total_empty,
        "detail": f"总标签: {total_checked}, 空标签: {total_empty}, 格式错误: {total_bad}"
    }
    if not fmt_ok:
        result["status"] = "fail"
        if platform == "labelme":
            result["actions_needed"].append({
                "action": "json2yolo",
                "priority": 2,
                "reason": f"{total_bad} 个标签与当前 task={task} 格式不符，需从 JSON 重新转换"
            })
        else:
            result["actions_needed"].append({
                "action": "fix_label_format",
                "priority": 2,
                "reason": f"{total_bad} 个标签文件格式错误（需检查 class_id 或归一化坐标）"
            })

    # 5. 训练集划分
    train_set = read_split(os.path.join(data_dir, "train.txt"))
    val_set = read_split(os.path.join(data_dir, "val.txt"))
    train_names = split_to_filenames(train_set)
    val_names = split_to_filenames(val_set)
    all_img_filenames = set(img_files.values())

    has_train = len(train_names) > 0
    has_val = len(val_names) > 0
    overlap = train_names & val_names
    missing_from_split = all_img_filenames - (train_names | val_names)
    extra_in_split = (train_names | val_names) - all_img_filenames
    coverage = len(train_names | val_names) / len(all_img_filenames) * 100 if all_img_filenames else 0

    split_ok = has_train and has_val and len(overlap) == 0 and len(missing_from_split) == 0
    result["checks"]["split"] = {
        "pass": split_ok,
        "has_train": has_train,
        "has_val": has_val,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "overlap": len(overlap),
        "missing": len(missing_from_split),
        "coverage": round(coverage, 1),
        "detail": f"train: {len(train_names)}, val: {len(val_names)}, 重叠: {len(overlap)}, 覆盖率: {coverage:.1f}%"
    }
    if not split_ok:
        result["status"] = "fail"
        if not has_train or not has_val:
            result["actions_needed"].append({
                "action": "split",
                "priority": 4,
                "reason": f"缺失 train/val 或覆盖率不足 ({coverage:.1f}%)"
            })
        if len(overlap) > 0:
            result["actions_needed"].append({
                "action": "split",
                "priority": 4,
                "reason": f"train/val 存在 {len(overlap)} 个重叠样本"
            })

    # 6. 图像可读性检查（抽样，最多检查 100 张）
    img_dir = os.path.join(data_dir, "images")
    unreadable = 0
    checked_imgs = 0
    sample_size = min(100, n_img)
    if n_img > 0 and os.path.isdir(img_dir):
        import random
        sample_files = random.sample(list(img_files.values()), sample_size)
        for fn in sample_files:
            try:
                from PIL import Image
                with Image.open(os.path.join(img_dir, fn)) as img:
                    img.verify()
            except Exception:
                unreadable += 1
            checked_imgs += 1

    img_ok = unreadable == 0
    result["checks"]["image_readable"] = {
        "pass": img_ok,
        "total": checked_imgs,
        "unreadable": unreadable,
        "detail": f"抽样检查 {checked_imgs} 张，{unreadable} 张无法读取"
    }

    # 7. XML 存在性检查（labelimg：有 annotations/ 但无 labels/）
    annot_dir = os.path.join(data_dir, "annotations")
    if platform == "labelimg" and os.path.isdir(annot_dir) and n_label == 0:
        xml_count = len([f for f in os.listdir(annot_dir) if f.endswith(".xml")])
        if xml_count > 0:
            result["actions_needed"].append({
                "action": "xml2yolo",
                "priority": 2,
                "reason": f"有 annotations/ ({xml_count} XML) 但无 labels/，需要 XML → YOLO 转换"
            })

    # 7b. LabelMe JSON 存在性检查（labelme：有 jsons-*/ 但 labels/ 不足）
    json_dirs = ("jsons-segment", "jsons-detect", "jsons-labelme")
    total_json = 0
    for dir_name in json_dirs:
        json_dir = os.path.join(data_dir, dir_name)
        if os.path.isdir(json_dir):
            total_json += len([f for f in os.listdir(json_dir) if f.lower().endswith(".json")])
    if platform == "labelme" and total_json > 0 and n_label < total_json:
        found_dirs = [d for d in json_dirs if os.path.isdir(os.path.join(data_dir, d))]
        dirs_str = ", ".join(found_dirs)
        result["actions_needed"].append({
            "action": "json2yolo",
            "priority": 2,
            "reason": f"有 {dirs_str} ({total_json} JSON) 但 labels/ 仅 {n_label} 个，需要 JSON → YOLO 转换 ({task})"
        })

    # 8. 重命名需求（文件名不符合 yolo_rename 标准格式）
    if n_img > 0:
        pattern = renamed_name_pattern(prefix)
        non_standard = [f for f in img_files.values() if not pattern.match(f)]
        if non_standard:
            result["status"] = "fail"
            hint = f"前缀 {prefix}" if prefix else "标准格式 prefix-YYMMDD-001"
            result["actions_needed"].append({
                "action": "rename",
                "priority": 3,
                "reason": f"{len(non_standard)} 个图像未按 {hint} 命名（示例: {non_standard[:3]}）"
            })

    # 9. 有待处理操作时标记为未通过
    if result["actions_needed"]:
        result["status"] = "fail"

    # 10. 排序 actions
    result["actions_needed"].sort(key=lambda x: x["priority"])

    return result


def print_report(result: dict):
    """打印人类可读报告。"""
    print(f"\n{'=' * 60}")
    print(f"  YOLO 数据集校验: {result['dataset']}")
    print(f"  路径: {result['path']}")
    print(f"{'=' * 60}")

    counts = result['counts']
    print(f"\n  文件计数: images={counts['images']}, labels={counts['labels']}")

    for check_name, check_data in result['checks'].items():
        icon = "✅" if check_data['pass'] else "❌"
        print(f"\n  [{icon}] {check_name}: {check_data['detail']}")

    if result['actions_needed']:
        print(f"\n{'─' * 60}")
        print(f"  需要执行的操作:")
        for i, action in enumerate(result['actions_needed'], 1):
            icon_map = {
                "align": "🔗",
                "xml2yolo": "🔄",
                "json2yolo": "📋",
                "fix_label_format": "📝",
                "rename": "📛",
                "split": "✂️"
            }
            icon = icon_map.get(action['action'], "🔧")
            print(f"    [{i}] {icon} {action['action']}: {action['reason']}")
    else:
        print(f"\n  ✅ 无需整理 — 数据集完整")

    status_icon = "✅" if result['status'] == 'pass' else "❌"
    print(f"\n{'─' * 60}")
    print(f"  状态: {status_icon} {'通过' if result['status'] == 'pass' else '未通过'}")
    print(f"{'=' * 60}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLO 数据集一致性校验器")
    parser.add_argument("dataset", help="YOLO 数据集目录路径")
    parser.add_argument("--json", action="store_true", help="仅输出 JSON")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dataset):
        print(f"[错误] 目录不存在: {args.dataset}")
        sys.exit(1)

    result = validate(args.dataset)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    sys.exit(0 if result['status'] == 'pass' else 1)


if __name__ == "__main__":
    main()
