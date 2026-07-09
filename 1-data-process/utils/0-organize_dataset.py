"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-07-09
# @Description: YOLO 数据集自动整理（格式转换 → 校验 → 执行 actions → 再校验）
# @Command: python 1-data-process/utils/0-organize_dataset.py --path /path/to/dataset --task detect --platform labelimg
"""

import argparse
import os
import sys
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from validate import labels_match_task, print_report, validate

SPLIT_RATIO = 0.9
RANDOM_SEED = 42

ACTION_PRIORITY = {
    "xml2yolo": 0,
    "json2yolo": 0,
    "align": 1,
    "fix_label_format": 2,
    "rename": 3,
    "split": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 单文件夹数据集自动整理")
    parser.add_argument("--path", required=True, help="数据集目录路径")
    parser.add_argument("--prefix", default=None, help="重命名前缀（留空则自动提取）")
    parser.add_argument("--task", default="detect", choices=("detect", "segment"), help="detect | segment")
    parser.add_argument("--platform", default="labelimg", choices=("labelimg", "labelme"), help="labelimg | labelme")
    parser.add_argument("--yaml", default=None, help="类别 yaml（labelme 必填；labelimg 可选）")
    return parser.parse_args()


def abort(step: str, reason: str, hint: str = "") -> None:
    """任一步骤失败即停止流水线并说明原因。"""
    print(f"\n{'=' * 60}")
    print(f"❌ [{step}] 失败，流水线已停止")
    print(f"   原因: {reason}")
    if hint:
        print(f"   建议: {hint}")
    print(f"{'=' * 60}")
    sys.exit(1)


def failed_check_details(result: dict) -> str:
    """汇总校验失败项，用于 abort 提示。"""
    lines = []
    for name, check in result.get("checks", {}).items():
        if not check.get("pass", True):
            lines.append(f"{name}: {check.get('detail', '')}")
    actions = [a["action"] for a in result.get("actions_needed", [])]
    if actions:
        lines.append(f"待处理: {', '.join(actions)}")
    return "；".join(lines) if lines else "未知错误"


def resolve_yaml(yaml_path: str | None) -> str | None:
    if not yaml_path:
        return None
    p = Path(yaml_path)
    if p.is_file():
        return str(p.resolve())
    abort("参数检查", f"类别 yaml 不存在: {yaml_path}")


def find_json_dir(data_dir: Path) -> Path | None:
    for dir_name in ("jsons-segment", "jsons-detect", "jsons-labelme"):
        candidate = data_dir / dir_name
        if candidate.is_dir() and list(candidate.glob("*.json")):
            return candidate
    return None


def validate_options(task: str, platform: str, yaml_path: str | None) -> None:
    if task == "segment" and platform == "labelimg":
        raise ValueError("segment 任务仅支持 platform=labelme")
    if platform == "labelme" and not yaml_path:
        raise ValueError("platform=labelme 时必须提供 --yaml")


def run_convert(path: str, task: str, platform: str, yaml_path: str | None) -> dict | None:
    """按 task + platform 选择转换脚本，返回统计信息。"""
    data_dir = Path(path)
    yaml_path = resolve_yaml(yaml_path)

    if platform == "labelimg":
        annot_dir = data_dir / "annotations"
        if not annot_dir.is_dir():
            abort("Step 0", f"未找到 annotations/ 目录: {annot_dir}")
        from convert_xml2txt import convert_all

        print("  使用 convert_xml2txt.py（detect + labelimg）")
        stats = convert_all(str(data_dir), yaml_path)
        print(f"  转换完成: {stats['ok']}/{stats['total']}")
        return stats

    json_dir = find_json_dir(data_dir)
    if json_dir is None:
        abort(
            "Step 0",
            "未找到 LabelMe JSON 目录（jsons-segment/、jsons-detect/ 或 jsons-labelme/）",
            "确认 JSON 标注已放入上述目录之一",
        )

    from convert_json2txt import labelme2txt

    txt_dir = data_dir / "labels"
    print(f"  使用 convert_json2txt.py（{task} + labelme），源目录: {json_dir.name}")
    stats = labelme2txt(json_dir, txt_dir, yaml_path, task=task)
    print(
        f"  转换完成: {stats['ok']}/{stats['total']}，"
        f"未知类别 {stats['skip_unknown_label']}，形状跳过 {stats['skip_shape']}"
    )
    return stats


def verify_convert(path: str, task: str, platform: str, stats: dict | None) -> None:
    """Step 0 完成后校验转换结果。"""
    if stats and stats.get("total", 0) > 0 and stats.get("ok", 0) == 0:
        abort(
            "Step 0",
            "JSON/XML 转换未生成任何有效 label",
            f"检查 task={task} 与标注 shape_type 是否匹配（detect=rectangle，segment=polygon）",
        )
    if not labels_match_task(path, task):
        abort(
            "Step 0",
            f"转换后 labels/ 格式仍不符合 task={task}",
            "detect 需要 bbox 五行；segment 需要多边形坐标（≥7 列）",
        )


def needs_convert(path: str, platform: str, task: str = "detect") -> bool:
    if platform == "labelimg":
        annot = os.path.join(path, "annotations")
        labels = os.path.join(path, "labels")
        if not os.path.isdir(annot):
            return False
        xml_count = len([f for f in os.listdir(annot) if f.lower().endswith(".xml")])
        label_count = len([f for f in os.listdir(labels) if f.endswith(".txt")]) if os.path.isdir(labels) else 0
        return xml_count > 0 and (label_count < xml_count or not labels_match_task(path, task))

    json_dirs = ("jsons-segment", "jsons-detect", "jsons-labelme")
    total_json = 0
    for dir_name in json_dirs:
        json_dir = os.path.join(path, dir_name)
        if os.path.isdir(json_dir):
            total_json += len([f for f in os.listdir(json_dir) if f.lower().endswith(".json")])
    if total_json == 0:
        return False
    labels = os.path.join(path, "labels")
    label_count = len([f for f in os.listdir(labels) if f.endswith(".txt")]) if os.path.isdir(labels) else 0
    return total_json > 0 and (label_count < total_json or not labels_match_task(path, task))


def get_ordered_actions(result: dict) -> list[str]:
    seen = set()
    ordered = []
    for item in sorted(
        result.get("actions_needed", []),
        key=lambda x: ACTION_PRIORITY.get(x["action"], x["priority"]),
    ):
        action = item["action"]
        if action not in seen:
            seen.add(action)
            ordered.append(action)
    return ordered


def run_action(action: str, path: str, prefix: str | None, task: str, platform: str, yaml_path: str | None) -> None:
    if action == "align":
        from yolo_merge import main as merge_main
        merge_main([path])
    elif action in ("xml2yolo", "json2yolo"):
        stats = run_convert(path, task, platform, yaml_path)
        verify_convert(path, task, platform, stats)
    elif action == "rename":
        from yolo_rename import main as rename_main
        argv = [path]
        if prefix:
            argv += ["--prefix", prefix]
        rename_main(argv)
    elif action == "split":
        from yolo_split import main as split_main
        split_main([
            path,
            "--mode", "txt",
            "--split-ratio", str(SPLIT_RATIO),
            "--seed", str(RANDOM_SEED),
        ])
    else:
        abort("Step 2", f"未知或不可自动执行的操作: {action}")


def verify_action(path: str, task: str, action: str) -> None:
    """单个 action 执行后的即时校验。"""
    if action in ("xml2yolo", "json2yolo") and not labels_match_task(path, task):
        abort("Step 2", f"执行 {action} 后 labels 格式仍不符合 task={task}")

    if action == "split":
        train_txt = os.path.join(path, "train.txt")
        val_txt = os.path.join(path, "val.txt")
        if not os.path.isfile(train_txt) or not os.path.isfile(val_txt):
            abort("Step 2", "split 未生成 train.txt / val.txt")

    if action == "align" and not labels_match_task(path, task):
        abort("Step 2", "align 后仍存在与 task 不符的 labels")


def print_header(path: str, task: str, platform: str) -> None:
    print("=" * 60)
    print(f"  YOLO 数据集整理: {os.path.basename(path)}")
    print(f"  路径: {path}")
    print(f"  任务: {task}  |  平台: {platform}")
    print("=" * 60)


def main():
    args = parse_args()
    path = os.path.abspath(args.path)

    try:
        validate_options(args.task, args.platform, args.yaml)
    except ValueError as e:
        abort("参数检查", str(e))

    if not os.path.isdir(path):
        abort("参数检查", f"数据集目录不存在: {path}")

    if args.platform == "labelme":
        resolve_yaml(args.yaml)

    print_header(path, args.task, args.platform)

    # Step 0: 格式转换
    if needs_convert(path, args.platform, args.task):
        print("\n[Step 0] 标签格式转换...")
        stats = run_convert(path, args.task, args.platform, args.yaml)
        verify_convert(path, args.task, args.platform, stats)
        print("  ✅ Step 0 完成")
    else:
        print("\n[Step 0] 跳过（labels 已存在且格式匹配）")

    # Step 1: 初始校验
    print("\n[Step 1] 初始校验...")
    initial = validate(path, task=args.task, platform=args.platform, prefix=args.prefix)
    print_report(initial)

    if initial.get("status") == "pass" and not get_ordered_actions(initial):
        print("\n✅ 数据集已完整，整理完成。")
        sys.exit(0)

    actions = get_ordered_actions(initial)
    if not actions:
        abort("Step 1", "校验未通过，且没有可自动修复的操作", failed_check_details(initial))

    # 无法自动修复的操作，直接停止
    if "fix_label_format" in actions and args.platform != "labelme":
        abort(
            "Step 1",
            "标签格式错误，无法自动修复",
            "人工检查 labels/ 下 txt 的 class_id 与归一化坐标",
        )

    # Step 2: 执行流水线（任一步失败即 break）
    print("\n[Step 2] 执行整理流水线...")
    print(f"  待执行: {' '.join(actions)}")
    for action in actions:
        if action == "fix_label_format":
            abort(
                "Step 2",
                "标签格式与当前 task 不符，需从 JSON 重新转换或人工修正",
                f"切换 task 后重跑脚本，或检查 labels/ 与 task={args.task} 是否匹配",
            )
        print(f"\n  → 执行: {action}")
        run_action(action, path, args.prefix, args.task, args.platform, args.yaml)
        verify_action(path, args.task, action)
        print(f"  ✅ {action} 完成")

    # Step 3: 最终校验
    print("\n[Step 3] 最终校验...")
    final = validate(path, task=args.task, platform=args.platform, prefix=args.prefix)
    print_report(final)

    if final.get("status") != "pass" or final.get("actions_needed"):
        abort("Step 3", "最终校验未通过", failed_check_details(final))

    counts = final.get("counts", {})
    print(f"\n  最终计数: images={counts.get('images', 0)}, labels={counts.get('labels', 0)}")
    print("✅ 整理完成，校验通过。")
    sys.exit(0)


if __name__ == "__main__":
    main()
