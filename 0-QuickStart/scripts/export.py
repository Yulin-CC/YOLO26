"""
# @Author: AI产品研发组
# @Date: 2026-07-09
# @Description: Ultralytics YOLO 模型导出（ONNX 等格式）
# @Command: python 0-QuickStart/scripts/export.py --task detect
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config_loader import default_weights, get_task, load_config, resolve_path
from utils.vis_export import export_model


def parse_args():
    root_cfg = load_config()
    cfg = load_config("export")
    train_cfg = load_config("train")
    default_task = root_cfg.get("task", "detect")
    parser = argparse.ArgumentParser(description="Ultralytics YOLO export")
    parser.add_argument("--task", default=default_task, choices=("detect", "segment"), help="任务类型")
    parser.add_argument("--devices", default=str(cfg.get("dev_id", 0)))
    parser.add_argument("--weights", default=cfg.get("weights") or None, help="模型权重路径")
    parser.add_argument("--project", default=train_cfg.get("project", "exp01"), help="实验名（推导默认权重）")
    parser.add_argument("--imgsz", nargs="+", type=int, default=cfg.get("imgsz", [640, 640]))
    parser.add_argument("--dynamic", action="store_true", default=cfg.get("dynamic", False))
    parser.add_argument("--format", default=cfg.get("format", "onnx"), help="导出格式")
    return parser.parse_args()


def main():
    args = parse_args()
    task = get_task(args.task)
    weights = resolve_path(args.weights or default_weights(task, args.project))
    print(f"📦 开始导出 | task={task} | weights={weights} | format={args.format}")
    export_model(
        weights=weights,
        fmt=args.format,
        imgsz=args.imgsz,
        device=args.devices,
        dynamic=args.dynamic,
    )


if __name__ == "__main__":
    main()
