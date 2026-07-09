"""
# @Author: AI产品研发组
# @Date: 2026-07-09
# @Description: Ultralytics YOLO 评估入口
# @Command: python 0-QuickStart/scripts/eval.py --task detect
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from utils.config_loader import (
    default_dataset,
    default_weights,
    eval_run_dir,
    get_task,
    load_config,
    resolve_path,
)


def parse_args():
    root_cfg = load_config()
    cfg = load_config("eval")
    train_cfg = load_config("train")
    default_task = root_cfg.get("task", "detect")
    parser = argparse.ArgumentParser(description="Ultralytics YOLO eval")
    parser.add_argument("--task", default=default_task, choices=("detect", "segment"), help="任务类型")
    parser.add_argument("--devices", default=str(cfg.get("dev_id", 0)))
    parser.add_argument("--weights", default=cfg.get("weights") or None, help="模型权重路径")
    parser.add_argument("--dataset", default=cfg.get("dataset") or None, help="数据集 yaml")
    parser.add_argument("--output", default=None, help="评估结果输出目录")
    parser.add_argument("--project", default=train_cfg.get("project", "exp01"), help="实验名")
    parser.add_argument("--imgsz", type=int, default=cfg.get("imgsz", 640))
    parser.add_argument("--conf", type=float, default=cfg.get("conf", 0.4))
    parser.add_argument("--iou", type=float, default=cfg.get("iou", 0.6))
    return parser.parse_args()


def main():
    args = parse_args()
    task = get_task(args.task)
    weights = resolve_path(args.weights or default_weights(task, args.project))
    data_path = resolve_path(args.dataset or default_dataset(task))
    output = Path(args.output) if args.output else resolve_path(eval_run_dir(task)) / args.project
    output.mkdir(parents=True, exist_ok=True)

    print(f"📊 开始评估 | task={task} | weights={weights} | data={data_path}")
    model = YOLO(str(weights), task=task)
    model.val(
        device=args.devices,
        data=str(data_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=str(output.parent),
        name=output.name,
    )


if __name__ == "__main__":
    main()
