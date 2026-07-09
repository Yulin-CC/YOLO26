"""
# @Author: AI产品研发组
# @Date: 2026-07-09
# @Description: Ultralytics YOLO 训练入口
# @Command: python 0-QuickStart/scripts/train.py --task detect
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from utils.config_loader import (
    default_dataset,
    default_model,
    get_task,
    load_config,
    resolve_path,
    train_run_dir,
)
from utils.vis_export import backup_train_configs


def parse_args():
    root_cfg = load_config()
    cfg = load_config("train")
    default_task = root_cfg.get("task", "detect")
    parser = argparse.ArgumentParser(description="Ultralytics YOLO train")
    parser.add_argument("--task", default=default_task, choices=("detect", "segment"), help="任务类型")
    parser.add_argument("--devices", default=cfg.get("dev_id", "0"), help="GPU ID")
    parser.add_argument("--project", default=cfg.get("project", "exp01"), help="实验名")
    parser.add_argument("--model", default=cfg.get("model") or None, help="YOLO26 预训练权重")
    parser.add_argument("--dataset", default=load_config("dataset").get("yaml") or None, help="数据集 yaml")
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 100))
    parser.add_argument("--batch", type=int, default=cfg.get("batch", 16))
    parser.add_argument("--imgsz", type=int, default=cfg.get("imgsz", 640))
    parser.add_argument("--workers", type=int, default=cfg.get("workers", 8))
    parser.add_argument("--run_base_dir", default=None, help="训练输出根目录")
    return parser.parse_args()


def main():
    args = parse_args()
    task = get_task(args.task)
    run_base_dir = args.run_base_dir or train_run_dir(task)
    model_path = resolve_path(args.model or default_model(task))
    data_path = resolve_path(args.dataset or default_dataset(task))
    run_dir = resolve_path(run_base_dir) / args.project
    backup_train_configs(run_dir)

    print(f"🚀 开始训练 | task={task} | project={args.project} | data={data_path}")
    model = YOLO(str(model_path), task=task)
    model.train(
        device=args.devices,
        model=str(model_path),
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(resolve_path(run_base_dir)),
        name=args.project,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
