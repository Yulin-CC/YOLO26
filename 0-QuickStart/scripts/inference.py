"""
# @Author: AI产品研发组
# @Date: 2026-07-09
# @Description: Ultralytics YOLO 推理入口
# @Command: python 0-QuickStart/scripts/inference.py --task detect
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config_loader import default_weights, get_task, load_config, resolve_path
from utils.inference_utils import run_predict


def parse_args():
    root_cfg = load_config()
    cfg = load_config("predict")
    train_cfg = load_config("train")
    default_task = root_cfg.get("task", "detect")
    parser = argparse.ArgumentParser(description="Ultralytics YOLO predict")
    parser.add_argument("--task", default=default_task, choices=("detect", "segment"), help="任务类型")
    parser.add_argument("--devices", type=int, default=cfg.get("dev_id", 0))
    parser.add_argument("--weights", default=cfg.get("weights") or None, help="模型权重路径")
    parser.add_argument("--dataset", default=cfg.get("dataset"), help="测试输入（单文件或目录）")
    parser.add_argument("--output", default=None, help="推理结果输出目录")
    parser.add_argument("--imgsz", type=int, default=cfg.get("imgsz", 640))
    parser.add_argument("--conf", type=float, default=cfg.get("conf", 0.4))
    parser.add_argument("--iou", type=float, default=cfg.get("iou", 0.45))
    parser.add_argument("--save_conf", action="store_true", default=cfg.get("save_conf", True))
    parser.add_argument("--save_crop", action="store_true", default=bool(cfg.get("save_crop", False)))
    parser.add_argument("--retina_masks", dest="retina_masks", action="store_true", help="segment 高分辨率 mask")
    parser.add_argument("--no-retina_masks", dest="retina_masks", action="store_false")
    parser.set_defaults(retina_masks=None)
    parser.add_argument("--project", default=train_cfg.get("project", "exp01"), help="实验名（推导默认权重）")
    return parser.parse_args()


def derive_output(dataset: str) -> Path:
    src = Path(dataset)
    if src.is_file():
        return src.parent / "repro"
    return src / "repro"


def main():
    args = parse_args()
    task = get_task(args.task)
    weights = resolve_path(args.weights or default_weights(task, args.project))
    dataset = args.dataset if Path(args.dataset).is_absolute() else str(resolve_path(args.dataset))
    output = Path(args.output) if args.output else derive_output(dataset)
    output.mkdir(parents=True, exist_ok=True)

    print(f"🔍 开始推理 | task={task} | weights={weights} | source={dataset} | output={output}")
    run_predict(
        weights=weights,
        source=dataset,
        device=args.devices,
        output=output,
        task=task,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        retina_masks=args.retina_masks,
    )


if __name__ == "__main__":
    main()
