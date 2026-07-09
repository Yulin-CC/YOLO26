"""
# @Author: AI产品研发组
# @Date: 2026-07-09
# @Description: 训练配置备份与导出辅助
"""

import shutil
from pathlib import Path

from .config_loader import ROOT_DIR


def backup_train_configs(run_dir: Path) -> None:
    cfg_backup_dir = run_dir / "configs"
    cfg_backup_dir.mkdir(parents=True, exist_ok=True)
    backup_files = [ROOT_DIR / "config" / "default.yaml"] + sorted((ROOT_DIR / "data").glob("*.yaml"))
    for src in backup_files:
        if src.exists():
            shutil.copy2(src, cfg_backup_dir / src.name)


def export_model(
    weights: str | Path,
    fmt: str = "onnx",
    imgsz=None,
    device: str = "0",
    dynamic: bool = False,
):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return model.export(
        format=fmt,
        simplify=True,
        imgsz=imgsz or [640, 640],
        opset=17,
        dynamic=dynamic,
        device=device,
    )
