"""
# @Author: AI产品研发组
# @Date: 2026-07-09
# @Description: 加载 config/default.yaml，支持 CLI 覆盖
"""

from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CFG = ROOT_DIR / "config" / "default.yaml"
SUPPORTED_TASKS = ("detect", "segment")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def load_config(namespace: str | None = None) -> dict:
    with open(DEFAULT_CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if namespace:
        return cfg.get(namespace, {})
    return cfg


def get_task(override: str | None = None) -> str:
    """Return validated task name: detect or segment."""
    task = (override or load_config().get("task") or "detect").lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}', choose from {SUPPORTED_TASKS}")
    return task


def default_model(task: str | None = None) -> str:
    return "weights/yolo26s-seg.pt" if get_task(task) == "segment" else "weights/yolo26s.pt"


def default_dataset(task: str | None = None) -> str:
    return "data/0-coco8-seg.yaml" if get_task(task) == "segment" else "data/0-coco8.yaml"


def train_run_dir(task: str | None = None) -> str:
    return f"runs/{get_task(task)}/0-train"


def eval_run_dir(task: str | None = None) -> str:
    return f"runs/{get_task(task)}/2-eval"


def default_weights(task: str | None = None, project: str | None = None) -> str:
    project = project or load_config("train").get("project", "exp01")
    return f"{train_run_dir(task)}/{project}/weights/best.pt"
