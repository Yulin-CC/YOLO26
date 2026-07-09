"""
# @Author: AI产品研发组
# @Date: 2026-07-09
# @Description: YOLO 推理逻辑封装
"""

from pathlib import Path

from ultralytics import YOLO

from .config_loader import get_task


def run_predict(
    weights: str | Path,
    source: str | Path,
    device,
    output: str | Path,
    task: str | None = None,
    imgsz: int = 640,
    conf: float = 0.4,
    iou: float = 0.45,
    save_conf: bool = True,
    save_crop: bool = False,
    retina_masks: bool | None = None,
    **kwargs,
):
    task = get_task(task)
    if retina_masks is None:
        retina_masks = task == "segment"

    output = Path(output).resolve()
    model = YOLO(str(weights), task=task)
    predict_kwargs = dict(
        source=str(source),
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save_dir=str(output),
        exist_ok=True,
        save=True,
        save_conf=save_conf,
        **kwargs,
    )
    if task == "detect":
        predict_kwargs["save_crop"] = save_crop
    if task == "segment":
        predict_kwargs["retina_masks"] = retina_masks

    return model.predict(**predict_kwargs)
