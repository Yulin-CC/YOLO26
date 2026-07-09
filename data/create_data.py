"""
Author: 算法组 蔡雨霖
Description: 生成训练数据集 yaml、数据集统计信息，以及数据版本更新信息
Date: 2024-05-11
"""

import os
from datetime import datetime
from os.path import join
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def list_images(root: str | Path, dirs: list[str]) -> tuple[dict, int]:
    """统计各 PCI 子目录下的 label 数量。"""
    root = Path(root)
    dictionary = {}
    sample_sum = 0
    for dir_name in dirs:
        if dir_name.split("-")[0] == "PCI":
            labels_dir = root / dir_name / "labels"
            if not labels_dir.is_dir():
                continue
            labels = os.listdir(labels_dir)
            key = f"{root.name}/{dir_name}".ljust(90)
            dictionary[key] = len(labels)
            sample_sum += len(labels)
    return dictionary, sample_sum


def collect_pci_dirs(root: str | Path) -> list[Path]:
    """返回 root 下所有含 train.txt / val.txt 的 PCI 子目录。"""
    root = Path(root)
    pci_dirs = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.split("-")[0] != "PCI":
            continue
        train_txt = entry / "train.txt"
        val_txt = entry / "val.txt"
        if train_txt.is_file() and val_txt.is_file():
            pci_dirs.append(entry)
        else:
            raise ValueError(f"There is no train/val.txt in dir # {entry.name} #, please check it!")
    return pci_dirs


def normalize_split_line(line: str, prefix: str | None) -> str:
    """将 split 行转为相对 dataset_root 的路径（保留 ./ 前缀供 YOLO 解析）。"""
    line = line.strip()
    if not line:
        return ""
    if line.startswith("./"):
        rel = line[2:]
    else:
        rel = line.lstrip("/")
    if prefix:
        rel = f"{prefix}/{rel}"
    rel = rel.replace("\\", "/")
    return rel if rel.startswith("./") else f"./{rel}"


def merge_split_txts(txt_paths: list[Path], output_path: Path, prefixes: list[str | None]) -> int:
    """合并多个 train.txt / val.txt，返回总行数。"""
    lines: list[str] = []
    for txt_path, prefix in zip(txt_paths, prefixes):
        with open(txt_path, encoding="utf-8") as f:
            for raw in f:
                normalized = normalize_split_line(raw, prefix)
                if normalized:
                    lines.append(normalized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    return len(lines)


def resolve_dataset_layout(scan_paths: list[str], dataset_root: str | Path) -> tuple[list[Path], list[Path], int, list[str | None]]:
    """解析数据集结构：PCI 多目录或单目录 flat（根目录自带 train/val.txt）。"""
    dataset_root = Path(dataset_root).resolve()
    train_files: list[Path] = []
    val_files: list[Path] = []
    prefixes: list[str | None] = []
    dataset_quantity = 0

    for scan_path in scan_paths:
        root = Path(scan_path).resolve()
        pci_dirs = collect_pci_dirs(root) if root.is_dir() else []

        if pci_dirs:
            for pci_dir in pci_dirs:
                train_files.append(pci_dir / "train.txt")
                val_files.append(pci_dir / "val.txt")
                prefixes.append(pci_dir.name)
                dataset_quantity += 1
            continue

        train_txt = root / "train.txt"
        val_txt = root / "val.txt"
        if train_txt.is_file() and val_txt.is_file():
            train_files.append(train_txt)
            val_files.append(val_txt)
            prefixes.append(None)
            dataset_quantity += 1
            continue

        raise ValueError(f"No PCI subdirs or train/val.txt found under: {root}")

    return train_files, val_files, dataset_quantity, prefixes


def write_dataset_yaml(
    out_file: Path,
    dataset_root: Path,
    nc: int,
    names: list[str],
    header_lines: list[str],
) -> None:
    """写入 YOLO 数据集 yaml（path + 相对 train.txt / val.txt）。"""
    dataset_root = dataset_root.resolve()
    with open(out_file, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(f"{line}\n")
        f.write("\n\n")
        f.write(f"path: {dataset_root}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write(f"nc: {nc}\n")
        f.write("names:\n")
        for name in names:
            f.write(f"- {name}\n")


def update(file, incharge, model_vision, vision_instruction, quantity, change_dir):
    """追加版本更新说明。"""
    infos = [
        f"## {model_vision} ##\n",
        f" # Date：{datetime.now()}",
        f" # Responsible Person：{incharge}",
        f" # Dataset Used: {quantity}",
        f" # Dataset update：{change_dir}",
        f" # Update news：{vision_instruction}",
    ]
    for comment in infos:
        file.write(f"\n{comment}")


if __name__ == "__main__":
    # 修改 --------------------------------------------------------------------------------------------------------------#
    incharge = "yulin"  # 负责人
    project = "Person"
    model_vision = "8-Person_2605-test-yolo26"  # 模型版本
    nc, names = 1, ["person"]
    vision_instruction = "[地铁]行人检测：测试。"
    # yaml 中 path 字段；合并后的 train.txt / val.txt 也写在此目录
    dataset_root = "/home/yulin/1-project/2-YOLO/ultralytics/sample"
    # 扫描路径：可与 dataset_root 相同；或 PCI 父目录（如 /data/0-data/8-Person-Metro）
    scan_paths = [dataset_root]
    # -------------------------------------------------------------------------------------------------------------------#

    os.chdir(SCRIPT_DIR)

    dataitem: dict[str, list[str]] = {}
    dataset_dictionary: dict[str, int] = {}
    train_files: list[Path] = []
    val_files: list[Path] = []
    prefixes: list[str | None] = []
    dataset_quantity = 0
    sample_sum = 0

    for scan_path in scan_paths:
        root = Path(scan_path)
        dataitem[root.name] = sorted(os.listdir(root))
        part_train, part_val, part_qty, part_prefixes = resolve_dataset_layout([scan_path], dataset_root)
        part_dict, part_sum = list_images(root, dataitem[root.name])

        train_files.extend(part_train)
        val_files.extend(part_val)
        prefixes.extend(part_prefixes)
        dataset_quantity += part_qty
        sample_sum += part_sum
        dataset_dictionary.update(part_dict)

    dataset_root_path = Path(dataset_root).resolve()
    train_count = merge_split_txts(train_files, dataset_root_path / "train.txt", prefixes)
    val_count = merge_split_txts(val_files, dataset_root_path / "val.txt", prefixes)

    header_lines = [
        f"#【Train dataset for {project}】",
        f"# Dataset Used: {dataset_quantity}",
        f"# Dataset Sum：{sample_sum}",
        f"# Train lines: {train_count}",
        f"# Val lines: {val_count}",
        f"# Date：{datetime.now()}",
    ]
    yaml_out = SCRIPT_DIR / f"0-{project}.yaml"
    write_dataset_yaml(yaml_out, dataset_root_path, nc, names, header_lines)

    stat_path = SCRIPT_DIR / "1-Dataset-stat.yaml"
    if not stat_path.exists():
        with open(stat_path, "w", encoding="utf-8") as file:
            yaml.dump(dataset_dictionary, file, default_flow_style=False, indent=4)

    with open(stat_path, encoding="utf-8") as file:
        dataset_stat = yaml.safe_load(file)

    change_dir = []
    for key in dataset_dictionary:
        if key not in dataset_stat:
            change_dir.append(key.strip())
        elif dataset_dictionary[key] != dataset_stat[key]:
            change_dir.append(key.strip())

    vision_path = SCRIPT_DIR / "2-Vision-info.yaml"
    with open(vision_path, "a", encoding="utf-8") as file:
        file.write("\n\n")
        update(file, incharge, model_vision, vision_instruction, dataset_quantity, change_dir)

    with open(stat_path, "w", encoding="utf-8") as file:
        file.write(f"#【Statics of the {project} dataset】\n")
        file.write(f"# Dataset Used: {dataset_quantity}\n")
        file.write(f"# Dataset Sum：{sample_sum}\n")
        file.write(f"# Date：{datetime.now()}\n")
        file.write("\n\n")
        yaml.dump(dataset_dictionary, file, default_flow_style=False, indent=4)

    print(f"✅ Wrote {yaml_out}")
    print(f"   path: {dataset_root_path}")
    print(f"   train: {train_count} lines → {dataset_root_path / 'train.txt'}")
    print(f"   val:   {val_count} lines → {dataset_root_path / 'val.txt'}")
