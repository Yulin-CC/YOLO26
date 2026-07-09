#!/bin/bash
###
 # @Author: AI产品研发组 蔡雨霖
 # @Date: 2026-07-09 14:00:00
 # @LastEditTime: 2026-07-09
 # @Description: YOLO26 检测/分割模型导出（ONNX）
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$WORK_DIR/.." && pwd)

#--------------------------------------#
# 需要修改的值
#--------------------------------------#
task=detect                                                      # detect | segment
#--------------------------------------#
devices=0                                                        # GPU 设备 ID
#--------------------------------------#
project="exp01-coco8-yolo26"                                     # 实验名（与训练时一致）
#--------------------------------------#
weights="./weights/yolo26s.pt"         # 训练权重完整路径
#--------------------------------------#
imgsz=640                                                        # 导出图像尺寸
#--------------------------------------#


#---------------#
# 切换到虚拟环境（conda 路径因机器而异，clone 后请修改 CONDA_BASE）
#   常见：$HOME/miniconda3 | $HOME/anaconda3 | /opt/conda
#   查找：dirname "$(dirname "$(which conda)")"
#   环境名 yolo 须与 README §1 中 conda create -n 一致
#---------------#
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"   # 本机示例：/home/ubuntu/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate yolo


#---------------#
# 运行导出程序
#---------------#
cd "$ROOT_DIR"

python 0-QuickStart/scripts/export.py \
  --task      "$task" \
  --devices   "$devices" \
  --weights   "$weights" \
  --project   "$project" \
  --imgsz     "$imgsz" "$imgsz" \
  --format    onnx
