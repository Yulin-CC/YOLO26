#!/bin/bash
###
 # @Author: AI产品研发组 蔡雨霖
 # @Date: 2026-07-09 14:00:00
 # @LastEditTime: 2026-07-09
 # @Description: YOLO26 检测/分割评估启动脚本
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
dataset="data/0-coco8.yaml"
#--------------------------------------#


#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yulin


#---------------#
# 运行评估程序
#---------------#
cd "$ROOT_DIR"

# 固定路径（无需修改）
output="./runs/${task}/2-eval/$project"

mkdir -p "$output"
python 0-QuickStart/scripts/eval.py \
  --task      "$task" \
  --devices   "$devices" \
  --weights   "$weights" \
  --dataset   "$dataset" \
  --output    "$output" \
  --project   "$project" \
  --imgsz     "$imgsz" \
  --conf      "$conf" \
  --iou       "$iou"
