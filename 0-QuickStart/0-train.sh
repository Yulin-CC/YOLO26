#!/bin/bash
###
 # @Author: AI产品研发组 蔡雨霖
 # @Date: 2026-07-09 14:00:00
 # @LastEditTime: 2026-07-09
 # @Description: YOLO26 检测/分割训练启动脚本
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$WORK_DIR/.." && pwd)

#--------------------------------------#
# 需要修改的值
#--------------------------------------#
task=detect                           # detect | segment
#--------------------------------------#
devices="0"                            # GPU 设备 ID（多卡如 "0,1"）
#--------------------------------------#
project="yolo26-2607-testv1"           # 实验名 → ./runs/${task}/0-train/$project
#--------------------------------------#
model="weights/yolo26s.pt"         # 预训练权重
dataset="data/0-Person.yaml"           # 数据集 yaml
#--------------------------------------#


#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yulin


#---------------#
# 运行训练程序
#---------------#
cd "$ROOT_DIR"

# 固定路径（无需修改）
output="./runs/${task}/0-train/$project"

mkdir -p "$output"
python 0-QuickStart/scripts/train.py \
  --task         "$task" \
  --devices      "$devices" \
  --project      "$project" \
  --model        "$model" \
  --dataset      "$dataset"
