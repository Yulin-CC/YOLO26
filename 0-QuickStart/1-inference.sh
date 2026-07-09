#!/bin/bash
###
 # @Author: AI产品研发组 蔡雨霖
 # @Date: 2026-07-09 14:00:00
 # @LastEditTime: 2026-07-09
 # @Description: YOLO26 检测/分割推理启动脚本
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$WORK_DIR/.." && pwd)

#--------------------------------------#
# 需要修改的值
#--------------------------------------#
task=segment                             # detect | segment
#--------------------------------------#
devices=0                              # GPU 设备 ID
#--------------------------------------#
weights="./weights/yolo26s-seg.pt"         # 训练权重完整路径
#--------------------------------------#
dataset="./sample/images"              # 测试输入（单文件或目录）
#--------------------------------------#

#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yulin


#---------------#
# 运行推理程序
#---------------#
cd "$ROOT_DIR"

# 固定路径（无需修改，cd 后判断确保相对路径正确）
# 单文件输入 → <文件所在目录>/repro/
# 目录输入   → <目录>/repro/
if [ -f "$dataset" ]; then
    output="$(dirname "$dataset")/repro/"
elif [ -d "$dataset" ]; then
    output="$dataset/repro"
else
    echo "Error: dataset not found: $dataset"
    exit 1
fi

mkdir -p "$output"
PREDICT_ARGS=(
  --task      "$task"
  --devices   "$devices"
  --weights   "$weights"
  --dataset   "$dataset"
  --output    "$output"
)
python 0-QuickStart/scripts/inference.py "${PREDICT_ARGS[@]}"
