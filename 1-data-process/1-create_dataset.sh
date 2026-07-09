#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-07-09 14:00:00
 # @LastEditTime: 2026-07-09
 # @Description: YOLO 数据集自动整理（格式转换 + 校验 + 划分）
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

#--------------------------------------#
# 需要修改的值
#--------------------------------------#
Path='../sample'
#--------------------------------------#
prefix="COCO2017"                      # 重命名前缀（留空则自动从文件夹名提取）
#--------------------------------------#
task="detect"                          # detect | segment
#--------------------------------------#
platform="labelme"                     # labelimg | labelme
#--------------------------------------#
yaml='../sample/classes.yaml'          # 类别 yaml（labelme 必填；labelimg 可选）
#--------------------------------------#


#---------------#
# 参数校验
#---------------#
if [ "$task" = "segment" ] && [ "$platform" = "labelimg" ]; then
    echo ""
    echo "============================================================"
    echo "❌ [参数检查] 失败，流水线已停止"
    echo "   原因: segment 任务仅支持 platform=labelme"
    echo "============================================================"
    exit 1
fi
if [ "$platform" = "labelme" ] && [ -z "$yaml" ]; then
    echo ""
    echo "============================================================"
    echo "❌ [参数检查] 失败，流水线已停止"
    echo "   原因: platform=labelme 时必须设置 yaml"
    echo "============================================================"
    exit 1
fi
if [ "$platform" = "labelme" ] && [ ! -f "$yaml" ]; then
    echo ""
    echo "============================================================"
    echo "❌ [参数检查] 失败，流水线已停止"
    echo "   原因: 类别 yaml 不存在: $yaml"
    echo "============================================================"
    exit 1
fi


#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yulin


#---------------#
# 运行整理程序
#---------------#
cd "$WORK_DIR"

ARGS=(--path "$Path" --task "$task" --platform "$platform")
[ -n "$prefix" ] && ARGS+=(--prefix "$prefix")
[ -n "$yaml" ] && ARGS+=(--yaml "$yaml")

python utils/0-organize_dataset.py "${ARGS[@]}"
status=$?
if [ "$status" -ne 0 ]; then
    exit "$status"
fi
