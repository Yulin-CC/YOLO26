#!/bin/bash
###
 # @Author: AI产品研发组 蔡雨霖
 # @Date: 2026-11-09
 # @LastEditTime: 2026-11-09
 # @Description: 本槽评估包装。只转发 0-QuickStart/2-eval.sh，不改源码。
 # @Command: bash 0-Agent-demo/_demo_eval/util/2-eval.sh
###
UTIL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEMO_DIR="$(cd "$UTIL_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"

if grep -q '待确认' "$DEMO_DIR/Customer.md" && [ "${FORCE_RUN:-0}" != "1" ]; then
  echo "ERROR: Customer.md 仍有「待确认」，确认前不得开跑。"
  echo "  入口: bash 0-QuickStart/2-eval.sh"
  echo "  确认后可编辑 Customer.md，或 FORCE_RUN=1 强制转发。"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$DEMO_DIR/logs"
bash "$PROJECT_ROOT/0-QuickStart/2-eval.sh"
