# Agent-autodemo TODO

- **updated**: 2026-11-09 15:08
- **work_root**: 0-Agent-demo/_demo_infer
- **slot**: _demo_infer
- **phase**: confirm
- **current**: 三槽骨架已建，等待确认 Customer.md
- **next**: 逐项确认 Customer.md §202611 后再推理
- **blocked**: Customer 未确认

## Meta
- conda: yolo
- entry: bash 0-QuickStart/1-inference.sh
- customer: Customer.md §202611
- spec: 无
- log_dir: 0-Agent-demo/_demo_infer/logs
- results_dir: 0-Agent-demo/_demo_infer/0-results

## Checklist
- [x] §0.1 工作目录门禁（完备 / 构建 / 跳过）
- [ ] §2 确认 / 填写 Customer.md
- [x] §5 发现并冻结本仓库入口
- [ ] §4 检查环境与产物路径
- [ ] §4 执行（按单元一行）
- [ ] §4 写回 0-results
- [ ] §4 交付摘要

## Runs
| id | status | artifact | note |
| --- | --- | --- | --- |
| 202611 | pending | | 待确认权重 / 输入 / 阈值 |

## Notes
- 本仓库入口映射：`0-QuickStart/1-inference.sh` → `0-QuickStart/scripts/inference.py`
- Agent 禁止改仓库源码；脚本只放 `util/`
- 确认前不得开跑
