# 0-Agent-demo

Agent 工作目录：训练 / 验证 / 推理的需求、台账和包装脚本放这里，**不改仓库源码**。规范见 `.cursor/.skills/agent-autodemo/SKILL.md`（正文与 `~/.cursor/.skills/agent-auto_demo` 相同）。

```
0-Agent-demo/
├── _demo_train/     # 训练  → bash 0-QuickStart/0-train.sh
├── _demo_eval/      # 验证  → bash 0-QuickStart/2-eval.sh
└── _demo_infer/     # 推理  → bash 0-QuickStart/1-inference.sh
```

每个槽结构相同：`Customer.md` / `todo.md` / `customer/` / `util/` / `logs/` / `0-results/`。

确认对应槽 `Customer.md` 后再跑 `util/`，确认前不得开跑。导出走 `0-QuickStart/3-export.sh`，不设独立槽。
