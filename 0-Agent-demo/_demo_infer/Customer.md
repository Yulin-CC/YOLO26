# Customer · 推理需求

> 用户自写，或 Agent 引导确认后填写。按月份追加，旧批次不删。未确认项写「待确认」，确认前不得开跑。

## 202611

1. 目的：待确认

2. 环境：`conda activate yolo`；设备：待确认

3. 任务：infer

4. 本仓库入口：`bash 0-QuickStart/1-inference.sh`（本槽包装 `util/1-inference.sh`）

5. 模型 / ckpt：

    | tag | path |
    | --- | --- |
    | 待确认 | 待确认（QuickStart 当前默认 `weights/yolo26s-seg.pt`，须确认后才用） |

6. 数据：待确认（QuickStart 当前默认 `./sample/images`）

7. 本批规格：无

8. 成功标准：待确认

9. 结果：

    a. 文档写回：`0-Agent-demo/_demo_infer/0-results/`
    b. 可视化或权重：`<dataset>/repro/`（单文件则在文件所在目录 `repro/`）
    c. 日志：`0-Agent-demo/_demo_infer/logs/`
