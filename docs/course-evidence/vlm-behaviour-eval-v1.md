# VLM 行为分类准确率测试 v1

测试日期：2026-06-16

## 测试目标

验证“YOLO 人体检测 + 编号人体拼图 + 视觉大模型行为分类”方案中，大模型对六类课堂行为的分类准确率。

## 测试集

- 数据目录：`datasets/vlm-behaviour-eval-v1`
- 人工基准：`annotations/initial_human_seed_labels.csv`
- 有效目标：84 个
- 忽略目标：6 个
- 编号拼图：3 张

六类标签：

- `Hand-raise`
- `Reading`
- `Writing`
- `Useing-Phone`
- `Head-down`
- `Sleeping`

## 测试方法

对同一批编号人体拼图分别调用千问和 GPT 兼容视觉接口，要求模型只输出六类行为标签。随后以人工复核标签作为基准，逐编号计算预测是否正确。

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\run_vlm_behaviour_eval.py --providers qwen gpt
```

输出文件：

- `datasets/vlm-behaviour-eval-v1/annotations/vlm_predictions_comparison.csv`
- `datasets/vlm-behaviour-eval-v1/annotations/vlm_accuracy_summary.csv`
- `datasets/vlm-behaviour-eval-v1/annotations/qwen_predictions.csv`
- `datasets/vlm-behaviour-eval-v1/annotations/gpt_predictions.csv`

## 总体结果

| 模型 | 有效目标 | 正确数 | 准确率 | 漏返回 |
| --- | ---: | ---: | ---: | ---: |
| GPT | 84 | 46 | 54.76% | 0 |
| 千问 | 84 | 22 | 26.19% | 0 |

## 分类别结果

| 模型 | 类别 | 样本数 | 正确数 | 准确率 |
| --- | --- | ---: | ---: | ---: |
| GPT | Hand-raise | 2 | 2 | 100.00% |
| GPT | Reading | 16 | 9 | 56.25% |
| GPT | Writing | 35 | 27 | 77.14% |
| GPT | Useing-Phone | 22 | 3 | 13.64% |
| GPT | Head-down | 6 | 4 | 66.67% |
| GPT | Sleeping | 3 | 1 | 33.33% |
| 千问 | Hand-raise | 2 | 0 | 0.00% |
| 千问 | Reading | 16 | 2 | 12.50% |
| 千问 | Writing | 35 | 6 | 17.14% |
| 千问 | Useing-Phone | 22 | 12 | 54.55% |
| 千问 | Head-down | 6 | 1 | 16.67% |
| 千问 | Sleeping | 3 | 1 | 33.33% |

## 主要观察

1. GPT 的总体准确率更高，尤其在 `Writing` 上表现较好。
2. GPT 对 `Useing-Phone` 明显保守，常把看手机误判为 `Writing` 或 `Reading`。
3. 千问对 `Useing-Phone` 更敏感，但把大量 `Writing` 错判为 `Useing-Phone`、`Reading` 或 `Head-down`。
4. `Hand-raise` 和 `Sleeping` 当前样本太少，不能单独作为稳定结论。
5. 现阶段报告中可以把该测试作为“第一轮小样本验证”，后续建议扩大人工基准样本并优化提示词。

## 后续改进方向

1. 扩大人工复核基准，从 84 个有效目标扩展到 300 个以上。
2. 对 `Useing-Phone` 增加更明确的提示词和少量反例规则，减少 GPT 漏判。
3. 对千问增加“不要把普通伏案写字误判为手机”的约束。
4. 可尝试多模型融合：GPT 负责 `Writing/Reading`，千问辅助发现疑似 `Useing-Phone`，最后用规则合并。
