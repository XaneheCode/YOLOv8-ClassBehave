# 现场监控视频人体预标注记录

日期：2026-06-15

## 数据来源

```text
D:\Videos\屏幕录制\屏幕录制 2026-06-15 181147.mp4
```

视频信息：

```text
时长：161 秒
帧率：30 FPS
分辨率：2558x1438
抽帧间隔：5 秒
抽帧数量：33 张
```

抽帧数据集：

```text
datasets/field-video-frames-2026-06-15-181147
```

## 标注目标

当前 YOLO 只负责人体检测，因此本批数据只标注 1 类：

```text
0: person
```

## 预标注流程

1. 使用 CrowdHuman 训练版 YOLOv8n 模型生成原始人体框。
2. 使用较低阈值的 CrowdHuman 推理补充远处小目标。
3. 使用 `yolov8s.pt` 的 `person` 检测结果补充漏检。
4. 对三组框进行去重合并，得到第一版可训练标注。
5. 逐页查看可视化结果，确认大教室远景中的后排、远处和遮挡学生基本被覆盖。

## 使用模型

| 模型 | 用途 |
| --- | --- |
| `models/yolov8n_crowdhuman_best_2026-01-03.pt` | 主人体预标注模型 |
| `yolov8s.pt` | 辅助补漏模型 |

## 输出文件

最终训练标签：

```text
datasets/field-video-frames-2026-06-15-181147/labels
```

CrowdHuman 原始标签：

```text
datasets/field-video-frames-2026-06-15-181147/labels_crowdhuman_raw
```

合并补漏标签备份：

```text
datasets/field-video-frames-2026-06-15-181147/labels_person_auto_merged
```

数据配置：

```text
datasets/field-video-frames-2026-06-15-181147/person-field-video.yaml
```

可视化与统计：

```text
output/field_video_person_prelabels_2026-06-15-181147/summary.csv
output/field_video_person_prelabels_2026-06-15-181147/person_merged_visuals
output/field_video_person_prelabels_2026-06-15-181147/review_contact_sheets
```

## 标注统计

| 项目 | 数量 |
| --- | ---: |
| 图片数量 | 33 |
| CrowdHuman 原始框 | 3059 |
| `yolov8s` 辅助框 | 727 |
| 合并后最终框 | 3442 |
| 单图最少框数 | 84 |
| 单图最多框数 | 124 |
| 单图平均框数 | 104.3 |

## 说明

这是一版自动预标注加视觉复核后的初始人体框数据，适合用于 CrowdHuman 模型的现场适配微调。由于教室远景中目标密集、遮挡严重，仍建议训练前后抽查若干张单图，重点检查后排远处学生、画面边缘半身人物和讲台附近走动人员。
