# 现场监控视频人体预标注记录 200958

日期：2026-06-15

## 数据来源

```text
D:\Videos\屏幕录制\屏幕录制 2026-06-15 200958.mp4
```

视频信息：

```text
时长：115.57 秒
帧率：30 FPS
分辨率：2556x1434
抽帧间隔：5 秒
抽帧数量：24 张
```

抽帧数据集：

```text
datasets/field-video-frames-2026-06-15-200958
```

## 标注目标

当前 YOLO 只负责人体检测，因此本批数据只标注 1 类：

```text
0: person
```

## 预标注与修正流程

1. 使用 CrowdHuman 训练版 YOLOv8n 模型生成原始人体框。
2. 使用较低阈值的 CrowdHuman 推理补充后排和远处小目标。
3. 使用 `yolov8s.pt` 的 `person` 检测结果辅助补漏。
4. 对多组框进行去重合并，生成最终 YOLOv8 标签。
5. 对所有抽帧生成可视化复核图，并逐页检查后排密集区域、画面边缘和讲台附近走动人员。

## 使用模型

| 模型 | 用途 |
| --- | --- |
| `models/yolov8n_crowdhuman_best_2026-01-03.pt` | 主人体预标注模型 |
| `yolov8s.pt` | 辅助补漏模型 |

## 输出文件

最终训练标签：

```text
datasets/field-video-frames-2026-06-15-200958/labels
```

CrowdHuman 原始标签：

```text
datasets/field-video-frames-2026-06-15-200958/labels_crowdhuman_raw
```

合并补漏标签备份：

```text
datasets/field-video-frames-2026-06-15-200958/labels_person_auto_merged
```

数据配置：

```text
datasets/field-video-frames-2026-06-15-200958/person-field-video.yaml
```

可视化与统计：

```text
output/field_video_person_prelabels_2026-06-15-200958/summary.csv
output/field_video_person_prelabels_2026-06-15-200958/person_merged_visuals
output/field_video_person_prelabels_2026-06-15-200958/review_contact_sheets
```

## 标注统计

| 项目 | 数量 |
| --- | ---: |
| 图片数量 | 24 |
| CrowdHuman 原始框 | 1564 |
| `yolov8s` 辅助框 | 404 |
| 合并后最终框 | 1859 |
| 单图最少框数 | 68 |
| 单图最多框数 | 88 |
| 单图平均框数 | 77.46 |
| 标签格式错误 | 0 |

## 说明

这是一版自动预标注加视觉复核后的初始人体框数据，适合与上一段现场视频抽帧数据合并，用于 CrowdHuman 人体检测模型的现场适配微调。当前标注重点是提高大教室远景、后排小目标、遮挡人物和走动人物的检出能力。
