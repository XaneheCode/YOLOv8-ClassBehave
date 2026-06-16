# yolov8s 与 CrowdHuman 人体检测模型现场对比

日期：2026-06-15

## 对比目的

验证现有默认人体检测模型 `yolov8s.pt` 与 CrowdHuman 训练版 `yolov8n_best.pt` 在现场课堂照片上的人体检出能力差异。

## 模型

| 模型 | 路径 | 来源 |
| --- | --- | --- |
| YOLOv8s COCO | `yolov8s.pt` | Ultralytics COCO 预训练 |
| YOLOv8n CrowdHuman | `models/yolov8n_crowdhuman_best_2026-01-03.pt` | `yakhyo/yolov8-crowdhuman` |

CrowdHuman 权重校验：

```text
SHA256: 00AF512C66FC4E7B184DF7E171C8AC3BBB52CCF964706DB284D3A13586F7AE52
```

## 测试设置

```text
图片目录：datasets/field-photos-2026-06-12/images
测试图片：field_001.jpg 至 field_006.jpg
推理尺寸：imgsz=960
置信度阈值：conf=0.25
类别过滤：person
```

## 检出数量

| 图片 | YOLOv8s COCO | YOLOv8n CrowdHuman |
| --- | ---: | ---: |
| `field_001.jpg` | 14 | 85 |
| `field_002.jpg` | 18 | 76 |
| `field_003.jpg` | 10 | 10 |
| `field_004.jpg` | 9 | 7 |
| `field_005.jpg` | 13 | 15 |
| `field_006.jpg` | 5 | 4 |
| **合计** | **69** | **197** |

## 输出记录

```text
output/person_detector_compare/yolov8s_vs_crowdhuman_2026-06-15/summary.csv
output/person_detector_compare/yolov8s_vs_crowdhuman_2026-06-15/contact_sheet_yolov8s_vs_crowdhuman.jpg
output/person_detector_compare/yolov8s_vs_crowdhuman_2026-06-15/yolov8s_coco
output/person_detector_compare/yolov8s_vs_crowdhuman_2026-06-15/yolov8n_crowdhuman
```

## 初步结论

CrowdHuman 模型在 `field_001.jpg` 和 `field_002.jpg` 这类大教室远景场景中明显更适合，能补出大量远处和遮挡学生；但在实验室近景照片中，它不一定稳定优于 `yolov8s.pt`。

当前不建议马上替换默认模型。更稳妥的做法是继续保留 `yolov8s.pt` 作为默认基线，同时把 CrowdHuman 模型加入候选模型，在后续做更多现场视频抽帧测试。如果大教室远景是主要演示场景，再考虑将 CrowdHuman 版作为人体检测默认模型。
