# CrowdHuman 人体检测模型现场微调记录 v1

日期：2026-06-15

## 训练目的

以 CrowdHuman 预训练人体检测模型为基础，加入本项目真实课堂监控抽帧数据进行现场适配微调，使人体检测更适应固定监控机位、大教室远景、后排小目标、遮挡人物和走动人物。

## 训练数据

合并数据集：

```text
datasets/field-person-finetune-v1
```

来源：

| 来源 | 图片 |
| --- | ---: |
| `field-video-frames-2026-06-15-181147` | 33 |
| `field-video-frames-2026-06-15-200958` | 24 |
| `field-video-frames-2026-06-15-201717` | 26 |
| **合计** | **83** |

划分：

| 划分 | 图片 | 人体框 |
| --- | ---: | ---: |
| train | 66 | 5936 |
| val | 17 | 1500 |
| **合计** | **83** | **7436** |

类别：

```text
0: person
```

## 云端训练环境

```text
GPU: NVIDIA GeForce RTX 3060 12GB
Python: 3.10.12
PyTorch: 2.5.1+cu124
CUDA: 12.4
Ultralytics: 8.3.230
训练目录: /data/classroom-person-finetune
```

## 训练命令

```bash
yolo detect train \
  model=models/yolov8n_crowdhuman_best_2026-01-03.pt \
  data=datasets/field-person-finetune-v1/data.yaml \
  imgsz=960 \
  epochs=50 \
  batch=8 \
  device=0 \
  workers=4 \
  optimizer=AdamW \
  lr0=0.001 \
  lrf=0.01 \
  patience=15 \
  freeze=10 \
  close_mosaic=10 \
  cache=True \
  amp=False \
  project=output/training \
  name=field_person_crowdhuman_finetune_v1 \
  exist_ok=True
```

说明：`amp=False` 是为了跳过 Ultralytics AMP 自检阶段的外网权重下载，避免训练被网络阻塞。

## 训练结果

输出权重：

```text
models/yolov8n_crowdhuman_field_finetune_v1_2026-06-15_best.pt
models/yolov8n_crowdhuman_field_finetune_v1_2026-06-15_last.pt
```

`best.pt` 校验：

```text
SHA256: 003540CB1E3ABD603E7B3F0A66A0BDEA6BE84A26A34CADD30D0BEF6B7FBDBC05
```

最终验证指标：

| 模型 | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| CrowdHuman 原始权重 | 0.833 | 0.866 | 0.931 | 0.855 |
| 现场微调 v1 | 0.905 | 0.872 | 0.954 | 0.795 |

解释：现场微调后，Precision、Recall 和 mAP50 有提升，说明模型更倾向于检出本项目现场画面中的人体；但 mAP50-95 下降，说明高 IoU 阈值下框的位置贴合度不一定优于原始权重。由于本批标签本身来自自动预标注和复核，验证指标只作为参考，最终应结合独立现场照片视觉效果判断。

## 独立现场照片对比

使用未参与本次训练的 6 张现场照片进行对比：

```text
datasets/field-photos-2026-06-12/images
```

推理设置：

```text
imgsz=960
conf=0.25
classes=person
```

| 图片 | CrowdHuman 原始权重 | 现场微调 v1 |
| --- | ---: | ---: |
| `field_001.jpg` | 85 | 131 |
| `field_002.jpg` | 76 | 100 |
| `field_003.jpg` | 10 | 13 |
| `field_004.jpg` | 7 | 11 |
| `field_005.jpg` | 15 | 20 |
| `field_006.jpg` | 4 | 10 |
| **合计** | **197** | **285** |

可视化输出：

```text
output/person_detector_compare/crowdhuman_base_vs_field_finetune_v1_2026-06-15/contact_sheet_crowdhuman_base_vs_field_finetune_v1.jpg
```

## 初步结论

现场微调 v1 明显提高了现场照片中的人体检出数量，尤其对大教室远景、后排小目标和实验室遮挡人物更敏感。它更适合“尽量少漏人”的系统目标，但也可能带来更多低置信度或边缘误检，需要在后端通过置信度阈值、NMS 和大模型二阶段分类进一步过滤。

当前建议：先将该模型作为候选人体检测模型测试，不立即覆盖原始 CrowdHuman 权重。若实机演示更关注少漏人，可将默认人体模型切换到现场微调 v1；若更关注框的稳定和保守，可继续使用原始 CrowdHuman 权重。
