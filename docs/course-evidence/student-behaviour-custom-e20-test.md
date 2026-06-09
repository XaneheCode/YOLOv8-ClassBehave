# Student Behaviour Detection e20 多状态模型测试记录

日期：2026-06-09

## 1. 数据集

数据集目录：

```text
datasets/Student Behaviour Detection.v6i.yolov8
```

类别：

```text
Using_phone, bend, book, bow_head, hand-raising, phone, raise_head, reading, sleep, turn_head, upright, writing
```

数据规模：

| split | images | boxes |
| --- | ---: | ---: |
| train | 3192 | 118290 |
| valid | 581 | 27048 |
| test | 292 | 11615 |

## 2. 训练配置

本轮训练使用 20 轮和 640 输入尺寸，训练耗时约 3.902 小时。

```powershell
.\.venv\Scripts\python.exe scripts\prepare_yolo_data_yaml.py --dataset "datasets\Student Behaviour Detection.v6i.yolov8" --output tmp\student-behaviour-detection-abs.yaml
.\.venv\Scripts\yolo.exe detect train data=tmp\student-behaviour-detection-abs.yaml model=yolov8n.pt epochs=20 imgsz=640 batch=8 device=cpu workers=0 project=output\training name=student_behaviour_yolov8n_e20 exist_ok=True
```

训练产物：

```text
output/training/student_behaviour_yolov8n_e20/weights/best.pt
```

## 3. 验证集指标

最终验证集整体指标：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.739 | 0.685 | 0.709 | 0.466 |

各类别指标：

| class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Using_phone | 0.747 | 0.602 | 0.644 | 0.312 |
| bend | 0.778 | 0.826 | 0.857 | 0.672 |
| book | 0.806 | 0.754 | 0.817 | 0.493 |
| bow_head | 0.870 | 0.927 | 0.930 | 0.591 |
| hand-raising | 0.482 | 0.526 | 0.511 | 0.288 |
| phone | 0.867 | 0.414 | 0.448 | 0.234 |
| raise_head | 0.811 | 0.875 | 0.894 | 0.648 |
| reading | 0.607 | 0.497 | 0.550 | 0.343 |
| sleep | 0.924 | 0.854 | 0.904 | 0.579 |
| turn_head | 0.630 | 0.400 | 0.501 | 0.338 |
| upright | 0.879 | 0.971 | 0.969 | 0.770 |
| writing | 0.466 | 0.577 | 0.478 | 0.321 |

## 4. test/images 离线测试

命令：

```powershell
.\.venv\Scripts\python.exe scripts\offline_test_images.py --dataset "datasets\Student Behaviour Detection.v6i.yolov8" --model output\training\student_behaviour_yolov8n_e20\weights\best.pt --output-dir output\offline_test\student-behaviour-custom-e20 --limit 0 --conf 0.25
```

结果：

| item | value |
| --- | ---: |
| tested images | 292 |
| images with predictions | 292 |
| prediction rows | 12797 |
| normal candidates | 7059 |
| abnormal candidates | 5738 |

预测类别数量：

| label | count |
| --- | ---: |
| upright | 3973 |
| bow_head | 3172 |
| raise_head | 1398 |
| book | 1239 |
| bend | 1062 |
| Using_phone | 754 |
| reading | 449 |
| phone | 367 |
| turn_head | 193 |
| sleep | 190 |

输出目录：

```text
output/offline_test/student-behaviour-custom-e20
```

该模型相较 e3/imgsz320 版本有明显提升，可作为课程成果展示默认模型。
