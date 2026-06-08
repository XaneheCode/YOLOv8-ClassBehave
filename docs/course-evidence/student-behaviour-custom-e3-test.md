# Student Behaviour Detection 多状态模型测试记录

日期：2026-06-08

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

本轮按完整 3 轮训练执行，没有缩小训练轮数。

```powershell
.\.venv\Scripts\python.exe scripts\prepare_yolo_data_yaml.py --dataset "datasets\Student Behaviour Detection.v6i.yolov8" --output tmp\student-behaviour-detection-abs.yaml
.\.venv\Scripts\yolo.exe detect train data=tmp\student-behaviour-detection-abs.yaml model=yolov8n.pt epochs=3 imgsz=320 batch=8 device=cpu workers=0 project=output\training name=student_behaviour_yolov8n_e3 exist_ok=True
```

训练产物：

```text
output/training/student_behaviour_yolov8n_e3/weights/best.pt
```

## 3. 验证集指标

最终验证集整体指标：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.458 | 0.370 | 0.349 | 0.185 |

各类别指标：

| class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Using_phone | 0.476 | 0.220 | 0.202 | 0.0827 |
| bend | 0.496 | 0.634 | 0.573 | 0.354 |
| book | 0.575 | 0.428 | 0.404 | 0.183 |
| bow_head | 0.617 | 0.347 | 0.339 | 0.157 |
| hand-raising | 0.181 | 0.157 | 0.113 | 0.0391 |
| phone | 0.887 | 0.0613 | 0.126 | 0.0478 |
| raise_head | 0.641 | 0.653 | 0.635 | 0.366 |
| reading | 0.369 | 0.203 | 0.186 | 0.0828 |
| sleep | 0.396 | 0.360 | 0.477 | 0.194 |
| turn_head | 0.000 | 0.000 | 0.0403 | 0.0207 |
| upright | 0.604 | 0.932 | 0.875 | 0.560 |
| writing | 0.250 | 0.448 | 0.224 | 0.133 |

说明：本轮训练目标是扩展系统功能并得到可运行的多状态模型。由于只训练 3 轮，且部分类别样本较少或区分难度较高，`turn_head`、`phone`、`hand-raising` 等类别指标偏低。课程报告中可说明后续可通过增加训练轮数、提高输入分辨率或扩充样本提升准确率。

## 4. test/images 离线测试

命令：

```powershell
.\.venv\Scripts\python.exe scripts\offline_test_images.py --dataset "datasets\Student Behaviour Detection.v6i.yolov8" --model output\training\student_behaviour_yolov8n_e3\weights\best.pt --output-dir output\offline_test\student-behaviour-custom-e3 --limit 0 --conf 0.25
```

结果：

| item | value |
| --- | ---: |
| tested images | 292 |
| images with predictions | 292 |
| prediction rows | 9121 |
| normal candidates | 6087 |
| abnormal candidates | 3034 |

预测类别数量：

| label | count |
| --- | ---: |
| upright | 4196 |
| bow_head | 1575 |
| raise_head | 1078 |
| bend | 861 |
| book | 752 |
| Using_phone | 406 |
| sleep | 157 |
| reading | 61 |
| phone | 35 |

输出目录：

```text
output/offline_test/student-behaviour-custom-e3
```

其中包含 292 张标注后的测试图片和 `predictions.csv`。
