# 自定义 YOLOv8n 课堂行为模型测试摘要

测试日期：2026-06-08

## 训练目的

通用 `yolov8n.pt` 是 COCO 预训练模型，不包含 `sleep`、`study` 等课堂行为类别。为使系统能够直接识别课堂睡觉行为，使用 Roboflow 下载的数据集进行轻量自定义训练。

## 数据集

- 数据集目录：`datasets/REBALANCE-instance-new-student-classroom-activity-5.v1i.yolov8`
- 本地训练配置：`tmp/student-classroom-activity-abs.yaml`
- 类别：`phone`、`sleep`、`study`
- test 集：112 张图片，234 个标注框，其中 `sleep` 63 个

## 训练命令

```powershell
.\.venv\Scripts\yolo.exe detect train data=tmp/student-classroom-activity-abs.yaml model=yolov8n.pt epochs=3 imgsz=320 batch=8 device=cpu workers=0 project=<workspace-output-training> name=student_sleep_yolov8n_e3 exist_ok=True
```

说明：训练时使用绝对路径版 YAML，是因为 Ultralytics 会把相对路径解析到全局 datasets 目录，导致当前项目中的数据集路径偏移。

## 训练输出

- 模型文件：`output/training/student_sleep_yolov8n_e3/weights/best.pt`
- 训练结果：`output/training/student_sleep_yolov8n_e3/results.csv`
- 混淆矩阵和曲线图：`output/training/student_sleep_yolov8n_e3/*.png`

## 验证集结果

第 3 个 epoch 的整体验证指标：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.824 |
| Recall | 0.787 |
| mAP50 | 0.842 |
| mAP50-95 | 0.568 |

各类别验证结果：

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| phone | 0.724 | 0.683 | 0.716 | 0.477 |
| sleep | 0.929 | 0.888 | 0.939 | 0.622 |
| study | 0.815 | 0.792 | 0.871 | 0.606 |

## test/images 离线测试命令

```powershell
.\.venv\Scripts\python.exe .\scripts\offline_test_images.py --dataset datasets\REBALANCE-instance-new-student-classroom-activity-5.v1i.yolov8 --model output\training\student_sleep_yolov8n_e3\weights\best.pt --limit 0 --conf 0.25 --output-dir output\offline_test\student-classroom-activity-v2-custom-e3
```

## test/images 离线测试结果

- 测试图片数：112
- 预测记录数：272
- 有预测框的图片数：112
- `sleep` 预测数：73

预测类别分布：

| 预测类别 | 数量 |
| --- | ---: |
| study | 121 |
| phone | 78 |
| sleep | 73 |

输出文件：

- 预测 CSV：`output/offline_test/student-classroom-activity-v2-custom-e3/predictions.csv`
- 标注结果图：`output/offline_test/student-classroom-activity-v2-custom-e3/*.jpg`

## 后端使用方式

后续双机实时监测建议使用训练后的自定义模型：

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model output\training\student_sleep_yolov8n_e3\weights\best.pt
```

这样后端检测结果可以直接出现 `sleep` 标签，现有 `SleepAnalyzer` 会把 `sleep` 作为疑似睡觉信号并按连续时间阈值触发报警。

