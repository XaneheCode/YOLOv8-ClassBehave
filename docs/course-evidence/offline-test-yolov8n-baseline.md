# YOLOv8n 通用模型离线测试摘要

测试日期：2026-06-08

## 数据集

- 数据集目录：`datasets/REBALANCE-instance-new-student-classroom-activity-5.v1i.yolov8`
- 格式：YOLOv8
- 类别：`phone`、`sleep`、`study`
- 来源：Roboflow Universe, Student Classroom Activity v-2 相关导出

## 数据集规模

| 划分 | 图片数 | 标注文件数 | 标注框数 | 类别分布 |
| --- | ---: | ---: | ---: | --- |
| train | 2277 | 2277 | 4747 | phone 1428, sleep 1484, study 1835 |
| valid | 223 | 223 | 437 | phone 126, sleep 143, study 168 |
| test | 112 | 112 | 234 | phone 67, sleep 63, study 104 |

## 测试命令

```powershell
.\.venv\Scripts\python.exe .\scripts\offline_test_images.py --dataset datasets\REBALANCE-instance-new-student-classroom-activity-5.v1i.yolov8 --model yolov8n.pt --limit 0 --output-dir output\offline_test\student-classroom-activity-v2-yolov8n-full
```

## 输出文件

- 预测 CSV：`output/offline_test/student-classroom-activity-v2-yolov8n-full/predictions.csv`
- 标注结果图：`output/offline_test/student-classroom-activity-v2-yolov8n-full/*.jpg`

## 测试结果

- 测试图片数：112
- 预测记录数：429
- 有预测框的图片数：108
- 无预测框的图片数：4
- `sleep` 预测数：0

预测类别分布：

| 预测类别 | 数量 |
| --- | ---: |
| person | 206 |
| chair | 139 |
| dining table | 18 |
| umbrella | 10 |
| suitcase | 9 |
| book | 6 |
| tv | 6 |
| laptop | 4 |
| cell phone | 4 |
| dog | 4 |
| none | 4 |
| 其他 COCO 类别 | 19 |

## 结论

当前 `yolov8n.pt` 是 COCO 预训练通用模型，类别集中没有 `sleep`、`study` 等课堂行为类别。因此它能够检测 `person`、`chair` 等通用目标，但不能直接输出 `sleep` 类。

这次测试证明离线推理流程、图片读取、结果保存和标注图生成正常。若要在本课程设计中实现更准确的课堂睡觉识别，需要使用该数据集训练自定义 YOLO 模型，并用训练得到的 `best.pt` 替换后端启动参数中的 `yolov8n.pt`。

## 后续训练准备

原始 Roboflow 解压目录中的 `data.yaml` 使用了 `../train/images` 这类路径，在当前项目目录下直接训练会解析到错误位置。已新增本地训练配置：

```txt
configs/student-classroom-activity-local.yaml
```

后续训练建议使用该配置文件。
