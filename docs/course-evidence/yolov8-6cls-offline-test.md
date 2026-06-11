# YOLOv8 六类课堂行为模型离线测试记录

## 模型

- 来源：`D:\Documents\YOLOv8\yolov8_onnx\models\best_last.pt`
- 本项目路径：`models/classroom_behaviour_6cls.pt`
- 类别：`Hand-raise`、`Reading`、`Writing`、`Useing-Phone`、`Head-down`、`Sleeping`

## 测试命令

```powershell
.\.venv\Scripts\python.exe scripts\offline_test_images.py --images "D:\Documents\YOLOv8\yolov8_onnx\测试" --model models\classroom_behaviour_6cls.pt --output-dir output\offline_test\yolov8-6cls --limit 0 --conf 0.25
```

## 输出

- 测试图片数：3
- 预测记录数：114
- 预测 CSV：`output/offline_test/yolov8-6cls/predictions.csv`
- 标注图片：`output/offline_test/yolov8-6cls/*.jpg`

## 类别统计

| 类别 | 状态 | 数量 |
| --- | --- | ---: |
| `Hand-raise` | 正常 | 1 |
| `Reading` | 正常 | 52 |
| `Writing` | 正常 | 13 |
| `Useing-Phone` | 异常 | 35 |
| `Head-down` | 异常 | 10 |
| `Sleeping` | 异常 | 3 |

## 课程说明

该测试用于证明六类模型能在本项目环境中加载并输出课堂行为类别。双机实时演示时，前端 GUI 负责摄像头采集与网络发送，后端 GUI 负责调用同一模型进行实时检测、目标级异常标红和报警记录。

报警状态是整帧提示，用于界面状态、截图和 CSV 日志；检测框颜色按每个目标独立判断，异常目标显示红色，正常目标保持绿色。
