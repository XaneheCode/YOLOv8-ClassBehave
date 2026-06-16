# YOLOv8 样例图片与视频验证记录

## 验证时间

2026-06-11

## 模型

- 模型路径：`models/classroom_behaviour_6cls.pt`
- 来源：`D:\Documents\YOLOv8\yolov8_onnx\models\best_last.pt`
- 置信度阈值：`0.25`
- 类别：`Hand-raise`、`Reading`、`Writing`、`Useing-Phone`、`Head-down`、`Sleeping`

## 样例来源

```text
D:\Documents\YOLOv8\yolov8_onnx\测试
```

样例文件：

- `1.jpg`
- `2-6.jpg`
- `6.jpg`
- `test.mp4`

## 图片验证

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\offline_test_images.py --images "D:\Documents\YOLOv8\yolov8_onnx\测试" --model models\classroom_behaviour_6cls.pt --output-dir output\offline_test\yolov8-6cls-images --limit 0 --conf 0.25
```

输出文件：

- 标注图片目录：`output/offline_test/yolov8-6cls-images/`
- 预测 CSV：`output/offline_test/yolov8-6cls-images/predictions.csv`

图片统计：

| 项目 | 数值 |
| --- | ---: |
| 测试图片数 | 3 |
| 预测记录数 | 114 |
| 正常记录数 | 66 |
| 异常记录数 | 48 |

图片类别统计：

| 类别 | 状态 | 数量 |
| --- | --- | ---: |
| `Hand-raise` | 正常 | 1 |
| `Reading` | 正常 | 52 |
| `Writing` | 正常 | 13 |
| `Useing-Phone` | 异常 | 35 |
| `Head-down` | 异常 | 10 |
| `Sleeping` | 异常 | 3 |

## 视频验证

视频信息：

| 项目 | 数值 |
| --- | ---: |
| 文件 | `D:\Documents\YOLOv8\yolov8_onnx\测试\test.mp4` |
| 帧数 | 529 |
| FPS | 25.0 |
| 分辨率 | 1920x1080 |

输出文件：

- 标注视频：`output/offline_test/yolov8-6cls-video/test_annotated.mp4`
- 预测 CSV：`output/offline_test/yolov8-6cls-video/predictions.csv`
- 摘要 JSON：`output/offline_test/yolov8-6cls-video/summary.json`
- 关键帧截图：`output/offline_test/yolov8-6cls-video/key_frames/`

视频统计：

| 项目 | 数值 |
| --- | ---: |
| 处理帧数 | 529 |
| 有检测结果的帧数 | 529 |
| 有异常候选的帧数 | 529 |
| 预测记录数 | 15850 |
| 正常记录数 | 8722 |
| 异常记录数 | 7128 |

视频类别统计：

| 类别 | 状态 | 数量 |
| --- | --- | ---: |
| `Hand-raise` | 正常 | 6 |
| `Reading` | 正常 | 6324 |
| `Writing` | 正常 | 2392 |
| `Useing-Phone` | 异常 | 5547 |
| `Head-down` | 异常 | 765 |
| `Sleeping` | 异常 | 816 |

## 结论

样例图片和视频都能加载六类 YOLOv8 模型并输出检测框。样例视频每一帧都有检测结果，标注视频可用于展示模型识别效果。

实时 GUI 烟测中如果没有框，主要原因通常是现场摄像头画面中目标太远、太小或角度与样例差异较大，导致置信度低于阈值。样例验证结果可作为模型本身可用的证据。
