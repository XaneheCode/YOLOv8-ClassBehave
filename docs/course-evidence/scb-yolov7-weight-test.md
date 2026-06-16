# SCB YOLOv7 权重测试记录

测试日期：2026-06-12

## 测试对象

- 权重：`models/yolov7_HRW_4.2k.pt`
- 来源：SCB 相关训练权重
- 本地测试代码：临时使用 `tmp/yolov7` 中的 YOLOv7 推理脚本

## 兼容性结论

该权重不能被当前项目的 Ultralytics YOLOv8 后端直接加载。

直接运行：

```powershell
YOLO("models/yolov7_HRW_4.2k.pt")
```

报错：

```text
ModuleNotFoundError: No module named 'models.yolo'
```

原因是该 `.pt` 是 YOLOv7 checkpoint，依赖 YOLOv7 仓库中的 `models.yolo` 结构，不是 YOLOv8/Ultralytics 原生权重。

## 测试处理

为验证模型效果，临时下载 YOLOv7 推理代码到：

```text
tmp/yolov7
```

由于 YOLOv7 旧脚本不兼容中文路径，测试时将样例图和权重复制到了英文临时目录。

同时，为兼容当前 PyTorch 版本，对临时 YOLOv7 代码做了测试用补丁：

```python
torch.load(..., weights_only=False)
```

该补丁仅用于临时测试，不属于主项目后端代码。

## 样例图片测试结果

测试图片：

```text
D:\Documents\YOLOv8\yolov8_onnx\测试\1.jpg
D:\Documents\YOLOv8\yolov8_onnx\测试\2-6.jpg
D:\Documents\YOLOv8\yolov8_onnx\测试\6.jpg
```

YOLOv7 HRW 权重输出：

| 图片 | rise hand | read | write | 合计 |
| --- | ---: | ---: | ---: | ---: |
| `1.jpg` | 1 | 22 | 14 | 37 |
| `2-6.jpg` | 0 | 28 | 20 | 48 |
| `6.jpg` | 0 | 26 | 23 | 49 |
| 合计 | 1 | 76 | 57 | 134 |

输出文件：

```text
output/scb_weight_test/yolov7_HRW_samples/1.jpg
output/scb_weight_test/yolov7_HRW_samples/2-6.jpg
output/scb_weight_test/yolov7_HRW_samples/6.jpg
output/scb_weight_test/yolov7_HRW_samples/labels/*.txt
```

## 当前 YOLOv8 50 轮模型对比

同样 3 张图片，当前 50 轮模型：

```text
models/student_behaviour_v6_6cls_img960_e50_best.pt
```

输出统计：

| 类别 | 数量 |
| --- | ---: |
| `Head-down` | 21 |
| `Reading` | 3 |
| `Writing` | 44 |
| 合计 | 68 |

输出文件：

```text
output/scb_weight_test/current_yolov8_e50_samples/*.jpg
output/scb_weight_test/current_yolov8_e50_samples/predictions.csv
```

## 结论

SCB 的 `yolov7_HRW_4.2k.pt` 可以通过 YOLOv7 代码推理，但不适合作为当前项目的默认模型直接替换。

原因：

- 它不是 YOLOv8 权重，当前后端不能直接加载。
- 它只有 3 类：举手、阅读、写字。
- 它不包含当前系统关键异常类：低头、睡觉、使用手机。
- 若要接入，需要单独写 YOLOv7 推理适配层和类别映射。

建议：

- 将该权重作为“正常课堂行为识别”的参考对比模型。
- 不建议替换当前 6 类 YOLOv8 50 轮模型。
- 更推荐使用 SCB 数据集整理为 6 类后，在当前 YOLOv8 模型上继续微调。
