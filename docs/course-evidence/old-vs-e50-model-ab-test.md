# 旧 6 类模型与 50 轮模型 A/B 对比记录

测试日期：2026-06-12

## 对比模型

旧模型：

```text
models/classroom_behaviour_6cls.pt
```

新模型：

```text
models/student_behaviour_v6_6cls_img960_e50_best.pt
```

统一置信度阈值：

```text
conf=0.25
```

## 测试素材

样例图片：

```text
D:\Documents\YOLOv8\yolov8_onnx\测试\1.jpg
D:\Documents\YOLOv8\yolov8_onnx\测试\2-6.jpg
D:\Documents\YOLOv8\yolov8_onnx\测试\6.jpg
```

样例视频：

```text
D:\Documents\YOLOv8\yolov8_onnx\测试\test.mp4
```

从视频中均匀抽取 8 帧：

```text
output/model_ab_compare/video_frames_8
```

## 样例图片结果

| 模型 | 图片数 | 检测框数 | 平均置信度 | 正常框 | 异常框 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 6 类模型 | 3 | 114 | 0.8661 | 66 | 48 |
| 50 轮模型 | 3 | 68 | 0.5391 | 47 | 21 |

类别统计：

| 类别 | 旧 6 类模型 | 50 轮模型 |
| --- | ---: | ---: |
| `Hand-raise` | 1 | 0 |
| `Reading` | 52 | 3 |
| `Writing` | 13 | 44 |
| `Useing-Phone` | 35 | 0 |
| `Head-down` | 10 | 21 |
| `Sleeping` | 3 | 0 |

每张图片检测框数：

| 图片 | 旧 6 类模型 | 50 轮模型 |
| --- | ---: | ---: |
| `1.jpg` | 38 | 23 |
| `2-6.jpg` | 40 | 20 |
| `6.jpg` | 36 | 25 |

输出目录：

```text
output/model_ab_compare/sample_images_old_6cls
output/model_ab_compare/sample_images_new_e50
```

对比图：

```text
output/model_ab_compare/compare_sample_images.jpg
```

## 视频抽帧结果

| 模型 | 抽帧数 | 检测框数 | 平均置信度 | 正常框 | 异常框 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 6 类模型 | 8 | 234 | 0.9127 | 136 | 98 |
| 50 轮模型 | 8 | 202 | 0.5248 | 160 | 42 |

类别统计：

| 类别 | 旧 6 类模型 | 50 轮模型 |
| --- | ---: | ---: |
| `Hand-raise` | 0 | 1 |
| `Reading` | 90 | 26 |
| `Writing` | 46 | 133 |
| `Useing-Phone` | 74 | 0 |
| `Head-down` | 12 | 42 |
| `Sleeping` | 12 | 0 |

每帧检测框数：

| 帧 | 旧 6 类模型 | 50 轮模型 |
| --- | ---: | ---: |
| `frame_000025.jpg` | 32 | 22 |
| `frame_000101.jpg` | 40 | 30 |
| `frame_000151.jpg` | 32 | 33 |
| `frame_000227.jpg` | 22 | 19 |
| `frame_000302.jpg` | 29 | 25 |
| `frame_000378.jpg` | 26 | 27 |
| `frame_000428.jpg` | 20 | 21 |
| `frame_000504.jpg` | 33 | 25 |

输出目录：

```text
output/model_ab_compare/video_frames8_old_6cls
output/model_ab_compare/video_frames8_new_e50
```

对比图：

```text
output/model_ab_compare/compare_video_frames8.jpg
```

## 结论

这轮 A/B 结果支持一个判断：旧 6 类模型在当前样例图片和样例视频上展示效果更强。

主要证据：

- 旧模型检测框更多。
- 旧模型平均置信度明显更高。
- 旧模型能输出 `Useing-Phone` 和 `Sleeping`，50 轮模型在这批样例中没有输出这两类。
- 50 轮模型明显偏向 `Writing` 和 `Head-down`，类别分布更单一。

需要注意：

- 检测框更多不一定代表全都正确，旧模型的 `Useing-Phone` 可能存在误检，需要人工看图确认。
- 50 轮模型在自己训练集验证集上指标较好，但对这些样例图和样例视频的泛化效果不如旧模型。

建议：

- 课程展示默认模型可以先切回旧模型 `models/classroom_behaviour_6cls.pt`。
- 50 轮模型保留为训练实验结果，不作为展示默认模型。
- 后续继续训练时，不建议只用 `Student Behaviour Detection v6` 继续堆轮数，应加入更多真实现场画面和 SCB 等更丰富场景数据。
