# 多学生多状态课堂行为检测扩展设计

日期：2026-06-08

## 1. 背景

当前系统已经完成双机图像传输、后端 YOLO 检测、睡觉异常分析、报警截图和报警日志。现有版本的异常判断以“整帧是否报警”为主：只要检测到睡觉或符合规则的疑似趴桌目标，所有检测框都会跟随全局报警状态变红。

本次扩展目标是使用 `datasets/Student Behaviour Detection.v6i.yolov8` 数据集，把模型能力从单一睡觉检测扩展为课堂多人多状态检测。系统需要在同一画面中分别显示每个学生或行为目标的状态，并且只把异常状态对应的检测框标红。

## 2. 目标

1. 使用 `Student Behaviour Detection` 数据集训练 12 类课堂行为检测模型。
2. 后端对每个检测框独立判断正常或异常，不再用一个全局报警状态控制所有框颜色。
3. 异常框标红，正常框标绿，标签显示类别、置信度和状态。
4. 同一帧允许同时存在多个正常和异常目标。
5. 后端顶部状态栏显示异常数量和异常类别摘要。
6. 报警截图和 CSV 日志记录异常目标数量、类别和持续时间，便于课程报告引用。
7. 保持现有双机 TCP 图像传输方式不变，避免扩大网络部分风险。

## 3. 数据集类别

`Student Behaviour Detection.v6i.yolov8` 包含 12 类：

- `Using_phone`
- `bend`
- `book`
- `bow_head`
- `hand-raising`
- `phone`
- `raise_head`
- `reading`
- `sleep`
- `turn_head`
- `upright`
- `writing`

类别样本量不完全均衡，其中 `sleep`、`writing`、`hand-raising` 等类别相对较少。训练结果需要在报告中说明：模型能展示多状态检测能力，但少样本类别的准确率可能低于高频类别。

## 4. 状态映射规则

系统采用清晰、可解释的类别映射规则：

异常状态：

- `Using_phone`
- `phone`
- `sleep`
- `bend`
- `bow_head`
- `turn_head`

正常状态：

- `upright`
- `reading`
- `writing`
- `book`
- `hand-raising`
- `raise_head`

异常状态表示课堂注意力异常或疑似违纪行为，包括睡觉、玩手机、低头、弯腰和转头。正常状态表示课堂参与或正常学习行为，包括坐直、阅读、写字、看书、举手和抬头。

## 5. 后端设计

### 5.1 检测结果

`YoloDetector` 继续负责把 YOLO 输出转换为统一的 `Detection` 对象。该对象保留：

- `label`
- `confidence`
- `bbox`
- `width`
- `height`

模型类别直接来自训练后的 YOLO 模型，不在检测器中硬编码业务规则。

### 5.2 行为分析

新增或替换现有睡觉分析器为课堂行为分析器。分析器接收一帧中的所有检测框，并为每个框输出：

- 原始检测框
- 状态：`normal` 或 `abnormal`
- 原因：检测类别
- 该框是否达到报警持续时间

持续时间判断仍保留，避免单帧误检立即触发报警。由于当前系统还没有人员跟踪 ID，第一版采用“类别级持续时间”：只要某种异常类别连续出现超过阈值，就认为该类异常达到报警条件。界面画框仍按每个检测框的类别独立着色。

### 5.3 可视化

后端窗口画框规则：

- 异常类别：红色框。
- 正常类别：绿色框。
- 未知类别或低置信度类别：灰色或忽略。

标签格式建议：

```text
sleep abnormal 0.91
upright normal 0.86
```

顶部状态栏：

- 无异常：`normal`
- 有异常但未达到持续阈值：`suspicious: 2 abnormal`
- 达到报警阈值：`ALARM: 2 abnormal - sleep, phone`

### 5.4 报警记录

报警截图继续保存到 `output/alarms`。CSV 日志扩展字段：

- `frame_id`
- `timestamp_ms`
- `reason`
- `duration_seconds`
- `abnormal_count`
- `abnormal_labels`
- `image_path`

`reason` 可记录为 `multi_behaviour_abnormal`，`abnormal_labels` 记录本帧异常类别摘要。

## 6. 训练与测试设计

### 6.1 训练

使用 `Student Behaviour Detection.v6i.yolov8` 的 YOLOv8 数据格式训练新模型。由于 Roboflow 导出的 `data.yaml` 路径可能不适配本地 Ultralytics，训练前生成一个本地绝对路径配置文件放入 `tmp/`。

第一轮训练建议沿用轻量模型和较小输入尺寸，先验证流程：

- 模型：`yolov8n.pt`
- 数据集：`datasets/Student Behaviour Detection.v6i.yolov8`
- 输入尺寸：`320`
- 轮数：先用较少轮数跑通，再根据速度增加

训练输出建议路径：

```text
output/training/student_behaviour_yolov8n_e*/weights/best.pt
```

### 6.2 离线测试

使用新数据集的 `test/images` 做离线测试，输出预测图和 CSV：

```text
output/offline_test/student-behaviour-custom-*/
```

测试报告需要记录：

- 测试图片数量
- 预测框数量
- 各类别预测数量
- 验证集整体指标和重点异常类别指标

## 7. 前端影响

前端摄像头发送程序不需要改变。它只负责采集、压缩、发送图像，后端对图像进行多状态检测和异常分析。

## 8. README 与打包影响

README 需要新增多状态模型训练、离线测试和后端启动命令。后端打包脚本默认模型需要从原来的睡觉三分类模型切换到新的多状态模型。

打包结果仍为：

```text
dist/backend-student-sleep-server.zip
```

包内模型文件名建议改为：

```text
models/student_behaviour_yolov8n_best.pt
```

## 9. 验收标准

1. 单元测试通过。
2. 后端能加载新训练的 12 类模型。
3. 同一画面中正常框和异常框可以同时出现。
4. 异常框只影响对应检测目标，不让整帧所有框一起变红。
5. 报警截图和 CSV 能记录多状态异常信息。
6. README 中能按步骤完成训练、离线测试、双机运行和后端打包。

## 10. 非目标

本阶段不做人脸识别、学生身份识别、跨帧人员 ID 跟踪和网页端管理后台。若课程后续需要“某个学生连续异常”的严格语义，可以在本版本基础上增加目标跟踪算法。
