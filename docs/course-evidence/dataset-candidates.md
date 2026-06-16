# 课堂睡觉检测相关数据集候选

检索日期：2026-06-11

## 当前问题

单机烟测中，前端和后端传输正常，但远景课堂画面中检测框可能很少或没有。原因通常是：

- 现场摄像头视角比样例图更远，学生目标更小。
- 课堂遮挡较多，桌椅、显示器会挡住身体和手部动作。
- 现有 6 类模型对现场角度、距离和光照覆盖不足。
- 单一数据集训练容易对特定教室、拍摄角度和标注风格过拟合。

因此后续训练建议不只增加同类样本数量，还要补充远景、多目标、遮挡、不同教室视角的数据。

## 推荐优先级

### 0. SCB-dataset / SCB-Dataset5

- 链接：https://github.com/Whiffe/SCB-dataset
- 平台：GitHub / Hugging Face / Baidu Netdisk
- 类型：课堂行为目标检测数据集
- 规模：SCB-Dataset5 论文记录为 7,428 张图片、106,830 个标注，覆盖 20 类课堂对象/行为。
- 类别匹配：包含 `hand-raising`、`read`、`write`、`bow head`、`leaning on the desk`、`using the phone` 等，和当前系统的举手、看书、写字、低头、玩手机高度相关。
- 适合用途：最适合补强“远景课堂、多学生、小目标、遮挡”的泛化能力。
- 注意事项：项目 README 明确限制商业用途，课程学习/研究可用；下载源包含百度网盘和 Hugging Face，具体可用性需要现场确认。
- 推荐处理：把 `hand-raising` 映射为 `Hand-raise`，`read` 映射为 `Reading`，`write` 映射为 `Writing`，`bow head` 映射为 `Head-down`，`leaning on the desk` 可作为 `Sleeping` 或单独保留为 `Head-down/Sleeping` 相邻类。

### 0.5. SCB-changed / Roboflow 大规模版本

- 链接：https://universe.roboflow.com/search?q=like%3Astudentclassroomanalytics%2Fstudent-behaviour-in-classroom-2+object+detection+trained+model
- 平台：Roboflow Universe
- 类型：目标检测数据集
- 规模：约 14.2k 张图片
- 类别：`reading`、`Using_phone`、`bow_head`、`hand-raising`、`raise_head`、`sleep`、`turn_head`、`upright`、`writing`
- 适合用途：比当前小数据集更适合重新训练一版课堂行为模型，尤其能补 `sleep`、`bow_head`、`Using_phone` 和 `writing`。
- 注意事项：Roboflow 搜索页给出规模和类别，下载时需要确认具体项目版本、许可证和导出格式。

### 1. Student Classroom Activity v-2

- 链接：https://universe.roboflow.com/studentactivity/student-classroom-activity-v-2
- 平台：Roboflow Universe
- 类型：目标检测数据集/模型
- 规模：约 1.1k 张图片
- 类别：`phone`、`sleep`、`study`
- 许可证：CC BY 4.0
- 适合用途：最适合本课程设计的测试集，因为它直接包含课堂场景中的 `sleep` 类，和系统目标一致。
- 注意事项：下载通常需要 Roboflow 账号；导出时建议选择 YOLOv8/YOLOv5 格式。

### 2. Student Behaviour Detection

- 链接：https://universe.roboflow.com/mywork-lkwz4/student-behaviour-detection-neazg
- 平台：Roboflow Universe
- 类型：目标检测数据集/模型
- 规模：2,471 张图片
- 类别：包含 `sleep`、`bow_head`、`bend`、`upright`、`writing`、`Using_phone` 等 12 类
- 许可证：CC BY 4.0
- 适合用途：适合测试“趴桌睡觉”和“低头写字”的区分能力，也适合后续训练自定义 YOLO。
- 注意事项：类别较多，若只做睡觉检测，可以只保留 `sleep`、`bow_head`、`bend`、`upright`、`writing` 相关样本。

### 3. Class Monitoring

- 链接：https://universe.roboflow.com/noumancs192043/class-monitoring
- 平台：Roboflow Universe
- 类型：目标检测数据集/模型
- 规模：约 2.7k 张图片
- 类别：包含 `sleep`、`reading`、`bend`、`bow_head`、`upright`、`Using_phone`、`writing` 等 9 类
- 许可证：CC BY 4.0
- 适合用途：适合做课堂行为多类别检测测试，和当前系统中的“趴桌/低头连续判定”逻辑比较匹配。

### 3.5. Classroom Student Dataset

- 链接：https://universe.roboflow.com/search?q=class%3A%22using+phone%22
- 平台：Roboflow Universe
- 类型：目标检测数据集
- 规模：约 1.78k 张图片
- 类别：页面检索结果显示包含 `Hand Rising`、`Reading`、`Sleeping`、`Using Phone`、`Writing` 等。
- 适合用途：和当前 6 类最接近，可用于补充 `Hand-raise`、`Reading`、`Writing`、`Useing-Phone`、`Sleeping`。
- 注意事项：需要进入具体项目确认是否能导出 YOLOv8、标注质量和许可证。

### 3.6. Student behavior detection for YOLOv8

- 链接：https://www.kaggle.com/datasets/cubeai/student-behavior-detection-for-yolov8
- 平台：Kaggle
- 类型：YOLOv8 目标检测数据集
- 规模：页面显示约 1 GB、93 个文件，包含 `data.yaml`
- 适合用途：直接面向 YOLOv8，适合下载后快速检查 `data.yaml` 类别并尝试训练。
- 注意事项：Kaggle 页面摘要没有充分展示类别和许可证；下载后必须先检查 `data.yaml`、标注框数量和样例图质量。

### 3.7. Dataset of student classroom behavior

- 链接：https://www.kaggle.com/datasets/kaiyueyyds/dataset-of-student-classroom-behavior
- 平台：Kaggle
- 类型：课堂行为数据集
- 类别匹配：页面摘要提到包含“低头写字、抬头听课、举手”等多种课堂姿态。
- 适合用途：如果包含检测框标注，可用于补举手、低头、写字；如果只是分类图，则适合作为报告调研或二次标注来源。
- 注意事项：下载后需要确认是否是 YOLO 检测格式。

### 4. student_sleep

- 链接：https://universe.roboflow.com/rong-fengliang-aong2/student_sleep-f9vlf
- 平台：Roboflow Universe
- 类型：目标检测数据集/模型
- 规模：60 张图片
- 类别：包含 `sleep`、`not_sleep`、`student_sleep`
- 许可证：CC BY 4.0
- 适合用途：适合快速烟测，不适合作为主要训练集。
- 注意事项：样本量较小，类别命名需要清洗。

### 5. CampusGuard

- 链接：https://www.juheapi.com/datasets/campusguard
- 平台：JuheAPI
- 类型：校园行为目标检测数据集
- 规模：3,345 张图片
- 类别：`Mobile-Phone`、`No-Helmet`、`Sleeping`、`Triples`、`Violence`
- 适合用途：适合写报告中的相关数据集调研，也可用于扩展校园异常行为检测。
- 注意事项：页面显示需要登录后查看下载链接，获取成本高于 Roboflow。

### 6. Student Concentration Image Dataset

- 链接：https://www.kaggle.com/datasets/programmer3/student-concentration-image-dataset/data
- 平台：Kaggle
- 类型：图像分类数据集
- 规模：2,120 张 JPG 图片
- 类别：包含 `Drowsy`、`Engaged`、`Focused`、`Looking Away` 等 8 类
- 许可证：CC0 Public Domain
- 适合用途：适合补充“疲劳/困倦”分类测试，不直接适合 YOLO 检测。
- 注意事项：如果用于当前系统，需要先转换为检测框标注，或只作为报告中对比数据集。

### 7. Drowsiness dataset

- 链接：https://www.kaggle.com/datasets/hoangtung719/drowsiness-dataset
- 平台：Kaggle
- 类型：图像分类数据集
- 类别：眼睛睁开/闭合、打哈欠/未打哈欠
- 适合用途：适合补充疲劳识别背景，不直接对应课堂趴桌睡觉。
- 注意事项：更偏驾驶员疲劳和面部局部特征，不是课堂场景。

## 建议使用方案

第一轮重新训练建议使用 `Student Behaviour Detection` 或 `SCB-changed`，因为它们覆盖当前系统最关键的 `sleep`、`bow_head`、`Using_phone`、`writing`、`reading`、`hand-raising` 类别。

第二轮如果重点解决“远景小目标识别不出人”，优先加入 `SCB-dataset / SCB-Dataset5`。它的数据量和标注数量更适合补课堂远景、多学生、遮挡场景。

第三轮可以加入 `Classroom Student Dataset`、`Class Monitoring` 和 Kaggle 的 `Student behavior detection for YOLOv8` 做交叉验证。Kaggle 的 `Student Concentration Image Dataset` 和普通 drowsiness 数据集仍只建议作为报告参考或二次标注素材，不建议作为当前 YOLO 检测系统的首选训练集。

## 类别统一建议

当前系统运行类别：

```text
Hand-raise, Reading, Writing, Useing-Phone, Head-down, Sleeping
```

建议把新数据集统一成 6 类后再训练：

| 原始类别示例 | 统一类别 |
| --- | --- |
| `hand-raising`, `hand raising`, `Hand Rising`, `Raise_hand` | `Hand-raise` |
| `reading`, `read`, `book`, `study` | `Reading` |
| `writing`, `write`, `noting` | `Writing` |
| `Using_phone`, `using phone`, `Using Phone`, `phone`, `using-phone` | `Useing-Phone` |
| `bow_head`, `bowing the head`, `looking down`, `head_down` | `Head-down` |
| `sleep`, `sleeping`, `leaning over the table`, `leaning on the desk`, `lying_desk` | `Sleeping` |

若某些数据集没有 `Sleeping`，不要强行映射所有低头为睡觉，否则会增加误报。可以保留 `Head-down` 和 `Sleeping` 分开训练。

## 训练策略建议

1. 先用现有 `models/classroom_behaviour_6cls.pt` 做基线，记录现场失败场景。
2. 下载 1 个主数据集训练第一版，不要一开始混太多来源。
3. 用统一脚本把类别名清洗到 6 类，删除无法映射的类别。
4. 训练时使用较大输入尺寸，例如 `imgsz=960` 或 `imgsz=1280`，优先改善远景小目标。
5. 使用现场摄像头采集 100 到 300 张失败画面，手动补标后加入训练集。
6. 验证集必须包含现场教室远景，不然 mAP 好看但 GUI 仍可能没有框。

示例训练命令：

```powershell
.\.venv\Scripts\yolo.exe detect train data=tmp\merged-classroom-6cls.yaml model=models\classroom_behaviour_6cls.pt epochs=50 imgsz=960 batch=4 device=cpu workers=0 project=output\training name=classroom_6cls_finetune_v1 exist_ok=True
```

CPU 训练会很慢。如果有 NVIDIA 显卡，把 `device=cpu` 改成 `device=0`。

## 与当前代码的衔接

如果下载 Roboflow YOLO 格式数据集，通常会得到类似结构：

```txt
dataset/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
```

用于测试当前后端模型时，可以先挑选 `test/images` 下的若干图片，离线运行 YOLO 推理并观察是否输出 `sleep` 类。若后续训练自定义模型，应确认 `data.yaml` 中的类别名包含 `sleep`，并把训练得到的 `best.pt` 作为后端启动参数：

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model runs/detect/train/weights/best.pt
```
