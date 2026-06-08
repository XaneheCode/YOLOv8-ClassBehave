# 课堂睡觉检测相关数据集候选

检索日期：2026-06-08

## 推荐优先级

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
- 规模：约 2.5k 张图片
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

第一轮测试建议使用 Roboflow 的 `Student Classroom Activity v-2`，因为它最贴合当前项目的 `sleep`、`study`、`phone` 场景。

第二轮如果需要增加测试样本，使用 `Student Behaviour Detection` 或 `Class Monitoring`。这两个数据集都包含 `sleep`、`bow_head`、`bend` 等类别，适合检验低头写字和趴桌睡觉之间的误报。

Kaggle 的 `Student Concentration Image Dataset` 和普通 drowsiness 数据集可以作为报告中的参考数据集，不建议作为当前 YOLO 检测系统的首选测试集。

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

