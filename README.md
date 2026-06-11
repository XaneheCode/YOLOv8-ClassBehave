# 基于双机的实时图像远程监测与异常分析系统

## 运行环境

- Python 3.12。不要使用 Python 3.14 创建 `.venv`，否则 NumPy/OpenCV 的二进制包可能不兼容。
- 两台笔记本位于同一无线局域网
- 前端笔记本连接摄像头
- 后端笔记本安装依赖并运行检测端

## 安装依赖

推荐直接运行环境脚本。脚本会优先寻找 Python 3.12，并把 pip 源设置为清华源：

```powershell
.\scripts\setup_env.ps1
```

如果脚本找不到 Python 3.12，可以手动指定解释器路径：

```powershell
.\scripts\setup_env.ps1 -Python C:\Path\To\Python312\python.exe
```

## 后端 GUI 启动

验收和成果展示推荐使用 GUI：

```powershell
.\START_BACKEND_GUI.ps1
```

后端默认监听 `0.0.0.0:5001`，模型路径为：

```text
models/classroom_behaviour_6cls.pt
```

该模型来自 `D:\Documents\YOLOv8\yolov8_onnx\models\best_last.pt`，类别为 6 类课堂行为。

## 前端 GUI 启动

在前端笔记本运行：

```powershell
.\START_FRONTEND_GUI.ps1
```

在前端窗口填写后端电脑的无线网卡 IPv4 地址，例如 `192.168.1.20`，点击“开始发送”。

## 六类课堂行为

正常状态：

- `Hand-raise`：举手
- `Reading`：看书
- `Writing`：写字

异常状态：

- `Useing-Phone`：使用手机
- `Head-down`：低头
- `Sleeping`：睡觉

报警触发时，系统只给异常目标标红，正常目标保持绿色。报警状态用于顶部提示、截图和 CSV 日志，不会把整帧所有目标统一变红。

## 命令行后端启动

如果不使用 GUI，也可以直接启动命令行后端：

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001
```

如果只想验证基础环境，也可以临时使用通用模型：

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model yolov8n.pt
```

记录后端笔记本的无线网卡 IPv4 地址，例如 `192.168.1.20`。

## 命令行前端启动

如果不使用 GUI，也可以在前端笔记本运行命令行发送端：

```powershell
.\.venv\Scripts\python.exe -m src.frontend.camera_client --host 192.168.1.20 --port 5001 --camera 0
```

## 验收演示

1. 后端显示前端摄像头画面。
2. 后端画面显示每个检测目标的类别、状态和置信度。
3. 正常状态框显示绿色，异常状态框显示红色；红色只作用于对应异常目标。
4. 异常状态持续 3 秒后，后端显示报警信息。
5. `output/alarms/alarms.csv` 保存报警记录。
6. `output/alarms/*.jpg` 保存报警截图。

## 单机烟测

在没有第二台笔记本时，可以先在同一台电脑上完成闭环测试：

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 127.0.0.1 --port 5001
.\.venv\Scripts\python.exe -m src.frontend.camera_client --host 127.0.0.1 --port 5001 --camera 0
```

正常坐姿应保持 normal 状态。睡觉、使用手机或低头等异常状态持续超过 3 秒后，应出现报警提示，并在 `output/alarms` 下生成报警记录。

## YOLOv8 六类模型离线测试

使用 `D:\Documents\YOLOv8\yolov8_onnx\测试` 样例图片完成离线测试：

```powershell
.\.venv\Scripts\python.exe scripts\offline_test_images.py --images "D:\Documents\YOLOv8\yolov8_onnx\测试" --model models\classroom_behaviour_6cls.pt --output-dir output\offline_test\yolov8-6cls --limit 0 --conf 0.25
```

本轮测试结果：

- 图片数：3
- 预测记录：114
- 正常记录：66
- 异常记录：48
- 输出 CSV：`output/offline_test/yolov8-6cls/predictions.csv`
- 标注图片：`output/offline_test/yolov8-6cls/*.jpg`

测试记录见：

- `docs/course-evidence/yolov8-6cls-offline-test.md`

## 旧版多状态模型训练与离线测试

当前多状态数据集目录：

```text
datasets/Student Behaviour Detection.v6i.yolov8
```

生成本地 YOLO 配置：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_yolo_data_yaml.py --dataset "datasets\Student Behaviour Detection.v6i.yolov8" --output tmp\student-behaviour-detection-abs.yaml
```

训练 12 类课堂行为模型：

```powershell
.\.venv\Scripts\yolo.exe detect train data=tmp\student-behaviour-detection-abs.yaml model=yolov8n.pt epochs=20 imgsz=640 batch=8 device=cpu workers=0 project=output\training name=student_behaviour_yolov8n_e20 exist_ok=True
```

如果训练结果被 Ultralytics 保存到 `runs/detect/output/training/student_behaviour_yolov8n_e20`，可以直接把该目录复制到 `output/training/student_behaviour_yolov8n_e20`。

本轮 e20/imgsz640 验证集指标：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.739 | 0.685 | 0.709 | 0.466 |

使用 `test/images` 离线测试：

```powershell
.\.venv\Scripts\python.exe scripts\offline_test_images.py --dataset "datasets\Student Behaviour Detection.v6i.yolov8" --model output\training\student_behaviour_yolov8n_e20\weights\best.pt --output-dir output\offline_test\student-behaviour-custom-e20 --limit 0 --conf 0.25
```

本轮训练和测试记录见：

- `docs/course-evidence/student-behaviour-custom-e20-test.md`

## 后端打包

重新生成后端交付包：

```powershell
.\scripts\package_backend.ps1
```

打包产物：

- `dist/backend-student-sleep-server.zip`
- `dist/backend-student-sleep-server/`

后端包内包含训练后的模型、后端 GUI、命令行后端、公共模块、依赖文件、`START_BACKEND_GUI.ps1` 和 `START_BACKEND.ps1`。

默认打包模型：

```text
models/classroom_behaviour_6cls.pt
```

## 可选：下载旧三分类数据集

Roboflow 下载数据集需要 API key。登录 Roboflow 后，在账户设置或数据集下载代码中复制 API key，然后只在当前终端设置环境变量：

```powershell
$env:ROBOFLOW_API_KEY="你的 Roboflow API Key"
.\.venv\Scripts\python.exe .\scripts\download_roboflow_dataset.py
```

默认下载目标是 `datasets/student-classroom-activity-v2`，导出格式是 YOLOv8。脚本使用的数据集信息：

- workspace: `studentactivity`
- project: `new-student-classroom-activity-2`
- version: `2`
- format: `yolov8`

下载完成后，目录中应包含：

```txt
datasets/student-classroom-activity-v2/
  data.yaml
  test/images/
  train/images/
  valid/images/
```

## 可选：旧三分类数据集离线测试

下载完成后，可以用 `test/images` 批量做离线推理：

```powershell
.\.venv\Scripts\python.exe .\scripts\offline_test_images.py --dataset datasets/student-classroom-activity-v2 --model yolov8n.pt --limit 50
```

输出结果：

- `output/offline_test/student-classroom-activity-v2/predictions.csv`
- `output/offline_test/student-classroom-activity-v2/*.jpg` 标注结果图
