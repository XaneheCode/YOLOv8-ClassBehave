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

本地 Qt 后端 GUI 已采用与浏览器控制台一致的 dashboard 风格，但底层仍保留课程设计的自定义 `NSGD` TCP 帧包协议和 Qt 工作线程链路。正式双机联调时，后端窗口负责监听 TCP 连接、解包前端 JPEG 帧、执行 YOLO/大模型分析并保存报警记录。

后端默认监听 `0.0.0.0:5001`。默认识别模式为 `人体YOLO+大模型`，模型路径为：

```text
yolov8s.pt
```

该模式使用 COCO 预训练 YOLOv8 只检测 `person` 人体框；后端会把人体框外扩裁剪、编号拼成一张大图，再交给大模型判断每个编号人物的六类行为，最后把分类结果回填到原始人体框上显示。GUI 中仍保留 `六类YOLO` 模式，可切回训练好的六分类权重做对比或回退。当前默认使用 `yolov8s.pt`，人体检测精度通常高于 `yolov8n.pt`。

后端默认 YOLO 推理尺寸为 `imgsz=960`，用于兼顾大教室小目标和实时速度。如果现场远处学生漏检较多，可以在 `src/backend/detector.py` 中调大 `DEFAULT_INFERENCE_SIZE`；如果实时视频卡顿，可以调小。

## 浏览器 Web 控制台

如果希望在单机浏览器中演示和操作系统，可以启动 SaaS 风格 Web 控制台：

```powershell
.\START_WEB_DASHBOARD.ps1
```

启动后打开：

```text
http://127.0.0.1:8765
```

Web 控制台分为四个模块：

- `发送端`：浏览器摄像头选择、本地图片、本地视频输入；按设定间隔抽帧，并发送到后端 YOLO 接口。
- `实时分析`：只展示 YOLO 实时分析结果，包括 YOLO 画面、检测框、FPS、延迟、分辨率、目标数、行为计数和报警状态。
- `大模型`：只展示大模型分析结果，包括编号人体拼图、分类列表、运行状态和大模型回填画面。该模块使用实时分析模块返回的 YOLO 人体框，不会覆盖 YOLO 控制台画面。
- `日志设置`：展示上传、YOLO 分析、大模型分析、跳过和错误状态。

浏览器端调用两个独立接口：

- `/api/yolo-frame`：接收浏览器发送端上传的图像帧，只执行 YOLO 分析。
- `/api/vlm-frame`：接收同一帧和 YOLO 人体框，只执行大模型行为分类。

Web 控制台使用本地 HTTP API 调用现有 YOLO 和大模型分析逻辑，不需要 Node.js。它适合单机演示、图片/视频快速测试和前端设计展示；正式双机网络设计展示仍推荐使用本地 Qt 前后端，因为 Qt 版本完整保留 `NSGD` TCP 包头、帧编号、时间戳、长度字段、粘包处理和收发线程。若启用大模型分类，请继续使用 `.env` 或当前 PowerShell 环境变量配置 `DASHSCOPE_API_KEY` / `OPENAI_API_KEY` 等参数。视频和摄像头连续分析时，前端会在上一轮大模型请求未返回前跳过下一轮大模型上传，避免请求堆积；YOLO 实时分析仍可按发送端间隔持续更新。Web 版触发报警后会保存截图和 CSV 到 `output/web-alarms/yolo` 或 `output/web-alarms/vision`。

## 模型清单

| 文件 | 日期 | 用途 |
| --- | --- | --- |
| `yolov8s.pt` | Ultralytics COCO 预训练 | 当前默认人体检测模型，只保留 `person` 类 |
| `yolov8n.pt` | Ultralytics COCO 预训练 | 轻量人体检测模型，速度更快但精度通常低于 `yolov8s.pt` |
| `models/merged_classroom_6cls_v2_img960_e50_2026-06-13_best.pt` | 2026-06-13 | 六类YOLO回退模型，merged-classroom-6cls-v2 训练 50 轮最佳权重 |
| `models/classroom_behaviour_6cls_2024-10-02_baseline.pt` | 2024-10-02 | 原始 6 类课堂行为基线模型，保留用于对比和回退 |
| `models/student_behaviour_v6_6cls_img960_e50_2026-06-12_best.pt` | 2026-06-12 | Student Behaviour Detection v6 单数据集训练 50 轮模型 |
| `models/scb_yolov7_HRW_4.2k_2026-06-12.pt` | 2026-06-12 | SCB YOLOv7 HRW 权重，仅用于兼容性测试，不作为当前 YOLOv8 默认模型 |

为兼容旧命令，`models/classroom_behaviour_6cls.pt`、`models/student_behaviour_v6_6cls_img960_e50_best.pt`、`models/yolov7_HRW_4.2k.pt` 和 `models/merged_classroom_6cls_v2_img960_e50_best.pt` 仍保留为历史文件名。

后端 GUI 也支持直接载入本地测试素材展示识别效果：

- 点击 `选择图片测试`，选择一张课堂图片，后端会直接显示 YOLO 识别框、行为计数和异常状态。
- 点击 `选择视频测试`，选择一段课堂视频，后端会逐帧播放并实时识别。
- 点击 `停止测试`，停止当前本地图片/视频展示。

本地素材测试不需要启动前端发送端，适合课程验收时快速展示模型效果；双机协同时仍使用 `开始监听` 接收前端摄像头。

### 可选：大模型视觉辅助分析

后端 GUI 已同步原 YOLOv8 项目的视觉大模型分析功能。默认 `人体YOLO+大模型` 模式下，系统以 YOLO 人体框为基础，按间隔抽取当前画面中的人体框，拼成编号大图给大模型分类，并在独立的“大模型分析结果”窗口显示分类后的行为框、分析结果和运行状态。

使用前在当前 PowerShell 终端设置 DashScope API Key：

```powershell
$env:DASHSCOPE_API_KEY="你的 DashScope API Key"
.\START_BACKEND_GUI.ps1
```

可选环境变量：

```powershell
$env:QWEN_VL_MODEL="qwen3.6-flash"
$env:QWEN_UPLOAD_INTERVAL_SECONDS="10"
$env:QWEN_MAX_YOLO_TARGETS="30"
$env:QWEN_COORDINATE_GRID="1"
$env:DASHSCOPE_BASE_HTTP_API_URL="https://dashscope.aliyuncs.com/api/v1"
```

如果未配置大模型 API Key，后端 YOLO 人体检测仍可正常显示；大模型窗口会提示未配置。若当前 YOLO 检测目标数超过 `QWEN_MAX_YOLO_TARGETS`，系统会跳过本次大模型分类，避免密集课堂场景下上传过大、返回过慢。

也可以切换到 OpenAI 兼容视觉接口。后端会把本地处理后的标准 PNG 编码成 `data:image/png;base64,...`，并调用 `/v1/chat/completions`：

```powershell
$env:VISION_PROVIDER="openai"
$env:OPENAI_BASE_URL="https://ai.laodog.top/"
$env:OPENAI_API_KEY="你的 OpenAI 兼容接口 API Key"
$env:OPENAI_VISION_MODEL="gpt-5.5"
.\START_BACKEND_GUI.ps1
```

GPT 视觉分析可能较慢。OpenAI 兼容路径默认会把上传图缩到 640 像素宽，并把接口读取超时设为 120 秒：

```powershell
$env:OPENAI_IMAGE_MAX_WIDTH="640"
$env:OPENAI_TIMEOUT_SECONDS="120"
$env:OPENAI_IMAGE_FORMAT="png"
```

如果只是模型返回较慢，可以把 `OPENAI_TIMEOUT_SECONDS` 调到 `180`；如果仍然超时，可以临时把 `OPENAI_IMAGE_MAX_WIDTH` 调到 `480`。默认首传仍使用 PNG；如果 PNG 上传遇到网络断连，程序会自动用体积更小的 JPEG 再试一次。

视频或直播连续上传时，部分 OpenAI 兼容中转站可能会偶发 HTTPS 断连。程序会对每次大模型上传使用独立连接、失败后自动重试 1 次，并在网络错误后暂停 30 秒再发送下一帧，避免连续重传。如果仍然频繁报错，建议先降低上传频率或图片宽度：

```powershell
$env:QWEN_UPLOAD_INTERVAL_SECONDS="20"
$env:OPENAI_IMAGE_MAX_WIDTH="480"
$env:OPENAI_TIMEOUT_SECONDS="180"
$env:OPENAI_IMAGE_FORMAT="jpeg"
```

OpenAI 兼容路径仍复用同一个辅助分析窗口，窗口标题为“大模型分析结果”。

大模型分析结果会被限制到六类行为：`Hand-raise`、`Reading`、`Writing`、`Useing-Phone`、`Head-down`、`Sleeping`。底层仍保留 `Reading` 和 `Writing` 两个标签，但系统展示时会把二者合并显示为“学习”。结果窗口中，图上只显示彩色框和编号，详细类别、坐标、状态和置信度在下方 `分析结果` 区域查看；上传、失败、跳过和完成信息显示在 `大模型状态` 区域。展示颜色分别为：举手蓝色、学习绿色、使用手机红色、低头橙色、睡觉紫色。

其中 `Useing-Phone` 使用优先判定：只要大模型看到手机、手持小矩形屏幕、手里拿着手机、双手低头看小屏幕，就优先标为使用手机；只有台式电脑、笔记本电脑、键盘、鼠标或显示器时，不按手机处理。

## 前端 GUI 启动

在前端笔记本运行：

```powershell
.\START_FRONTEND_GUI.ps1
```

本地 Qt 前端 GUI 同样采用 dashboard 风格，但底层仍使用 OpenCV 摄像头采集、JPEG 编码和 `NSGD` TCP 帧包发送。在前端窗口填写后端电脑的无线网卡 IPv4 地址，例如 `192.168.1.20`，选择摄像头编号、宽度、FPS 和 JPEG 质量后点击“开始发送”。

## 六类课堂行为

正常状态：

- `Hand-raise`：举手
- `Reading`：学习
- `Writing`：学习

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
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model yolov8s.pt
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
.\.venv\Scripts\python.exe scripts\offline_test_images.py --images "D:\Documents\YOLOv8\yolov8_onnx\测试" --model models\merged_classroom_6cls_v2_img960_e50_2026-06-13_best.pt --output-dir output\offline_test\merged-v2-latest --limit 0 --conf 0.25
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
models/merged_classroom_6cls_v2_img960_e50_2026-06-13_best.pt
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
