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

## 后端启动

在后端笔记本运行：

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model yolov8n.pt
```

记录后端笔记本的无线网卡 IPv4 地址，例如 `192.168.1.20`。

## 前端启动

在前端笔记本运行：

```powershell
.\.venv\Scripts\python.exe -m src.frontend.camera_client --host 192.168.1.20 --port 5001 --camera 0
```

## 验收演示

1. 后端显示前端摄像头画面。
2. 后端画面显示检测框和检测标签。
3. 出现疑似趴桌睡觉动作并持续 3 秒后，后端显示报警信息。
4. `output/alarms/alarms.csv` 保存报警记录。
5. `output/alarms/*.jpg` 保存报警截图。

## 单机烟测

在没有第二台笔记本时，可以先在同一台电脑上完成闭环测试：

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 127.0.0.1 --port 5001 --model yolov8n.pt
.\.venv\Scripts\python.exe -m src.frontend.camera_client --host 127.0.0.1 --port 5001 --camera 0
```

正常坐姿应保持 normal 状态。趴桌姿态持续超过 3 秒后，应出现报警提示，并在 `output/alarms` 下生成报警记录。
