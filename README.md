# 基于双机的实时图像远程监测与异常分析系统

## 运行环境

- Python 3.10+
- 两台笔记本位于同一无线局域网
- 前端笔记本连接摄像头
- 后端笔记本安装依赖并运行检测端

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
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

