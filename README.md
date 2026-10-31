# 白色目标跟踪助手

一个用于本机屏幕画面中识别近白色目标、短时遮挡预测和鼠标跟随的 Windows 桌面工具。

## 功能

- 从主显示器实时读取画面。
- 识别近白色图形，可调亮度阈值、颜色容差和最小面积。
- 使用卡尔曼滤波进行短时遮挡预测；目标重新出现时自动重新锁定。
- 可开关鼠标跟随，并调节跟随平滑度。
- 可选：目标消失后等待指定时间，自动发送确认键，然后等待下一轮目标出现。
- 有独立停止按钮；PyAutoGUI 鼠标安全角保护默认开启。
- GitHub Actions 自动运行测试并打包 Windows EXE。

## 运行

需要 Windows 10/11、Python 3.10 或 3.11。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## GPU

程序会在检测到可用 CUDA 版 PyTorch 时使用 GPU 生成颜色掩膜，否则自动使用 OpenCV CPU 模式。普通 `requirements.txt` 不安装大型 CUDA 运行库。

如需 CUDA，请先按本机显卡驱动和 CUDA 兼容性，从 PyTorch 官方安装页面选择对应命令安装 CUDA 版 PyTorch，再启动程序。GPU 状态会显示在窗口中。

## 使用步骤

1. 打开目标程序，把目标放在主显示器可见区域。
2. 启动本工具，先保持“鼠标跟随”关闭，调节阈值直到状态栏能稳定识别目标。
3. 确认识别正确后再开启鼠标跟随。
4. “自动确认”默认关闭。只有确认目标消失确实代表一轮结束时，才开启它；确认键和延迟可自行设置。
5. 随时点击“停止”终止捕获和鼠标控制。

## 注意

- 这是基于颜色和形状的通用视觉原型，不包含特定游戏模型、专用权重或对加密模型的破解。
- 透明/遮挡期间只能做短时间运动预测；目标长时间完全不可见时，任何纯视觉程序都无法保证准确知道它的位置。
- 不保证适用于所有游戏、反作弊环境或多显示器布局。请先在非关键场景测试。
- 自动确认可能向当前前台窗口发送按键，请谨慎启用。
- 本工具不提权、不注入进程、不读取游戏内存。

## 本地测试

```powershell
pip install -r requirements.txt
pip install pytest
pytest -q
```

## Windows EXE

推送代码后，GitHub Actions 会运行测试并生成 `WhiteTargetAssistant.exe` 构建产物。到仓库的 **Actions** 页面打开最新的成功运行，在 **Artifacts** 下载。
