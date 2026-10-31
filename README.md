# 冒险岛小游戏助手（跨品牌 GPU 版）

不再捆绑 NVIDIA CUDA 或 PyTorch。使用 OpenCL 自动枚举本机 GPU，支持安装了 OpenCL 驱动的 NVIDIA GeForce/RTX、AMD Radeon、Intel UHD/Iris Xe/Arc。程序启动时自动选择可用 GPU；若驱动缺失、设备不支持或 OpenCL 初始化失败，自动回退 OpenCV CPU，并显示当前实际后端。

GPU 执行 RGB 亮度、低饱和度判断和白色像素掩膜；OpenCV CPU 执行背景模型、连通区域分析、卡尔曼跟踪和鼠标控制。单帧屏幕图像的 GPU 数据传输也有成本，不保证所有显卡上都比 CPU 快。

## Windows 使用

从 GitHub Actions 下载 **WhiteTargetAssistant-Windows-OpenCL**，解压整个目录，运行 `WhiteTargetAssistant.exe`。不需要安装 CUDA 或 PyTorch。需要 GPU 模式时，安装显卡厂商提供的最新驱动及 OpenCL ICD。

源码运行：

    py -3.11 -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    python app.py

透明目标跟踪和成功/确认提示识别仍需在实际游戏分辨率下验证。
