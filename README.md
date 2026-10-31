# 冒险岛小游戏助手（CUDA 版）

Windows 桌面程序：白色目标锁定、目标变淡时的前景/运动预测、鼠标跟随、成功确认和下一轮等待。

## CUDA 加速范围

- **CUDA GPU（PyTorch）**：RGB 帧传输、亮度/饱和度计算、白色目标像素掩膜。
- **CPU（OpenCV）**：MOG2 时间背景模型、形态学处理、连通区域筛选、卡尔曼运动预测。
- **CPU（PyAutoGUI）**：鼠标跟随、确认点击和紧急停止。

程序启动时会检测 CUDA 运行时和 NVIDIA 设备。CUDA 不可用或运行中报错时，会自动切回 CPU，并在窗口底部显示当前实际使用的后端。GPU 版 PyTorch wheel 会把 CUDA 用户态运行库带入打包产物；运行电脑仍需安装兼容的 NVIDIA 显卡驱动。没有 NVIDIA GPU 的电脑仍可运行，但会使用 CPU。

## 下载 Windows 程序

打开 GitHub Actions，选择最新成功的 **Windows CUDA build**，下载 Artifacts 中的 **WhiteTargetAssistant-CUDA-windows**，解压后运行文件夹内的 **WhiteTargetAssistant.exe**。请保留同一文件夹中的依赖文件，不要只单独复制 EXE。

## 从源码运行

Windows PowerShell 命令：

    py -3.11 -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements-cuda.txt
    python app.py

若只需 CPU 版，将 requirements-cuda.txt 换成 requirements.txt。

## 构建和验证

GitHub Actions 在 Windows runner 上安装 CUDA-enabled PyTorch、运行测试，并使用 PyInstaller 收集 PyTorch 与 NVIDIA CUDA DLL。CI runner 本身不提供 NVIDIA GPU，因此 CI 能验证 CUDA wheel 导入和 EXE 打包，但不能代替在实际 NVIDIA 显卡电脑上的 GPU 执行验证。

## 识别限制

完全不可见且运动方向突变的目标无法仅靠屏幕画面保证准确预测。成功提示和橙色确认按钮仍基于提供画面的颜色特征，需要在实际游戏分辨率下验证。

这是独立实现，不解密第三方软件，也不绕过保护机制。
