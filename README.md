# 冒险岛小游戏助手（参考软件重构版）

Windows 构建从 `.build/reference_assistant.tar.gz.b64` 与 `.build/reference_assistant.part2` 还原参考软件对应的完整实现：`inminigame_assistant.py`、`gpu_backend.py`、测试和 Windows 启动/打包脚本。

## 功能
- 枚举并选择冒险岛游戏窗口；捕获客户区指定 ROI，正确映射屏幕坐标。
- 白色不规则图形多阈值分割，以最大内切圆点作为鼠标安全点。
- 目标身份关联、速度/加速度估计；目标淡出后进行有界轨迹预测。
- 黄色/红色结束标志检测；连续多帧确认后执行结束点击，标志消失后结束本轮并等待下一轮。
- OpenCL 自动使用可用的 NVIDIA / AMD / Intel GPU 进行白色掩膜计算；无可用设备时自动回退 CPU。
- 参考软件风格界面：小游戏、历史统计、运行日志；支持窗口捕获、预览、暂停、停止和本地记录。

## 构建与下载
GitHub Actions 的 **Windows reference-based assistant build** 会执行源码还原、Python 编译、自动化测试、Windows EXE 打包，并上传 `MapleStory-MiniGameAssistant-Reference-Windows` 构建产物。

## 使用
1. 下载并解压整个 Windows 构建文件夹。
2. 运行 `MapleStoryMiniGameAssistant.exe`。
3. 在“游戏窗口”选择冒险岛客户端，点击“捕获窗口”。
4. 确认捕获区域有效后按 F7 启动；F8 暂停；“停止”终止鼠标输入。
5. 建议先勾选“截图预览”，并在实际私服画面确认 ROI、目标锁定和结束标志识别，再启用自动确认。

## 已知边界
自动化测试和用户提供的参考截图回放不能代替用户电脑上的实际窗口/显卡验收。完全透明阶段只能根据最近轨迹进行预测；如果目标在完全不可见期间突然变向，纯视觉程序无法直接观测其新位置。OpenCL 加速的是白色掩膜计算，其余轮廓关联、轨迹预测和 UI 逻辑主要由 CPU 执行。
