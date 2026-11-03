# 冒险岛小游戏助手

> 当前版本是可回放、可测试的 Python 原型。不要把架构目标写成已经交付的功能；透明阶段的准确度和目标私服实机表现仍需验收。

## 已落地

- 白色低饱和目标分割、连通域关联和安全内部点计算。
- 透明过渡时的前向/反向 LK 光流桥接；观测不足时退回带时间间隔的 Kalman 预测，协方差增长时降低置信度并暂停低置信度鼠标输入。
- OpenCL 白色掩膜加速，复用设备缓冲区；设备不可用时自动回退 OpenCV CPU。这里的 GPU 加速**只覆盖掩膜计算**。
- 结果对话框和“确认”按钮双模板匹配；连续 3 帧位置稳定后才点击；点击后连续 3 帧确认对话框消失才计入完成轮数。
- 游戏窗口选择快照、失焦输入门控和紧急停止。
- Windows CI 从仓库根目录唯一活动入口 `app.py` 构建，不再从隐藏压缩包还原另一份源代码。

## 尚未落地的 V2 目标

- Windows Graphics Capture（WGC）独立捕获进程及 DXGI 兼容后端；当前原型仍使用 `mss`。
- ONNX 分割模型及 ONNX Runtime DirectML 多显卡适配；当前原型没有可交付的训练模型，也没有 DirectML 推理。
- IMM 多运动模型、目标再识别和完整的时序模型；当前跟踪器仍以单一 Kalman 运动模型为主。
- 更大规模、带真实目标标注的数据集和完整 P50/P90/P95 回放评估。

这些项目不能仅靠改 README 或架构文档算作完成。尤其是完全透明且没有任何残余图像线索时，纯视觉系统无法直接观测真实目标位置。

## V2 技术规范

详见 [V2 核心架构与验收规范](docs/ARCHITECTURE_V2.md)。

## 本地构建

Windows / Python 3.11：

1. 运行 `build_exe.bat`。
2. 构建脚本先编译源码、运行测试，再由 PyInstaller 打包。
3. 运行 `dist/MapleStoryMiniGameAssistant/MapleStoryMiniGameAssistant.exe`。

## CI 构建

GitHub Actions 的 **Windows V2 desktop assistant build** 会从仓库当前源码运行编译、回归测试和 Windows 打包。构建产物名为 `MapleStory-MiniGameAssistant-V2-Windows`。

自动化测试通过不等于目标私服实机通过；实机验收需记录 Windows 版本、游戏窗口尺寸、显卡/驱动、实际后端、跟踪误差和结束确认结果。

## 离线回放评估

将参考截图目录放在本机，运行：

```powershell
python tools/replay_evaluate.py --frames-dir "D:\\path\\to\\小游戏记录" --output-csv tracker_replay.csv
```

脚本会提取绿色准星轨迹作为参考软件输出伪标签，并导出逐帧模式、预测坐标、置信度和误差分布。**伪标签误差不是隐藏目标真实误差**；必须结合可见目标内部点检查和实机验收解释。
