# 冒险岛小游戏助手 V2：核心架构与验收规范

> 本文件是后续实现的唯一技术基线。目标是复现参考软件的实际行为，不以界面相似或“程序能运行”代替功能验收。

## 1. 从参考软件确认的约束

- 识别 ROI：游戏客户区偏移 `(306, 119)`，尺寸 `759 × 563`；不猜测缩放、不拉伸图像。窗口移动时更新屏幕映射，客户区尺寸变化时暂停并要求重新捕获。
- 默认发送间隔 20 ms；这是调度目标，不承诺实际 50 FPS。置信度默认 0.30、鼠标平滑系数 0.85、死区 10 px、失焦恢复宽限 500 ms。
- 捕获：独立捕获进程。优先 Windows Graphics Capture（WGC）；只有系统检查明确表示不支持时才使用 DXGI 兼容后端。捕获初始化、首帧、帧龄、窗口映射和缓冲区生命周期必须独立校验。
- 推理：ONNX Runtime；Windows GPU 推理使用 DirectML。设备选择按 Windows 高性能 GPU 顺序枚举并用 DXGI LUID 映射实际设备，尝试独显、其他硬件和核显，全部失败才回退 CPU。不能把 OpenCL 掩膜运算冒充模型 GPU 推理，也不能把 CUDA 作为唯一 GPU 路径。
- 游戏流程：开始星标一次确认；结束标志双次确认；完成后点击确认并复检；一轮结束后继续等待下一轮。
- 结束不是一个状态：参考截图先出现“成功！”横幅，随后出现独立结果对话框；对话框右下方才有“确认”按钮。不得点击横幅文字中心代替确认按钮。
- 原软件资源受 AES-GCM / 授权机制保护。本项目独立实现识别模型和算法，不把解密或绕过原软件授权当作构建步骤。

## 2. 总体分层

```text
Windows Game Window
       |
       v
Capture Supervisor (generation / mappingVersion / QPC timestamp)
       |
       +--> WGC Capture Worker (separate process)
       |        `-- DXGI fallback only on explicit WGC unsupported
       |
       v
Frame Gate + ROI Mapper
       |  reject black, stale, future, wrong-size, old-generation frames
       v
Perception
       +--> ONNX segmentation/detection (ONNX Runtime + DirectML)
       +--> low-opacity residual / temporal evidence
       +--> start marker / success banner / result dialog / confirm button
       |
       v
Target Association + Motion Estimation
       +--> identity / shape descriptor / appearance
       +--> pyramidal LK optical flow while image evidence exists
       +--> background/global-motion compensation when reliable
       +--> IMM motion models (constant velocity / acceleration / turn)
       +--> uncertainty and reacquisition gate
       |
       v
Safe Interior Point Solver
       +--> segmentation mask erosion
       +--> distance transform / largest inscribed circle
       +--> warped target mask when flow is reliable
       |
       v
Input Gate -> Win32 SendInput -> game client
       |
       v
Game State Machine -> result dialog -> confirm -> verify -> next round

UI / diagnostics / replay harness observe the pipeline; they do not own tracking state.
```

## 3. 模块边界

### 3.1 Capture

**职责：** 只产生带元数据的原始帧，不识别目标、不控制鼠标。

每帧至少携带：
- `frameId`、单调时钟时间戳、接收时间；
- `generation`、`mappingVersion`；
- 游戏 HWND、客户区原点和物理像素尺寸；
- 捕获后端（WGC / DXGI）及实际帧尺寸。

使用有界 latest-frame 缓冲；新帧到达时允许丢弃旧帧，不允许推理线程积压后把过时结果送给鼠标。WGC 辅助进程崩溃或 IPC 数据无效时，立即使帧源失效并停止输入。

### 3.2 ROI Mapper

以游戏客户区为坐标基准，裁剪固定 ROI `(306,119,759,563)`。屏幕坐标只在输入边界换算一次。处理负屏幕坐标、多显示器和窗口移动；拒绝屏幕外点、窗口尺寸变化和映射版本不匹配的结果。禁止通过截图尺寸比例猜测坐标缩放。

### 3.3 Perception

主检测器采用**分割模型**而不只是 bbox 检测：需要输出白色不规则目标的像素掩膜，才能验证鼠标点确实在图形内部。训练数据覆盖：
- 清晰、半透明、低对比度、不同旋转角度和尺度；
- 参考录屏中的岩石纹理背景、倒计时数字、“开始”字样和游戏 UI 干扰；
- 无目标帧、结束横幅、结果对话框和确认按钮。

白色 HSV / 灰度阈值、形态学和连通域只作为辅助证据与模型失效诊断，不是唯一识别器。低透明度阶段使用多帧证据、局部残差和光流特征；单帧没有足够信号时不得伪造高置信度。

### 3.4 Target Tracker

状态至少包含 `[x, y, vx, vy, ax, ay, heading, scale]`、目标身份、最近可靠观测时间、形状描述、可见度和协方差。

- 可见阶段：检测掩膜与预测门控联合关联，使用形状、面积、方向、位置和置信度，避免把倒计时、伤害数字或背景亮块切换成目标。
- 半透明阶段：融合分割置信度、LK 金字塔光流、局部残差和运动先验；观测仍存在时，观测优先于预测。
- 完全不可见阶段：使用 IMM（匀速、匀加速、转弯）和多假设轨迹，而不是单一恒速直线外推；可靠的背景运动只作为辅助观测，不得无条件当成目标运动。
- 重新显现：按预测门控、形状/外观描述和时间连续性重新关联；若检测与预测冲突，先判断身份，不允许大幅跳点。
- 每次预测输出协方差/置信等级和可预测时长。完全透明且没有任何残余像素或其他运动线索时，真实坐标在信息论上不可直接观测；此时只能预测，必须显示不确定性，不能声称百分百准确。

### 3.5 Safe Interior Point

鼠标目标不是 bbox 中心。对当前掩膜先做安全边界腐蚀，再用距离变换求最大内切区域中心；若图形形状旋转或变形，重新计算内部点，不能永久复用上一次固定偏移。预测阶段只有在形状变换可信时才变换旧掩膜；掩膜失效时使用预测点并降低置信度。

### 3.6 Input Gate

所有鼠标输入集中在一个 Win32 控制模块，使用物理像素坐标。每次输入前检查：
- 游戏 HWND 仍有效且在前台；
- 帧的 generation / mappingVersion 仍为当前值；
- 结果没有过时；
- 预测协方差和 ROI 边界满足安全阈值；
- 当前状态允许跟随。

保留 0.85 平滑系数和 10 px 死区作为参考默认值；输入限速和实际帧率按真实时间计算。失焦立即停止输入，按 500 ms 恢复策略处理；停止、异常和捕获失效必须立即关闭输入 gate。

### 3.7 Game State Machine

```text
WAIT_NEXT_ROUND
  -> START_CANDIDATE (start star: one confirmation)
  -> ACQUIRE_TARGET
  -> TRACK_VISIBLE
  -> TRACK_FADE
  -> TRACK_PREDICT
  -> REACQUIRE (new observation corrects trajectory)
  -> END_CANDIDATE (end marker: two confirmations)
  -> WAIT_RESULT_DIALOG
  -> CONFIRM_BUTTON_CANDIDATE
  -> CLICK_CONFIRM
  -> VERIFY_DIALOG_CLOSED
  -> WAIT_NEXT_ROUND
```

横幅检测只触发“等待结果对话框”，不能直接点击横幅。确认按钮需要由对话框布局、按钮模板/颜色边界和“确认”文本区域共同验证；连续帧位置一致后才点击。点击后必须验证对话框消失；超过参考超时或重试上限时停止本轮输入并记录取消/异常，不无限点击、不误入下一轮。

## 4. GPU 与性能

- Windows 首选 ONNX Runtime DirectML，覆盖 NVIDIA / AMD / Intel；CPU 是最终回退。
- GPU 选择和模型推理设备必须记录真实 provider、adapter LUID 和设备名；不能只显示“GPU 已启用”。
- 模型 Session、预处理资源和可复用缓冲区在轮次/设备生命周期内复用，禁止每帧重新创建 OpenCL context、program、buffer 或 ORT session。
- 尽量减少 CPU↔GPU 往返；先量测端到端 capture→inference→tracking→input latency，再决定哪些预处理算子值得 GPU 化。
- 运行中 GPU 推理失败先停止输入并使当前 generation 失效；不在同一轮悄悄换设备。暂停后允许重新初始化设备或切换 CPU。

## 5. 可观测性与回放

每帧记录轻量结构化数据：帧号/时间、目标观测、掩膜面积、可见度、关联分数、运动模型权重、预测协方差、鼠标目标点、实际鼠标点、输入 gate 原因、各阶段耗时和实际推理设备。截图只按用户显式选择本地保存，不逐帧联网。

回放模式使用原始截图时间戳和帧序号，**不发送鼠标输入**。两轮参考截图共 1550 帧；截图中的绿色准星可提取为“参考软件鼠标轨迹”伪标签，用于行为对照，但它不是游戏隐藏目标的独立真值。白色目标仍可见的帧另用分割掩膜检查点是否在图形内部。禁止把伪标签写成真实目标标注。

## 6. 验收门槛

1. **可见目标：** 所有被判为有效的跟随点必须落在腐蚀后的目标掩膜内部；报告帧覆盖率、内部距离余量和误锁率。
2. **透明过渡：** 按透明度分桶报告检测召回率、点位抖动和连续跟踪时长；不能只统计“有坐标输出”。
3. **完全不可见：** 与参考软件鼠标轨迹伪标签比较误差分布（P50/P90/P95），同时报告预测时长和协方差；不将该项冒充真实隐藏目标误差。
4. **重新显现：** 报告重新锁定耗时、误关联次数和校正跳变。
5. **结束确认：** 两轮结果对话框截图均能识别右下角“确认”按钮；横幅阶段不得点击；点击后必须检测到对话框消失。
6. **循环：** 确认后清空本轮 tracker 状态，继续等待下一次开始标志；连续多轮不得重叠线程或重复点击。
7. **硬件：** CPU、DirectML NVIDIA、DirectML AMD、DirectML Intel（可用设备）分别记录实际 provider 和延迟；设备不可用时 CPU 回退结果正确。
8. **实机：** 自动化回放通过不等于私服实机通过。实机验收必须单独记录 Windows 版本、显卡/驱动、捕获后端、实际 ROI、FPS、P95 延迟和每轮结果。

## 7. 仓库和构建纪律

- 只有一份真实源代码；CI 直接构建仓库中的源文件和测试。
- 删除隐藏的 base64 源码包作为构建输入；不得出现“修改了 app.py 但 EXE 仍从另一份压缩源码构建”的情况。
- 每个 Windows 构建产物记录 commit SHA、源码 SHA-256、测试结果和依赖清单。
- 先通过离线两轮回放和状态机/输入替身测试，再生成 Windows 包；最后由用户在目标私服做实机验收。任何未做的步骤明确标为未验证。

## 8. 已完成的参考资料核验

- 两轮原始截图共 1,550 帧，单帧尺寸为 759×563。绿色准星可从截图中稳定提取为参考软件的鼠标输出轨迹；它只能作为**参考控制轨迹伪标签**，不能当作隐藏目标真实坐标。
- 两轮轨迹按时间对齐后，开始阶段约前 3.5 秒非常接近；约 4 秒后明显分叉。因此不能把一条固定时间坐标曲线硬编码成“透明跟踪”，应使用实时观测与多假设状态估计。
- 旧预测器在第一轮 499 个“透明后预测”帧上，与截图中的参考软件光标位置平均相差约 218 px（P50 约 235 px）。这是参考输出对照误差，不是隐藏目标真实误差；但足以否定旧预测器作为交付核心。
- 两轮结束截图的结果对话框布局一致。对话框模板在两张结束截图上的归一化相关系数约 0.9888；真正的橙色“确认”按钮位于 ROI 约 `x=483..522, y=347..363`。旧 `app.py` 的橙色轮廓检测按最大面积选目标，会选中其他橙色游戏元素；旧 `inminigame_assistant.py` 则把结束横幅框中心当作点击点。两种做法都不合格。
- 当前仓库存在两个构建源：根目录的 `app.py/tracker.py` 与 CI 从 `.build` 隐藏 base64 包恢复的 `inminigame_assistant.py`。修改根目录文件不等于修改最终 EXE。V2 必须先统一为仓库内唯一源码入口，再允许生成交付包。

这些数据结论用于决定架构和测试方式，不构成私服实机验收。


## 9. 当前代码实现状态（与目标架构分开核对）

| 子系统 | 当前仓库代码 | 状态 |
|---|---|---|
| 窗口捕获 | `mss` 同进程抓取窗口矩形 | 已实现；不是 WGC worker |
| 图形分割 | 灰度/HSV 多阈值、形态学、连通域 | 原型已实现；不是 ONNX 模型 |
| GPU | OpenCL 白色掩膜 kernel，设备缓冲区复用，CPU 回退 | 已实现；不代表 DirectML 模型推理 |
| 跟踪 | 单一 CV Kalman + 前后向 LK 光流桥接 | 原型已实现；不是 IMM |
| 安全点 | 连通域距离变换内点 | 已实现；透明阶段只对上一次掩膜做有限平移传播 |
| 结果确认 | 整个结果对话框模板 + 独立按钮模板 + 连续帧验证 + 关闭复检 | 已实现并有合成回归测试；仍需实机确认 |
| 输入安全 | 窗口选择快照、失焦门控、PyAutoGUI failsafe | 已实现基础门控 |
| DirectML / ONNX | 当前无可交付模型和推理 session | 未实现 |
| WGC / DXGI | 当前没有独立捕获进程 | 未实现 |
| IMM / 再识别 | 当前没有多模型滤波器和外观 ReID | 未实现 |

**验收原则：** 当前已实现项只说明仓库里有代码和自动化测试；未实现项不得对外宣称支持。透明阶段若光流点不足、前后向误差过大或目标完全消失，系统应降低置信度并在输入 gate 中停止危险动作，而不是无限期地把 Kalman 直线预测伪装成视觉跟踪。
