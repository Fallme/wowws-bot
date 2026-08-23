# MAA 思路下的战舰世界自动化重构设计

## 1. 结论

项目不直接复制 MAA 源码，也不把连续海战硬套成顺序点击脚本。采用以下混合架构：

- 外围流程使用资源驱动任务链：校准、港口、匹配、加载、战斗、结算、恢复。
- 战斗内部使用高频闭环：截图、目标跟踪、距离 OCR、地形分析、策略、输入、反馈。
- 识别资源、任务配置和程序代码分离，支持分辨率、语言和游戏版本覆盖。
- 每个识别结果保留证据、置信度和拒绝原因，支持录像离线回放。
- UI 只订阅事件和发送任务，不直接调用视觉或输入实现。

MAA 使用 AGPL-3.0-only。本项目只借鉴公开架构思想，所有实现重新编写，不复制其代码或资源。

## 2. 为什么不能照搬 MAA

MAA 的普通任务适合“识别页面 -> 点击 -> 等待 -> 识别下一页面”。战舰世界战斗是连续控制系统：舰船、敌舰和地形一直运动，航向决策需要每秒多次更新。

因此只在低频外围流程中使用任务图；战斗由强类型代码实现。JSON/YAML 可以描述阈值、ROI、资源和状态转移，但不得包含任意代码，也不得直接决定单帧舵角。

## 3. 总体架构

```text
Frontend / CLI
      |
Task API + Event Bus
      |
Task Engine ---------------- Resource Registry
  |                                  |
  |                           pipeline / profiles
  |                           templates / models
  v
Battle Runtime
  Capture -> Perception -> World Model -> Policy -> Safety Arbiter -> Controller
                |              |            |              |
                |              |            |              +-- 指令反馈
                |              |            +-- 公里距离带策略
                |              +-- 目标轨迹/时序融合
                +-- OCR/小地图/岛屿/HUD

Recorder <------ 帧、ROI、识别结果、决策、指令、反馈 ------> Replay Runner
```

## 4. 目录结构

```text
wowws-bot/
├── app/
│   ├── api/                  # 对 UI/CLI 的任务接口
│   ├── events/               # 统一事件模型和订阅
│   └── runtime/              # 生命周期、线程和安全熔断
├── core/
│   ├── controller/           # 截图、窗口、虚拟手柄
│   ├── vision/
│   │   ├── analyzers/        # HUD、目标、小地图、岛屿
│   │   ├── ocr/              # OCR 后端、预处理、文本解析
│   │   ├── tracking/         # 目标和观测时序跟踪
│   │   └── results.py        # 强类型视觉结果
│   ├── world/                # 当前战场状态与观测融合
│   └── safety/               # 优先级仲裁、失败即停
├── tasks/
│   ├── engine.py             # 低频任务链执行器
│   ├── battle.py             # 高频战斗运行时入口
│   └── handlers/             # 港口、匹配、结算、恢复
├── strategy/
│   ├── secondary.py          # 公里距离带策略
│   └── navigation.py         # 航向、岛屿和脱困
├── resource/
│   ├── pipeline/             # 任务图配置
│   ├── profiles/             # 分辨率/语言/版本 ROI 覆盖
│   ├── templates/            # UI 模板
│   ├── models/ocr/           # OCR ONNX 模型和字符表
│   └── schemas/              # 配置 JSON Schema
├── recorder/                 # 运行记录与 ROI 证据
├── replay/                   # 截图/录像离线回放
├── tools/                    # ROI 标注、OCR 评测、录像抽帧
├── frontend/
└── tests/
    ├── unit/
    ├── replay/
    └── fixtures/
```

第一阶段不强制一次性搬动全部旧文件；先建立新接口，旧模块通过适配器接入，验证后逐块替换。

## 5. 真实距离 OCR

### 5.1 识别位置

实战截图中敌舰标签检测结果约位于 `(1053, 626)`，实际距离文字位于该标签下方。距离 ROI 不应写死在屏幕中央，而应由当前目标标签动态生成：

```text
敌舰颜色/血条锚点
  -> 目标标签框
  -> 向下扩展距离 ROI
  -> 4 倍放大 + 多路二值化
  -> 数字 OCR
  -> 单位与范围校验
```

只有与当前跟踪目标空间重合的 OCR 结果才可以进入控制器。准星附近的海面距离、队友距离和小地图数字全部排除。

### 5.2 数据模型

```python
DistanceObservation(
    value_km: float,
    raw_text: str,
    confidence: float,
    bbox: Rect,
    target_track_id: str,
    source: "target_label_ocr",
    captured_at: float,
    accepted: bool,
    reject_reason: str | None,
)
```

策略层只能读取 `accepted=True` 的稳定观测，不能读取 OCR 原始字符串。

### 5.3 OCR 流程

1. 在检测到的敌舰标签下方生成动态 ROI。
2. 生成原色、灰度、白字掩码、CLAHE 四种预处理候选。
3. OCR 后端只识别 `0-9`、小数点、`km/公里`；后端通过统一接口可替换。
4. 文本规范化：全角转半角、逗号转小数点、常见 `O -> 0` 仅在数字上下文替换。
5. 用正则提取 `0.1–40.0 km`，单位缺失时降低置信度而不是直接接受。
6. 同一目标连续两帧数值相近才进入稳定轨迹；目标切换时清空旧距离。
7. 违反舰船物理速度的跳变、整数位突然变化、ROI 离开目标框全部拒绝。
8. 低置信度和拒绝样本自动保存 ROI、原始文本和元数据，进入后续标注集。

第一版使用通用 PaddleOCR/ONNX 数字识别后端完成数据闭环；积累足够样本后，再训练只识别数字与小数点的小型模型。运行时不得依赖网络服务。

### 5.4 多源融合规则

| 来源 | 用途 | 是否可决定副炮距离状态 |
|---|---|---|
| 目标标签 OCR | 真实公里距离，第一优先级 | 是 |
| 小地图目标方位 | 转向与目标关联 | 否 |
| 小地图归一化距离 | OCR 丢失时的保守降级距离带 | 仅降级 |
| 准星海面距离 | 调试数据 | 否 |

OCR 在后台线程运行；暂时丢失时保留最近稳定距离最多 5 秒，并根据时间增加不确定度。超过期限后退出公里距离带控制：允许保守接近，但禁止宣称“已进入理想距离”或“距离过近”。

## 6. 公里距离带策略

旧的 `minimap_distance` 只保留为辅助量，主状态改为公里：

| 状态 | 波美拉尼亚建议初值 | 行为 |
|---|---:|---|
| CHASE | `> 13.0 km` | 全速按目标方位追近 |
| BRAKE_IN | `11.5–13.0 km` | 提前减速，防止冲过副炮外缘 |
| HOLD_SECONDARY | `7.0–11.5 km` | 中速侧向绕行，保持副炮输出 |
| OPEN_DISTANCE | `4.5–7.0 km` | 降速并转离目标 |
| EMERGENCY_SEPARATE | `< 4.5 km` | 倒车转离，恢复安全间隔 |

那不勒斯根据自身 `secondary.range` 使用独立 profile。阈值必须是显式配置值，不能用比例在运行时暗算。

安全优先级保持：

```text
控制失联/坏帧 > 紧急岛屿 > 鱼雷 > 普通岛屿 > 低血量 > 距离带策略
```

## 7. MAA 式资源与任务系统

### 7.1 资源配置

每个识别任务包含：

```yaml
TargetDistanceOCR:
  recognizer: OcrDigits
  roi_from: CurrentEnemyLabel
  roi_offset: [-110, 12, 220, 58]
  preprocess: [raw, gray, white_mask, clahe]
  pattern: '(\d{1,2}[\.,]\d)\s*(km|公里)'
  confidence: 0.82
  stable_frames: 2
  timeout_ms: 1200
```

配置必须经过 Schema 校验。公共配置先加载，分辨率、语言和游戏版本 profile 后加载并覆盖同名字段，思路与 MAA 的多资源覆盖一致。

### 7.2 任务链

外围任务节点支持：

- `recognizer`：模板、OCR、颜色、特征或直接通过。
- `action`：点击、等待、启动战斗核心、停止。
- `next`：按顺序尝试下一节点。
- `on_error`、`on_timeout`：显式错误出口。
- `max_attempts`：禁止无限重试。
- `pre_delay_ms`、`post_delay_ms`。
- `evidence`：节点成功所需的连续帧和置信度。

战斗核心是一个任务节点，但其内部不使用这个顺序任务解释器。

## 8. 事件、记录和回放

统一事件至少包括：

- `TaskStarted/Completed/Failed`
- `FrameAccepted/Rejected`
- `TargetAcquired/Lost/Switched`
- `DistanceObserved/Accepted/Rejected`
- `MovementPlanned/Dispatched/Verified`
- `SafetyTripped`

每个事件包含 `run_id`、`frame_id`、时间戳、模块、数据和证据路径。控制面板只消费这些事件，避免再次出现“日志说执行成功，但游戏没反应”。

每次运行保存轻量 manifest；只保存状态切换、低置信度、拒绝和故障帧。Replay Runner 读取历史截图或录像，执行完全相同的视觉与策略代码，但使用空控制器，输出可比较的决策序列。

## 9. 并发和频率

- 截图：8–12 Hz，单生产者，最新帧覆盖旧帧。
- HUD/战斗状态：2–3 Hz。
- 目标跟踪：随控制循环更新；完整距离 OCR 当前在 CPU 后台约 0.1–0.25 Hz，始终只处理最新提交帧。后续使用已定位文本行的轻量识别器提升频率。
- 小地图、岛屿：3–5 Hz。
- 策略与输入：5 Hz；使用最新完整世界状态快照。
- 记录器：异步写盘，有界队列；写盘拥塞时丢弃普通帧，不阻塞控制。

任何模块不得积压旧帧后再补算；实时控制只处理最新画面。

## 10. 开发顺序

### 阶段 A：OCR 可行性基线

- 从保留素材与新录像中抽取敌舰距离 ROI。
- 建立 OCR 后端接口、数字解析器和离线评测工具。
- 不接控制器，只输出距离轨迹和拒绝原因。

通过门槛：标签可见样本读取率不低于 95%，平均绝对误差不高于 0.15 km，错误接受率低于 0.1%。

当前实现状态：已建立 RapidOCR/ONNX 离线后端、动态 ROI、敌舰血条交叉验证、候选回退和离线评测工具；ONNX Runtime 默认选择 CUDAExecutionProvider，只有 CUDA 不可用、初始化或推理失败时才回退 CPU。两张现有参考图均读取成功。现有样本量不足，尚不能声称达到上述统计门槛。

### 阶段 B：世界模型与公里策略

- 接入目标 track id、OCR 时序过滤和距离失效时间。
- 把移动策略从归一化距离切换为公里距离。
- 小地图距离降级为辅助信息。

当前实现状态：已完成。稳定的 OCR 公里距离优先，目标切换和物理不可能跳变会清空历史；岛屿、鱼雷和低血量安全策略优先级高于距离策略。

### 阶段 C：任务链和资源包

- 建立 Task Engine、Resource Registry、Schema 和 profile 覆盖。
- 迁移港口、匹配、加载和结算流程。

### 阶段 D：事件与回放

- 建立统一事件总线、运行 manifest、录像回放和差异报告。
- 控制面板显示 OCR 原文、公里值、置信度、目标 track 和证据 ROI。

当前实现状态：已完成事件总线、逐局 JSONL、控制面板公里值/置信度/移动决策，以及事件回放安全检查；证据 ROI 图片按低置信度筛选保存仍待补充。

### 阶段 E：受监督实战验收

- 不同距离、天气、地图、缩放与 UI 语言。
- 目标切换、目标遮挡、队友标签重叠。
- OCR 丢失时不冲脸，恢复后状态无跳变。
- 含岛航线和鱼雷同时出现时安全优先级正确。

## 11. 本轮不开发的内容

- 不直接复制或依赖 MAA Core。
- 不用网络 OCR。
- 不在 OCR 尚未离线达标前替换现有运行策略。
- 不用单帧 OCR 数字直接驱动油门或舵角。
- 不把大模型放进实时控制回路。
