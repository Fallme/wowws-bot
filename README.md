抱歉还没完全跑通 

# 战舰世界自动化实验项目

这是一个仅面向本地人机模式研究的实验项目。当前版本采用自动自检与失败即停的闭环架构：用户在网页配置后只需点击开始，系统会自动启动游戏、检查港口与输入、进入战斗并按轮次或时间连续运行。

## 当前状态

- 自动测试验证代码逻辑，但不代表游戏一定接受输入。
- 默认使用游戏原生 `W/S/A/D` 的 Windows 虚拟键输入；开始任务时自动完成港口安全自检，不再要求预先手工校准。
- 游戏画面优先通过目标窗口直采，即使网页覆盖游戏也不会把网页误当成游戏；仅在直采不可用时回退桌面捕获。
- 运行中连续画面失效、HUD 不确定或位移无反馈都会触发安全熔断。
- 已接通港口、匹配、战斗、结算和下一局的自动循环，但在完成受监督的单局、连续三局和连续五局验收前，不标记为可长期无人值守版本。
- 控制台启动任务时若游戏未运行，会通过已安装的 Steam 自动启动游戏并等待港口；“持续运行”会不断开始下一局，直到手动停止。

## 快速开始

1. 在项目虚拟环境中安装依赖：

   ```powershell
   cd E:\aimemo\wowws-bot
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   默认键盘后端不依赖 ViGEmBus。仅当显式设置
   `WOWS_INPUT_BACKEND=vgamepad` 使用旧兼容后端时才需要 ViGEmBus。

2. 双击 `start_control_panel.bat` 打开网页控制台。
3. 在网页选择舰船、战斗模式以及轮次/时间。
4. 点击“启动任务”。系统自动启动游戏、等待港口、完成画面与输入自检、选择舰船和模式并进入战斗。
5. 正常情况下无需继续操作；只有画面、港口识别或输入闭环验证失败时，网页才显示人工介入提示。处理后点击“问题处理后重试”即可。

最近一次自动自检记录保存在 `data/input_calibration.json`。每次任务仍会重新执行实时自检，旧记录不会阻止点击开始。

每局结算后，程序会在确认完整结算页特征后点击右侧“继续战斗”直接排队下一局；按钮不可用或状态不确定时回港并走常规“加入战斗”流程。达到局数/时长限制后则返回港口，不再排队。

结算页会自动识别银币、舰船经验和指挥官经验，并按“任务 + 局数”去重写入网页收益统计。识别失败时不写入猜测值，网页会标记该局需要人工补录；OCR 与战斗距离识别共用 GPU 优先、CPU 兜底的推理策略。

## 安全链路

```text
网页点击开始
  -> 自动启动游戏并等待窗口
  -> 目标窗口直采与港口检查
  -> 输入安全释放自检
  -> 选择舰船和战斗模式
  -> 实时画面质量检查
  -> 强战斗 HUD 首帧确认（约 0.2 秒轮询）
  -> 指令派发
  -> 小地图玩家位置变化确认
  -> 闭环控制
```

任何一层失败都会执行虚拟手柄回中，并在控制面板显示具体原因。

运行时调试截图统一写入 `E:\aimemo\docs\screenshots\wowws_bot`，不再污染项目源码目录。

## 副炮船移动逻辑

移动策略以站点胜利条件为核心：中央占领点是持续航行目标，敌舰只在副炮圈外时对中央航线做小幅修正，不再因近敌或低血量提前掉头：

- 战斗 HUD 确认后立即下发四档全速，识别中央点后持续向点内推进。
- 点外保持满速；进入中央占领点圆圈后才降至约半速，并持续修正到点中心以减少漂出。
- 敌舰在副炮范围外时，中央点航向占主导、敌舰方位占小比例，尽量在进点途中把敌舰纳入副炮圈。
- 敌舰已经处于副炮范围时不再围绕敌舰掉头，继续向中央点推进或留点。
- 小地图固定按 10×10 网格、每格 5 km 换算敌距；敌舰标签 OCR 的真实公里数优先于网格估算。
- 只有当前前向航路确实与岛屿相交才绕行；预警距离缩短到约 2.8 km，约 1.4 km 内才允许短暂倒车脱困。
- 鱼雷规避保持全速，并优先选择岛屿净空侧。

参数位于 `config/ship.yaml`。同一目标需要连续两次 OCR 结果一致才会采用；准星海面距离没有敌舰血条证据时会被拒绝。OCR 在后台运行，不会阻塞战斗控制循环；默认优先使用 NVIDIA CUDA，只有 CUDA provider 不可用、初始化失败或实际推理失败时才回退 CPU。

控制面板实时显示目标距离及其 OCR/小地图来源、GPU/CPU、置信度、移动模式和决策理由。每局的结构化事件写入 `E:\aimemo\docs\screenshots\wowws_bot\run_*\events.jsonl`，可离线检查“点内仍满速”“点外无故减速”和“紧急岛距仍前进”等违规：

```powershell
.\.venv\Scripts\python.exe tools\replay_events.py E:\aimemo\docs\screenshots\wowws_bot\run_YYYYMMDD_HHMMSS\events.jsonl
```

## 项目结构

```text
wowws-bot/
├── calibrate_input.py       # 故障诊断时可选的受监督输入工具
├── main.py                  # 失败即停的运行生命周期
├── bot.py                   # 战斗观察、规则与闭环反馈
├── control_server.py        # 本地控制面板服务
├── core/
│   ├── calibration.py       # 校准凭证与有效性检查
│   ├── frame_guard.py       # 黑屏、坏帧和静止帧检测
│   ├── feedback.py          # 舰船位移反馈与安全熔断
│   ├── events.py            # 结构化事件总线与 JSONL 记录
│   ├── gamepad.py           # 虚拟手柄指令派发
│   ├── ocr.py               # 真实公里距离 OCR 与目标时序融合
│   ├── results.py           # 结算页收益 OCR
│   ├── replay.py            # 事件回放统计与安全断言
│   ├── vision.py            # 游戏画面识别
│   └── window.py            # 游戏窗口与受控点击
├── strategy/                # 副炮船移动策略
├── tools/                   # OCR 离线评估与事件回放工具
├── frontend/                # 本地控制面板
├── tests/                   # 单元与历史画面回归测试
└── training_assets/         # 保留的历史截图、YOLO 数据与旧素材
```

## 开发验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q main.py bot.py control_server.py calibrate_input.py core strategy tools tests
.\.venv\Scripts\python.exe tools\check_ocr_acceleration.py
.\.venv\Scripts\python.exe tools\evaluate_distance_ocr.py tests\fixtures\live_battle.png
```

测试分为三层：

- 纯逻辑测试：校准、帧质量、反馈熔断、状态策略。
- 历史画面回归：读取 `tests/fixtures` 中的精选实机截图。
- 受监督实机验收：必须由操作者确认游戏中的真实动作，不能用日志代替。

详细设计见 `docs/architecture.md`，MAA 思路下的下一阶段重构方案见 `docs/maa_inspired_architecture_plan.md`，验收门槛见 `docs/acceptance_criteria.md`。

## 训练素材

历史训练素材被保留但不直接进入运行路径：

- `training_assets/legacy_202607/training_data`
- `training_assets/legacy_202607/yolo`
- `training_assets/legacy_202607/dataset_v3`
- `training_assets/legacy_runtime/debug`
- `training_assets/legacy_runtime/snapshots`

旧训练脚本仅作为参考。以后若重新启用 YOLO，需要在新架构下建立独立、可测试的训练适配器。

## 风险说明

游戏自动化可能违反服务条款并导致账号处罚。本项目不规避反作弊、不保证游戏版本更新后的兼容性，也不会在无法证明输入和反馈有效时继续运行。
