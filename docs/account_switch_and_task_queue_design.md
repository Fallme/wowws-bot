# 批量账号切换可行性研究 & 任务序列调度器设计

> 调研时间：2026-09-18
> 调研对象：本机 wowsbot（`E:\aimemo\wowws-bot`）+ WeGame / WGC / Steam 发行渠道
> 结论先行：**WeGame 这条路对《战舰世界》不成立**；但**任务序列调度器完全可做**，且能不动 `RunnerManager` 与战斗循环地接入。

---

## 一、WeGame 现状核查

| 核查项 | 结果 |
|---|---|
| 本机是否安装 WeGame | **否**。`Program Files\Tencent`、`Program Files (x86)\Tencent`、`AppData`、注册表卸载项、全盘目录扫描均无 WeGame |
| 本机游戏本体 | `E:\SteamLibrary\steamapps\common\World of Warships\WorldOfWarships.exe` —— **Steam 版（WG 直营服）** |
| 《战舰世界》在 WeGame 上架吗 | **没有**。国服由 **360 代理**、走 **战游平台 WGC（Wargaming Game Center）**，用 360 账号登录；国际服走 Steam 或 WGC |

补充说明：WeGame 上确实有过《坦克世界》，但那是**空中网时代停更版**的历史遗留（WG 一度不回传版本包，空中网把停更老版本挂上 WeGame 试图榨剩余价值），与《战舰世界》无关，且该历史阶段早已结束。

**因此：不存在"通过 WeGame 批量切换《战舰世界》账号"这条路。** 需要切号的场景，动作发生在**游戏侧**（WGC 或游戏内登录界面），而不是 WeGame 侧。

---

## 二、账号切换的三条真实路径

### 路径 A · WGC 原生切号（推荐）

WGC 客户端**原生支持多账号**：左上角头像 → 切换账号即可在多个 WG 账号间跳转，无需退出客户端。360 国服官方下载公告原文即写明"可以添加或切换其他账号进行游戏登录"。

工程上还省一次下载：Steam 已装完整游戏后，安装 WGC 时选"使用现有游戏文件"指向 Steam 的 `World of Warships` 目录，WGC 会自动识别配置。之后 **Steam 入口固定绑一个账号，WGC 入口用来切号**，两者账号体系互不干扰。

- 自动化程度：**半自动**（切号本身可由脚本点，但登录常需验证码/扫码）
- 风控等级：低
- 结论：**首选**。任务调度器围绕它设计即可

### 路径 B · WGC 界面自动化切号

复用项目已有的感知/执行栈（`core/vision.py` 窗口识别 + `core/keyboard.py` / `gamepad.py` 输入 + `core/ocr.py`）去操作 WGC 窗口：定位头像 → 点"切换账号" → 填账号密码 → 点登录。

**硬断点**：腾讯/360 账号体系有强风控——异地登录、设备锁、图形验证码、滑块、短信验证。这些环节**无法在不破坏账号安全的前提下自动化**，必须人工接管。

- 自动化程度：**半自动**（识别与点击可自动，验证环节必须转人工）
- 风控等级：中
- 结论：可作为路径 A 的增强，但**不能承诺无人值守**。建议做成"自动走到验证环节 → 暂停 → 提示人工完成 → 继续"

### 路径 C · 登录令牌文件替换（不建议）

第三方资料提到 Steam 版游戏目录下存在自动登录令牌文件（形如 `res\WGC\autologin\config.xml`），备份不同账号的令牌后按需替换即可直进对应账号。

**列出来只为把选项讲全，明确不建议**：

1. 属灰色地带，官方不鼓励；操作不当会损坏文件，需走"验证游戏完整性"修复
2. 令牌会随改密/换设备/风控策略变化失效，稳定性差
3. 绕过正常登录流程，放大了被判定为异常登录的概率
4. 与"账号安全"直接冲突

- 自动化程度：可全自动
- 风控等级：高
- 结论：**不实现**

---

## 三、必须知道的约束

这几条不是"免责声明"，是会直接决定方案能不能落地的工程约束：

1. **验证码是硬停点。** 任何声称能全自动批量切腾讯系账号的方案，都在回避验证码问题。设计上必须把"人工接管"作为一等公民，而不是异常分支。
2. **同 IP 高频登录多账号会被风控标记。** 短时间内同一 IP 登录多个账号，容易被判定为工作室行为；公开资料提到部分腾讯游戏超过 4 个账号同时在线会被直接标记。任务之间**必须插入冷却**。
3. **协议层面有限制。** 《WeGame 软件许可及服务协议》第七章明确"同一身份信息可认证的软件账号数量将受到限制，不得超过合理数量"；《腾讯游戏许可及服务协议》6.5 条明确腾讯有权"根据您的游戏数据和表现异常判断您有无使用非法外挂或游戏同步器等作弊硬件行为"，并据此封号。
4. **本项目 README 已自述**：游戏自动化可能违反服务条款并带来账号风险。叠加多账号后，风险不是相加而是相乘。

**工程上的应对**：任务间强制冷却、单队列账号数设上限并给出提示、切号环节默认走人工确认、把"风控停点"当正常流程而非错误。

---

## 四、任务序列调度器设计

### 4.1 设计目标

一个队列 = 一串任务，可配循环次数、账号、船只、模式、局数/时长限制、任务间冷却与退出条件。**串行执行**（同一时刻仍只有一个游戏实例在跑，避免抢控制权）。

### 4.2 为什么不改底层

现有 `control_server.py` 的 `RunnerManager` 已经是一个设计良好的**单次运行**管理器：

- `start(payload)`：`_reconcile()` 后若 `self.process is not None` 则抛 `RuntimeError("已有任务正在运行")`
- `stop()` / `status()` / `_reconcile()`：进程生命周期、日志句柄、runtime_state 回读、DB 收尾都已封装

`main.py` 的战斗循环、`core/` 的感知与执行栈更不需要动。**调度器的职责只是"决定下一个 payload 是什么、什么时候发"，这正好是 `RunnerManager` 之上的一层。** 保持这个边界，改动风险最低、现有 700+ 测试基本不受影响。

### 4.3 数据模型

```python
@dataclass
class TaskSpec:
    # —— 下发字段，与 RunnerManager.start(payload) 的既有契约一致 ——
    ship: str                      # 内置船 key，或 "custom"
    mode: str                      # cooperative | asymmetric
    limit_type: str                # rounds | duration | continuous
    limit_value: float
    quick_battle: bool = False
    custom_ship_name: str = ""
    custom_secondary_range: float = 0.0
    custom_ship_preset_id: str = ""

    # —— 调度字段 ——
    repeat: int = 1                # 本任务重复次数
    account: str = ""              # 账号标识；空表示沿用当前账号、不切号
    cooldown_seconds: int = 60     # 本任务结束后的冷却
    on_error: str = "continue"     # continue | abort | retry
    max_retries: int = 1


@dataclass
class TaskQueue:
    name: str
    tasks: list[TaskSpec]
    loop_count: int = 1            # 整队循环次数；0 = 无限
    account_switch_mode: str = "manual"   # off | manual | wgc_auto
    stop_at: str = ""              # 定时停止，如 "07:30"
    max_duration_minutes: int = 0  # 总时长上限，0 = 不限制
    cooldown_between_tasks: int = 60
```

展开后得到**执行计划**：`flatten(tasks × repeat) × loop_count`。

示例：

```json
{
  "name": "夜间轮转",
  "account_switch_mode": "manual",
  "loop_count": 3,
  "max_duration_minutes": 480,
  "tasks": [
    {"ship": "pommern", "mode": "cooperative", "limit_type": "rounds", "limit_value": 5,
     "repeat": 2, "account": "acc_A", "cooldown_seconds": 90},
    {"ship": "custom", "custom_ship_preset_id": "p_7f3", "mode": "cooperative",
     "limit_type": "duration", "limit_value": 40, "account": "acc_B", "cooldown_seconds": 120}
  ]
}
```

### 4.4 QueueRunner 执行器

新增一个 supervisor 类，用守护线程跑循环：

```python
class QueueRunner:
    def __init__(self, runner: RunnerManager, store: ControlStore): ...

    def start(self, payload) -> str:      # 返回 queue_id，落库
    def stop(self)                        # 写 STOP 文件 + runner.stop()
    def resume_after_account_switch(self) # 人工切号完成后继续
    def status(self) -> dict              # 队列级状态，并入 /api/status

    def _supervisor(self):
        plan = expand(self.queue)
        loop = 0
        while self.queue.loop_count == 0 or loop < self.queue.loop_count:
            for item in plan:
                if self._should_stop(): return
                if item.account and item.account != self.current_account:
                    if not self._switch_account(item.account):   # 见 4.5
                        self._pause_for_manual(item.account); return
                self.runner.start(item.to_run_payload())
                while self.runner.status()["running"]:
                    if self._should_stop(): self.runner.stop(); return
                    sleep(1)
                self._record_task_result(item)
                self._cooldown(item.cooldown_seconds)
            loop += 1
        self._finish()
```

关键点：

- **`_should_stop()` 同时看**队列 stop 标志、总时长上限、`stop_at` 定时、以及 `max_duration`。这是队列级退出条件，与任务级 limit 是两个层次。
- **冷却复用现有 STOP/RESUME 文件模式**（项目已有 `WOWS_STOP_FILE` / `WOWS_RESUME_FILE` 机制），不引入新的 IPC。
- **一次只有一个进程**：`RunnerManager` 的单例语义被完整保留，天然避免"两个 runner 抢 runtime_state.json"。
- **错误处理**：任务失败时按 `on_error` 决定继续 / 中止 / 重试，并记录到队列日志。

### 4.5 账号切换钩子

`_switch_account()` 按 `account_switch_mode` 分派：

| 模式 | 行为 | 可行性 |
|---|---|---|
| `off` | 不切号，忽略 `account` 字段 | — |
| `manual`（默认） | 队列进入 `waiting_account` 态，前端弹横幅"请切换到账号 X，完成后点继续"，阻塞等待 `resume_after_account_switch()` | 高 |
| `wgc_auto` | 聚焦 WGC 窗口 → 定位头像 → 点"切换账号" → 填账号密码；**遇到验证码/扫码立即降级为 `manual`** | 中 |

`manual` 是默认值，也是唯一能保证不出事的模式。`wgc_auto` 复用现有 `core/window.py`（窗口绑定/DPI）+ `core/vision.py`（控件定位）+ `core/keyboard.py`（输入派发），**关键是不能宣称"切号成功"**——沿用项目既有的 `armed` vs `verified` 思路，必须由**验证环节之后的画面**（游戏主界面/港口）来确认切号真正生效，否则转人工。

### 4.6 HTTP API（沿用现有风格）

```
GET  /api/queue                 读取队列定义
POST /api/queue                 保存队列定义
POST /api/queue/start           启动队列
POST /api/queue/stop            停止队列
POST /api/queue/resume-account  manual 模式：确认人工切号完成，继续
GET  /api/status                现有接口，扩展 queue_* 字段
```

### 4.7 数据库

新增表（沿用 `ControlStore` 的锁 + `check_same_thread=False` 既有约定）：

- `task_queues`：队列定义与运行态
- `queue_items`：展开后的执行计划与逐项结果
- `runs` 增加 `queue_id` / `task_index` 两列，把每次运行挂回队列

> 注意：`control_server.py` 目前的迁移逻辑（`UPDATE runs SET completed_rounds=...`）**每次启动无条件执行、无"已迁移"标记**（此前排查发现的历史问题），新增列时建议一并引入 `schema_version` 表，避免同类问题累积。

### 4.8 前端

`frontend/index.html` + `app.js` 增加队列面板：

- 任务卡片列表，可增删改、拖拽排序
- 每卡片：船只（内置预设 / 自定义）、模式、限制方式与数值、重复次数、账号、冷却
- 队列级：循环次数、总时长上限、定时停止、切号模式
- 运行视图：当前 `loop i/N`、`任务 j/M`、剩余时间、下一个任务、人工切号提示横幅

现有 `state` 是扁平的单一任务模型，需要扩展为 `queue` 子对象；`renderConfig()` / `pollStatus()` 的现有结构可复用。

---

## 五、实施阶段

| 阶段 | 内容 | 依赖 |
|---|---|---|
| P0 | `TaskSpec` / `TaskQueue` 模型 + `QueueRunner` + `manual` 切号 + API + DB 迁移 | 无 |
| P1 | 前端队列编辑器（拖拽排序、循环、冷却、运行视图） | P0 |
| P2 | `stop_at` 定时停止、总时长上限、失败策略与重试 | P0 |
| P3 | `wgc_auto` 切号（含验证码降级到人工） | P0，且需先做 WGC 窗口绑定标定 |
| — | 令牌文件替换 | **不实现** |

P0 + P1 已能覆盖"设定任务序列、循环次数、船只、模式"的全部诉求，账号维度以人工确认方式接入。

---

## 六、风险与取舍

- **验证码不可绕过**，这是设计前提不是缺陷。方案把人工接管做成正常流程。
- **多账号叠加自动化，封号风险显著上升**。建议：账号数控制在个位数、任务间冷却不短于 60 秒、避开高频切号。
- **WGC 界面自动化需要标定**：窗口标题、控件位置、DPI 缩放都要用现有 `core/window.py` 的标定流程实测，不能照抄坐标。
- **改动集中在新层**，`RunnerManager` 与 `main.py` 不动，出问题可整体摘掉 `QueueRunner` 回到单任务模式。

---

## 七、虚拟机方案评估（补充调研）

思路：每个账号一台虚拟机，登录态各自独立，从此不再需要"切号"。

### 7.1 本机硬件快照（实测）

| 项 | 实测值 |
|---|---|
| CPU | Intel Core Ultra 9 275HX，24 核 / 24 线程（笔记本） |
| 内存 | 31.5 GB |
| 机型 | MECHREVO YAOSHI Series |
| 系统 | Windows 11 专业版，build 26200 |
| 独立显卡 | NVIDIA GeForce RTX 5070 Ti Laptop GPU，**11.9 GB 显存** |
| 集成显卡 | Intel Graphics |
| 虚拟化状态 | `HypervisorPresent: True`、`VirtualizationBasedSecurityStatus: 2`（VBS 运行中）；**Hyper-V 未安装**（无 `Get-VM`） |
| 已装虚拟化软件 | VMware Workstation **16.2.3**（`D:\vmware\`） |
| 磁盘余量 | C: **3 GB** / 300GB　D: 41 GB / 653GB　E: **111 GB** / 1908GB |
| 已装虚拟显示适配器 | Parsec、GameViewer、MuMu、向日葵（Oray）、LuminonCore —— 说明本机已在跑多层远控/模拟器 |

### 7.2 五个卡点

1. **GPU 直通不可用（决定性）**
   - 已安装的 **VMware Workstation 不支持 PCIe 直通**——直通是 ESXi 的能力，Workstation/Player 只有虚拟 SVGA 的有限 D3D 加速，跑不动 3D 游戏
   - 版本 16.2.3 发布于 2021 年，对 2025 年的 RTX 5070 Ti 与 D3D 特性级别支持都很勉强
   - 若改走 **Hyper-V + GPU-P（GPU 半虚拟化）**：Win11 专业版具备条件，但 Hyper-V 未安装，且消费级 NVIDIA **笔记本** GPU 上的 GPU-P 不受官方支持、成功率低
   - 若走 **Linux host + VFIO 直通**：笔记本只有 dGPU + iGPU 两块，涉及 MUX / Optimus 路由与 GPU reset 问题，实际不可行
   - 且 **VBS 已运行**，虚拟机只能以降级模式跑

2. **磁盘完全不够**
   - 一台 Windows 11 虚机系统盘至少 40–64 GB；WOWS 本体安装后约 **70 GB**
   - 单台 VM ≈ **130 GB**。E: 只剩 111 GB（**连一台都装不下**），C: 只剩 3 GB
   - 账号数 × 130 GB 的增长是线性的，两个账号就要 260 GB+

3. **内存最多勉强两台**
   - 整机 31.5 GB，宿主机自身占用后，每台游戏 VM 至少 8 GB
   - 2 台勉强，3 台不可行

4. **显存不够分**
   - 单块 dGPU 共 11.9 GB，需分给多台同时跑的 3D 游戏
   - 再叠加 bot 的 OCR 推理（运行时实测用的是 `CUDAExecutionProvider`）—— 显存争用会直接影响视觉识别的稳定性

5. **IP 与行为零隔离（概念性缺陷，见 7.3）**

### 7.3 概念性问题：VM 解决的不是真正的问题

账号被风控关联有三个维度，虚拟机只覆盖其中一个：

| 维度 | 多 VM | Windows 多用户账户 |
|---|---|---|
| 登录态隔离 | 隔离 | 隔离 |
| 设备指纹隔离 | 隔离 | 不隔离 |
| IP 与行为隔离 | **不隔离** | **不隔离** |

- 所有 VM 默认共用宿主机的公网出口，**同 IP 多账号登录**正是风控最直接的关联依据。要真正隔离必须给每台 VM 独立出口（代理/独立线路），这又会引入游戏延迟、代理稳定性与合规问题，复杂度和成本再上一个台阶
- 即便设备指纹不同，所有 VM 跑的是**同一套 bot**，走位与决策模式一致，服务端行为分析照样能关联。设备指纹隔离对此完全无能为力
- 游戏反作弊是否容忍在 VM 中运行，是未知数；不少游戏会直接拒绝或标记

### 7.4 额外成本

- **部署成本乘以 VM 数**：每台 VM 都要装 Python 环境、项目依赖、CUDA runtime，且要各自跑一个控制台实例
- **bot 与游戏必须同处一个 OS**：`core/window.py` 绑定游戏窗口、`mss` 抓屏、物理点击都依赖同机同会话。所以不能"宿主跑 bot、VM 跑游戏"，只能整栈进 VM
- 本机已经叠了 MuMu 模拟器、Parsec、向日葵、GameViewer 等多层虚拟显示与远控组件，再叠虚拟机，资源与稳定性都会互相打架

### 7.5 需要什么条件才值得上 VM

只有同时满足下列条件，VM 方案才划算：

- 有**独立显卡可整块直通**（台式机双卡，或支持 SR-IOV/vGPU 的专业卡）
- 有**充足磁盘**（每账号预留 150 GB 以上）
- 有**独立网络出口**（每 VM 一条线路）
- 接受**反作弊拒绝 VM** 的风险
- 目的是**并行**而非省去切号动作

### 7.6 结论与建议

**这台机器上不做虚拟机方案。** 卡点是硬性的（GPU 直通、磁盘、显存），且即使解决，它也覆盖不了 IP 与行为这两个真正的关联因素。

按目的分流建议：

| 真实目的 | 建议方案 | 成本 |
|---|---|---|
| 只是不想每次手动切号 | **Windows 多用户账户**——每个账号一个 Windows 用户，各有独立 profile，游戏登录态天然隔离 | 几乎为零 |
| 想无人值守连续跑多组任务 | 本文档第四节的 **QueueRunner 任务队列** + `manual` 切号 | 低 |
| 需要多账号并行 | 只能加**物理机**，本机不扩 | 高 |
| 想规避风控关联 | **没有可靠方案**；设备、IP、行为三个维度缺一不可，且行为分析无法靠隔离绕过 | — |
