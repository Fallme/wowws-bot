# 战舰世界自动作战实验项目

这是一个 Windows 本地实验工具：在网页中选择舰船、模式与运行限制后，由程序完成启动游戏、港口校验、选船、加入战斗、战斗监控和循环重开。所有运行数据、截图与日志都保存在项目目录中，复制或下载项目后不需要修改个人电脑路径。

## 一键启动

所有使用者只需双击：

```text
one_click_run.bat
```

它会自动完成以下步骤：

1. 创建项目内的 Python 虚拟环境。
2. 安装运行依赖。
3. 检测 NVIDIA 显卡；可用时安装 GPU OCR 运行时，失败或没有显卡时自动使用 CPU OCR。
4. 启动本地控制面板并打开浏览器。

首次运行需要联网，并要求已安装 Python 3.10 或更高版本、Steam 和《战舰世界》。之后仍只需要双击同一个 `one_click_run.bat`：它会检测控制台是否已经运行，避免重复启动服务。

## 使用方式

1. 先启动 Steam 并登录账号；游戏可开着，也可由控制面板启动。
2. 在浏览器控制面板选择战斗模式、舰船、局数或时长。
3. 点击“开始自动作战”。启动时游戏窗口默认最大化；用户切回网页或其他程序后，下一次需要下发游戏操作时会重新切回并保持最大化。
4. 控制台持续展示当前状态、小地图雷达、航速、航向、生命、火灾/漏水、消耗品和操作日志。

自定义舰船需要填写港口中显示的完整舰名与副炮射程。程序会将舰船栏滚回起点后单向遍历；连续无法确认时安全停止并提示重新选择。

## 快速战斗

勾选“快速战斗”后，单局达到五分钟、检测到沉没或离开战斗 HUD 时会立即 Esc 回港并重新开局。这类快速局不读取结算页，也不计入收益和对局统计。

## 数据保存位置

所有可变数据均在项目目录内自动生成：

```text
data/            控制台设置、任务历史、收益和运行状态
runtime/screenshots/runs/   每局诊断截图与结构化事件
runtime/screenshots/manual/ 手动截图工具输出
training_assets/user_captures/ 用户反馈截图（保留为回归与训练素材）
runtime/ocr_reports/  OCR 离线评估报告
```

这些目录不进入 Git，因此可以安全清理；删除后程序会在下次运行时自动重新创建。

## 打包给其他人

将项目推送到 Git 后，其他人下载或克隆仓库并双击 `one_click_run.bat` 即可完成安装。若需要生成一个独立压缩包，在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_release.ps1
```

压缩包会生成在项目内的 `dist/`，不包含虚拟环境、日志、账号信息和本机运行数据。

## 开发验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q main.py bot.py control_server.py core strategy tools tests
```

使用项目内的代表性截图离线回放场景识别、OCR 和拟下发驾驶指令：

```powershell
.\.venv\Scripts\python.exe .\tools\simulate_screenshot_scenarios.py
```

这个脚本不会查找游戏窗口，也不会发送键盘或鼠标操作。报告写入
`runtime/ocr_reports/screenshot_scenario_report.json`，包含场景、三项结算资源、生命值、航速、小地图态势与拟下发任务。

## 注意事项

本项目仅用于本地研究与受监督测试。游戏自动化可能违反服务条款并带来账号风险；程序不会规避反作弊，也不会在画面、输入或状态无法确认时继续运行。
