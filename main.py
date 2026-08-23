"""World of Warships bot entry point and high-level lifecycle."""

import ctypes
import logging
import os
import time
from pathlib import Path

from bot import BattleBot
from config_loader import load_ship_config, ship_key_from_env
from core.calibration import (
    AUTOMATIC_PREFLIGHT_KEY,
    CalibrationStore,
    InputCalibration,
)
from core.input import configured_input_backend
from core.feedback import SafetyFault
from core.frame_guard import CaptureFault
from core.launcher import launch_game
from core.results import BattleRewards, ResultRewardReader
from core.ui import ScreenState
from core.window import activate_window, find_game_window, get_window_rect, physical_click
from port_navigator import (
    confirm_no_commander,
    enter_battle,
    ensure_requested_mode,
    handle_post_battle,
    queue_next_battle,
    select_requested_ship,
)
from runtime_control import RunLimits, RuntimeReporter

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("runner")


def configure_dpi_awareness():
    """Enable physical-pixel window coordinates before capture begins."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            logger.debug("Unable to enable DPI awareness", exc_info=True)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(BASE_DIR / "wowws_bot.log", encoding="utf-8"),
        ],
    )


def wait_for_game_window(timeout: float = 60.0, poll_interval: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        windows = find_game_window()
        if windows:
            return windows[0]
        time.sleep(poll_interval)
    return None


def wait_for_recognized_screen(bot: BattleBot, timeout: float = 300.0):
    """Wait through login/splash/loading until an actionable screen is visible."""
    deadline = time.monotonic() + max(1.0, float(timeout))
    last_state = ScreenState.UNKNOWN
    while time.monotonic() < deadline:
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        except CaptureFault as error:
            logger.info("游戏仍在启动，画面暂不可用: %s", error)
            time.sleep(2)
            continue
        last_state = bot.vision.classify_screen(image)
        if last_state in {
            ScreenState.PORT,
            ScreenState.BATTLE,
            ScreenState.RESULTS,
        }:
            return image, last_state
        time.sleep(2)
    return None, last_state


def automatic_input_preflight(bot, title, rect, screen_state, store=None):
    """Safely validate capture, focus and input dispatch before matchmaking.

    No combat action is sent in port.  We release all movement keys and verify
    that the same game window remains capturable.  Actual throttle acceptance
    is then verified by the existing closed-loop minimap feedback in battle.
    """
    store = store or CalibrationStore()
    if screen_state not in {ScreenState.PORT, ScreenState.BATTLE, ScreenState.RESULTS}:
        raise SafetyFault(f"自动自检无法确认当前游戏界面: {screen_state.value}")
    if not activate_window(bot.hwnd):
        raise SafetyFault("无法激活游戏窗口，输入自检未通过")
    time.sleep(0.4)
    bot.gamepad.stop()
    verification_frame = bot.vision.grab(bot.hwnd)
    verified_state = bot.vision.classify_screen(verification_frame)
    if verified_state == ScreenState.UNKNOWN:
        raise SafetyFault("输入释放后无法确认游戏画面，自动自检未通过")

    window_size = [rect[2] - rect[0], rect[3] - rect[1]]
    capture_backend = getattr(bot.vision.screen_capture, "last_backend", "unknown")
    record = InputCalibration(
        backend=configured_input_backend(),
        game_title=title,
        resolution=window_size,
        observations={
            AUTOMATIC_PREFLIGHT_KEY: {
                "passed": True,
                "screen": verified_state.value,
                "capture_backend": capture_backend,
                "input_check": "safe_release_dispatched",
                "battle_feedback": "required",
                "source": "automatic_runtime_preflight",
            }
        },
    )
    status = store.save(record)
    if not status.valid:
        raise SafetyFault(f"自动自检记录校验失败: {status.reason}")
    return status


def wait_while_loading(bot: BattleBot, timeout: float = 120.0, should_stop=None):
    deadline = time.monotonic() + timeout
    image = bot.vision.grab(bot.hwnd, allow_stale=True)
    while (
        bot.vision.classify_screen(image) == ScreenState.LOADING
        and time.monotonic() < deadline
    ):
        if should_stop and should_stop():
            break
        time.sleep(2)
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
    return image


def prepare_battle(bot: BattleBot, should_stop=None, configure_port=True):
    """Normalize the current menu state and request matchmaking."""
    logger.info("检查当前界面")
    time.sleep(1)
    image = wait_while_loading(bot, should_stop=should_stop)
    if should_stop and should_stop():
        return False
    state = bot.vision.classify_screen(image)

    if state == ScreenState.BATTLE:
        logger.info("已处于战斗 HUD，直接接管当前战斗")
        return True

    if state == ScreenState.RESULTS:
        logger.info("检测到结算界面，按状态导航返回港口")
        handle_post_battle(bot.hwnd, vision=bot.vision)
        time.sleep(2)
        # Result and port screens can legitimately remain pixel-identical for
        # several seconds; freshness is only a combat safety requirement.
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
        state = bot.vision.classify_screen(image)

    if state != ScreenState.PORT:
        logger.warning("当前界面为 %s，等待下一轮识别", state.value)
        return False

    logger.info("已确认港口，准备点击“加入战斗”")
    activate_window(bot.hwnd)
    time.sleep(1)
    if configure_port:
        ship_key = os.environ.get("WOWS_SHIP", "pommern")
        mode = os.environ.get("WOWS_MODE", "asymmetric")
        if not select_requested_ship(bot.hwnd, ship_key, vision=bot.vision):
            logger.warning("未能安全选择目标舰船")
            return False
        if not ensure_requested_mode(bot.hwnd, mode, vision=bot.vision):
            logger.warning("未能安全选择目标战斗模式")
            return False
    if not enter_battle(bot.hwnd, vision=bot.vision, configure_port=False):
        logger.warning("未能安全定位“加入战斗”按钮")
        return False
    time.sleep(5)
    return True


def wait_for_battle(bot: BattleBot, timeout: float = 180.0, should_stop=None):
    logger.info("等待战斗 HUD")
    deadline = time.monotonic() + timeout
    commander_confirmed = False
    last_state = None
    last_activation = 0.0
    while time.monotonic() < deadline:
        if should_stop and should_stop():
            return False
        now = time.monotonic()
        if (
            now - last_activation >= 2.0
            and int(ctypes.windll.user32.GetForegroundWindow() or 0) != bot.hwnd
        ):
            activate_window(bot.hwnd)
            note_activity = getattr(
                getattr(bot, "gamepad", None),
                "note_automation_activity",
                None,
            )
            if note_activity is not None:
                note_activity()
            last_activation = now
        time.sleep(0.20)
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
        state = bot.vision.classify_screen(image)
        if state != last_state:
            logger.info(
                "等待战斗识别: %s | 截图=%s",
                state.value,
                getattr(
                    getattr(bot.vision, "screen_capture", None),
                    "last_backend",
                    "unknown",
                ),
            )
            last_state = state
        if state == ScreenState.BATTLE:
            logger.info("战斗 HUD 首帧已确认，立即接管移动")
            return True
        if (
            state == ScreenState.UNKNOWN
            and not commander_confirmed
            and confirm_no_commander(bot.hwnd, image, bot.vision)
        ):
            commander_confirmed = True
            time.sleep(0.25)
            continue
    logger.warning("等待战斗开始超时")
    return False


def run_battle(bot: BattleBot, should_stop=None, progress=None):
    bot.reset()
    activate_window(bot.hwnd)
    autopilot_set = configure_opening_autopilot(bot)
    if not autopilot_set:
        reassert = getattr(bot.gamepad, "reassert_full_speed", None)
        if reassert is not None:
            reassert()
        else:
            bot.gamepad.full_speed()
    # Seed intervention monitoring after our focus and initial throttle input,
    # otherwise the opening Alt/W events can be mistaken for player control.
    intervention = getattr(bot, "intervention", None)
    if intervention is not None:
        intervention.reset()
    if autopilot_set:
        logger.info("进入战斗，已交由游戏自动航行驶向中央点位")
    else:
        bot.last_movement_reason = "自动航行设置失败，四档全速直行"
        logger.warning("战术地图自动航行设置失败，临时使用四档全速直行")
    last_progress = 0.0
    last_activation = time.monotonic()
    while bot.combat_tick() != "ended":
        if should_stop and should_stop():
            logger.info("收到停止请求，终止当前控制")
            return False
        now = time.monotonic()
        if (
            now - last_activation >= 5.0
            and bot.current_movement_mode != "manual_pause"
            and int(ctypes.windll.user32.GetForegroundWindow() or 0) != bot.hwnd
        ):
            activate_window(bot.hwnd)
            note_activity = getattr(bot.gamepad, "note_automation_activity", None)
            if note_activity is not None:
                note_activity()
            last_activation = now
        if progress and now - last_progress >= 1.0:
            progress(bot)
            last_progress = now
        time.sleep(0.3)
    logger.info("战斗结束")
    return True


def tactical_map_local_point(
    width: int,
    height: int,
    normalized_target: tuple[float, float],
) -> tuple[int, int]:
    """Map minimap-normalized coordinates onto the centred tactical map."""
    map_size = min(float(width), float(height)) * 0.94
    left = (float(width) - map_size) / 2.0
    top = (float(height) - map_size) / 2.0
    target_x = max(0.05, min(float(normalized_target[0]), 0.95))
    target_y = max(0.05, min(float(normalized_target[1]), 0.95))
    return (
        int(round(left + target_x * map_size)),
        int(round(top + target_y * map_size)),
    )


def configure_opening_autopilot(bot: BattleBot) -> bool:
    """Set one game-native autopilot destination on the tactical map."""
    toggle_map = getattr(bot.gamepad, "toggle_tactical_map", None)
    enable = getattr(bot, "enable_opening_autopilot", None)
    if toggle_map is None or enable is None or not hasattr(bot, "vision"):
        return False
    try:
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
        if bot.vision.classify_screen(image) != ScreenState.BATTLE:
            return False
        height, width = image.shape[:2]
        normalized_target = (0.5, 0.5)
        target_label = "地图中心"
        minimap = bot.vision.find_minimap(image)
        if minimap is not None:
            zone = bot.vision.find_central_capture_zone(minimap)
            if zone is not None:
                normalized_target = (
                    zone.center[0] / max(minimap.shape[1], 1),
                    zone.center[1] / max(minimap.shape[0], 1),
                )
                target_label = "中央占领点"
        local_x, local_y = tactical_map_local_point(
            width,
            height,
            normalized_target,
        )
        toggle_map()
        time.sleep(0.65)
        rect = get_window_rect(bot.hwnd)
        if not physical_click(
            rect["left"] + local_x,
            rect["top"] + local_y,
            extra_delay=0.1,
        ):
            toggle_map()
            return False
        time.sleep(0.35)
        toggle_map()
        time.sleep(0.35)
        enable(target_label)
        logger.info(
            "[SYSTEM] 战术地图自动航行: %s | local=(%s,%s)",
            target_label,
            local_x,
            local_y,
        )
        return True
    except Exception:
        logger.exception("设置战术地图自动航行失败")
        try:
            toggle_map()
        except Exception:
            pass
        return False


def collect_battle_rewards(bot, reader: ResultRewardReader, attempts: int = 12):
    """Read rewards before any result-page navigation occurs."""
    if bot.distance_ocr_service is not None:
        bot.distance_ocr_service.close()
    fallback = BattleRewards(
        provider=str(getattr(reader.backend, "execution_provider", "custom"))
    )
    for attempt in range(max(1, attempts)):
        if attempt == 0 and bot.last_analysis is not None:
            image = bot.last_analysis.image
        else:
            time.sleep(0.5)
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        if bot.vision.classify_screen(image) != ScreenState.RESULTS:
            continue
        rewards = reader.read(image)
        fallback = rewards
        if rewards.recognized:
            return rewards
    return fallback


def return_to_port(bot: BattleBot, attempts: int = 5):
    logger.info("等待结算并返回港口")
    for attempt in range(1, attempts + 1):
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
        state = bot.vision.classify_screen(image)
        if state == ScreenState.PORT:
            logger.info("已返回港口")
            return True
        logger.info("返回港口检查 (%s/%s): %s", attempt, attempts, state.value)
        if state == ScreenState.RESULTS:
            handle_post_battle(bot.hwnd, vision=bot.vision)
        elif state in {ScreenState.ESCAPE_MENU, ScreenState.EXIT_CONFIRMATION}:
            handle_post_battle(bot.hwnd, vision=bot.vision, max_steps=1)
            return False
        time.sleep(3)
    logger.warning("未能确认已返回港口；未执行盲点操作")
    return False


def wait_for_web_resume(limits, reporter, *, resume_state="preparing"):
    """Freeze workflow actions while preserving the exact lifecycle step."""
    if not limits.pause_requested():
        return True
    logger.info("[USER] 网页手动暂停；保留当前流程位置和舰船操纵状态")
    reporter.update(
        "paused",
        "网页已暂停，不再下发新系统指令",
        paused_by_user=True,
        manual_intervention_latched=True,
        movement_mode="manual_pause",
        movement_reason="保持现有船速与舵位，等待网页继续",
    )
    while limits.pause_requested():
        if limits.stop_requested():
            return False
        time.sleep(0.15)
    logger.info("[SYSTEM] 网页继续，立即重新识别当前画面并接续原流程")
    reporter.update(
        resume_state,
        "正在快速识别当前状态并继续原操作",
        paused_by_user=False,
        manual_intervention_latched=False,
    )
    return True


def run():
    configure_dpi_awareness()
    configure_logging()

    ship_key = ship_key_from_env()
    ship_config = load_ship_config(ship_key)
    mode = os.environ.get("WOWS_MODE", "asymmetric").strip().lower() or "asymmetric"
    if mode not in {"asymmetric", "cooperative"}:
        logger.error("不支持的战斗模式: %s", mode)
        return 1
    limits = RunLimits.from_env()
    reporter = RuntimeReporter(limits, ship=ship_key, mode=mode)
    logger.info(
        "舰船: %s | 副炮射程: %skm",
        ship_config["name"],
        ship_config["secondary"]["range"],
    )
    logger.info("搜索战舰世界窗口")
    reporter.update("starting", "正在搜索游戏窗口")
    window = wait_for_game_window(timeout=2)
    if window is None:
        result = launch_game()
        if not result.started:
            logger.error("无法自动启动游戏: %s", result.detail)
            reporter.update(
                "failed",
                "无法自动启动游戏",
                error="game_launch_failed",
            )
            return 1
        logger.info("已请求自动启动游戏: %s (%s)", result.method, result.detail)
        reporter.update("launching_game", "正在通过 Steam 启动战舰世界")
        launch_timeout = float(os.environ.get("WOWS_GAME_LAUNCH_TIMEOUT", "300"))
        window = wait_for_game_window(timeout=launch_timeout)
    if window is None:
        logger.error("未找到游戏窗口")
        reporter.update("failed", "未找到游戏窗口", error="game_window_not_found")
        return 1

    hwnd, title, rect = window
    logger.info("找到游戏窗口: %s", title)
    logger.info("窗口坐标: %s", rect)
    activate_window(hwnd)
    bot = BattleBot(hwnd, ship_config)
    reward_reader = ResultRewardReader(bot.distance_reader.backend)
    # Steam may expose the game window before login and port loading finish.
    reporter.update("entering_game", "游戏已启动，正在等待港口界面")
    screen_timeout = float(os.environ.get("WOWS_GAME_SCREEN_TIMEOUT", "300"))
    initial_frame, initial_state = wait_for_recognized_screen(
        bot,
        timeout=screen_timeout,
    )
    if initial_state == ScreenState.UNKNOWN:
        reporter.update(
            "failed",
            "启动前无法可靠识别游戏画面",
            error="preflight_screen_unknown",
            safety_state="blocked",
            calibration_valid=True,
        )
        bot.stop()
        return 2

    reporter.update("preparing", "正在执行港口画面与输入自动自检")
    try:
        calibration = automatic_input_preflight(
            bot,
            title,
            rect,
            initial_state,
        )
    except (SafetyFault, CaptureFault) as error:
        logger.error("自动自检失败: %s", error)
        reporter.update(
            "failed",
            "自动自检失败，需要人工检查游戏窗口后重试",
            error=str(error),
            safety_state="blocked",
            calibration_valid=False,
        )
        bot.stop()
        return 2
    capture_backend = getattr(bot.vision.screen_capture, "last_backend", "unknown")
    reporter.update(
        "preparing",
        "自动自检通过，正在准备战斗",
        calibration_valid=True,
        frame_status="ok",
        capture_backend=capture_backend,
    )
    logger.info("自动自检通过: %s | 捕获后端: %s", calibration.reason, capture_backend)

    logger.info("Bot 就绪，按 Ctrl+C 随时停止")
    started_at = time.monotonic()
    completed_rounds = 0
    current_round = 0
    port_configured = False
    battle_already_ready = initial_state == ScreenState.BATTLE
    preparation_failures = 0

    def should_stop():
        return limits.reached(completed_rounds, started_at)

    def user_stop_requested():
        return limits.stop_requested()

    try:
        while not should_stop():
            if not wait_for_web_resume(limits, reporter):
                break
            current_round = completed_rounds + 1
            logger.info("=== 第 %s 局 ===", current_round)
            reporter.update(
                "preparing",
                "正在准备下一局",
                current_round=current_round,
                completed_rounds=completed_rounds,
            )
            prepared = battle_already_ready
            battle_already_ready = False
            if not prepared:
                prepared = prepare_battle(
                    bot,
                    should_stop=should_stop,
                    configure_port=not port_configured,
                ) and wait_for_battle(bot, should_stop=should_stop)
            if not prepared:
                if should_stop():
                    break
                preparation_failures += 1
                if preparation_failures >= 5:
                    logger.error("连续 %s 次准备失败，停止运行", preparation_failures)
                    reporter.update(
                        "failed",
                        "连续准备失败，已安全停止",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        error="prepare_retry_limit_reached",
                    )
                    return 1
                time.sleep(min(2 * preparation_failures, 8))
                continue
            preparation_failures = 0
            port_configured = True
            reporter.update(
                "battle",
                "战斗已开始，等待闭环反馈",
                current_round=current_round,
                completed_rounds=completed_rounds,
                safety_state="armed",
                calibration_valid=True,
                movement_verified=False,
            )
            def report_battle_progress(active_bot):
                quality = active_bot.vision.last_frame_quality
                analysis = active_bot.last_analysis
                reporter.update(
                    "battle",
                    "闭环控制已确认"
                    if active_bot.movement_verified
                    else "等待舰船位移反馈",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    safety_state="verified"
                    if active_bot.movement_verified
                    else "armed",
                    calibration_valid=True,
                    movement_verified=active_bot.movement_verified,
                    frame_status=quality.reason if quality else "unknown",
                    capture_backend=getattr(
                        active_bot.vision.screen_capture,
                        "last_backend",
                        "unknown",
                    ),
                    target_distance_km=None
                    if analysis is None
                    else analysis.minimap_distance_km,
                    distance_source="unknown"
                    if analysis is None
                    else "minimap_grid"
                    if analysis.minimap_distance_km is not None
                    else "unknown",
                    minimap_distance_km=None
                    if analysis is None
                    else analysis.minimap_distance_km,
                    distance_confidence=0.0
                    if analysis is None
                    else analysis.distance_confidence,
                    target_track_id=""
                    if analysis is None or analysis.target_track_id is None
                    else analysis.target_track_id,
                    ocr_status=active_bot.ocr_status,
                    ocr_provider=active_bot.ocr_provider,
                    movement_mode=active_bot.current_movement_mode,
                    movement_reason=active_bot.last_movement_reason,
                    capture_point_distance_km=None
                    if analysis is None
                    else analysis.capture_point_distance_km,
                    inside_capture_point=False
                    if analysis is None
                    else analysis.inside_capture_point,
                    route_phase="unplanned"
                    if analysis is None
                    else analysis.route_phase,
                    route_progress=0.0
                    if analysis is None
                    else analysis.route_progress,
                    route_waypoint=0
                    if analysis is None
                    else analysis.route_waypoint,
                    route_arrived=False
                    if analysis is None
                    else analysis.route_arrived,
                    manual_intervention_latched=(
                        active_bot.manual_intervention_latched
                    ),
                    manual_intervention_seconds=(
                        active_bot.manual_intervention_seconds
                    ),
                    stop_after_current=bool(
                        limits.duration_seconds
                        and time.monotonic() - started_at
                        >= limits.duration_seconds
                    ),
                )

            if not run_battle(
                bot,
                # A time limit is a soft boundary: finish the active battle.
                # Only an explicit user stop interrupts combat immediately.
                should_stop=user_stop_requested,
                progress=report_battle_progress,
            ):
                if user_stop_requested():
                    reporter.update(
                        "stopped",
                        "已按用户要求安全停止",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                    )
                    return 0
                break
            if not wait_for_web_resume(
                limits,
                reporter,
                resume_state="collecting_rewards",
            ):
                break
            reporter.update(
                "collecting_rewards",
                "战斗结束，正在使用 OCR 统计本局收益",
                current_round=current_round,
                completed_rounds=completed_rounds,
                rewards_status="reading",
            )
            rewards = collect_battle_rewards(bot, reward_reader)
            if rewards.recognized:
                logger.info(
                    "本局收益: 银币=%s 舰船经验=%s 全局经验=%s (%s)",
                    rewards.credits,
                    rewards.ship_xp,
                    rewards.free_xp,
                    rewards.provider,
                )
                reporter.update(
                    "collecting_rewards",
                    "本局收益已自动统计",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    rewards_status="recognized",
                    rewards_round=current_round,
                    last_rewards=rewards.resource_values(),
                )
            else:
                logger.warning("结算页收益 OCR 未通过，本局不写入错误数据")
                reporter.update(
                    "collecting_rewards",
                    "未能可靠识别本局收益，战斗循环继续",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    rewards_status="unrecognized",
                    rewards_round=current_round,
                    last_rewards={},
                )
            completed_rounds += 1
            if should_stop():
                reporter.update(
                    "returning",
                    "运行计划已完成，正在返回港口",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                )
                return_to_port(bot)
                time.sleep(2)
                reporter.update(
                    "completed",
                    "运行计划已完成",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    movement_mode="idle",
                    movement_reason="计划已完成，控制已安全释放",
                    route_phase="unplanned",
                    route_progress=0.0,
                    route_waypoint=0,
                    route_arrived=False,
                    inside_capture_point=False,
                )
                return 0
            reporter.update(
                "requeueing",
                "战斗结束，正在自动进入下一局",
                current_round=current_round,
                completed_rounds=completed_rounds,
            )
            if not wait_for_web_resume(
                limits,
                reporter,
                resume_state="requeueing",
            ):
                break
            if queue_next_battle(bot.hwnd, vision=bot.vision):
                if wait_for_battle(bot, should_stop=should_stop):
                    battle_already_ready = True
                    continue
                if should_stop():
                    break
                logger.error("已点击继续战斗，但未能确认下一局 HUD，安全停止")
                reporter.update(
                    "failed",
                    "下一局已排队，但未能确认战斗 HUD",
                    current_round=current_round + 1,
                    completed_rounds=completed_rounds,
                    error="requeue_battle_not_confirmed",
                    safety_state="tripped",
                )
                return 2
            logger.warning("无法直接继续战斗，回港后使用常规入口重试")
            return_to_port(bot)
            time.sleep(2)
        manually_stopped = user_stop_requested()
        reporter.update(
            "stopped" if manually_stopped else "completed",
            "已按用户要求安全停止" if manually_stopped else "运行计划已完成",
            current_round=current_round,
            completed_rounds=completed_rounds,
            movement_mode="idle",
            movement_reason="控制已安全释放",
            route_phase="unplanned",
            route_progress=0.0,
            route_waypoint=0,
            route_arrived=False,
            inside_capture_point=False,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("用户中断")
        reporter.update(
            "stopped",
            "已停止",
            current_round=current_round,
            completed_rounds=completed_rounds,
        )
        return 0
    except (SafetyFault, CaptureFault) as error:
        logger.error("安全熔断: %s", error)
        reporter.update(
            "failed",
            "安全熔断，所有控制已释放",
            current_round=current_round,
            completed_rounds=completed_rounds,
            error=str(error),
            safety_state="tripped",
            calibration_valid=True,
        )
        return 2
    except Exception as error:
        logger.exception("运行失败")
        reporter.update(
            "failed",
            "运行发生错误",
            current_round=current_round,
            completed_rounds=completed_rounds,
            error=str(error),
        )
        return 1
    finally:
        bot.stop()
        logger.info("Bot 已停止")


if __name__ == "__main__":
    raise SystemExit(run())
