"""World of Warships bot entry point and high-level lifecycle."""

import ctypes
import logging
import math
import os
import random
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
from core.window import (
    activate_window,
    ensure_game_window_foreground,
    find_game_window,
    get_client_rect,
    is_game_window,
    maximize_game_window,
    physical_click,
    set_interaction_pause_guard,
    window_message_click,
)
from port_navigator import (
    claim_daily_reward,
    enter_battle,
    ensure_selected_ship_commander,
    ensure_requested_mode,
    handle_post_battle,
    in_battle_type_selector,
    is_battle_survey_page,
    queue_next_battle,
    is_daily_reward_page,
    select_mode_from_screen,
    select_requested_ship,
    ShipSelectionError,
)
from runtime_control import RunLimits, RuntimeReporter

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("runner")


class GameWindowUnavailableWhilePaused(RuntimeError):
    """Raised after the paused workflow loses its game window for a grace period."""


QUICK_BATTLE_COMPLETION_REASONS = frozenset(
    {"quick_timeout", "quick_death", "quick_ended"}
)


def count_quick_battle_for_plan(completed_rounds: int, battle_finished) -> int:
    """Advance plan progress for a confirmed quick-battle completion only."""

    if battle_finished in QUICK_BATTLE_COMPLETION_REASONS:
        return max(0, int(completed_rounds)) + 1
    return max(0, int(completed_rounds))


def is_game_window_alive(hwnd) -> bool:
    """Inspect a window handle without restoring it or changing foreground focus."""
    try:
        return bool(hwnd and ctypes.windll.user32.IsWindow(int(hwnd)))
    except (AttributeError, OSError, TypeError, ValueError):
        # A platform/API inspection failure is not proof that the game closed.
        # Keep waiting rather than turning a diagnostic failure into a shutdown.
        return True


def shutdown_bot(bot) -> None:
    """Release controls only when their target window still exists.

    Closing OCR/event resources is safe for a stale handle.  Sending key-up or
    focus-management input is not: it can land in whichever application the
    user is currently using while the Web panel is paused.
    """
    if operation_paused(bot):
        logger.info("[USER] 暂停期间停止任务：不切游戏窗口，仅关闭内部资源")
        bot.stop(release_input=False)
        return
    if is_game_window_alive(getattr(bot, "hwnd", 0)):
        if ensure_game_window_foreground(bot.hwnd):
            bot.stop()
        else:
            logger.warning("无法确认游戏前台；跳过输入释放")
        return
    logger.info("游戏窗口已失效；跳过前台切换和输入释放，仅关闭内部资源")
    bot.stop(release_input=False)


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


def refresh_game_window(bot: BattleBot, *, maximize: bool = True) -> bool:
    """Reacquire a recreated game window without sending any game input."""
    if operation_paused(bot):
        logger.info("[USER] 暂停期间不重新绑定、不最大化游戏窗口")
        return False
    current = int(getattr(bot, "hwnd", 0) or 0)
    if is_game_window(current):
        return True
    windows = find_game_window()
    if not windows:
        logger.warning("原游戏窗口已失效，暂未发现新的战舰世界窗口")
        return False
    hwnd, title, _rect = windows[0]
    rebind = getattr(bot, "rebind_window", None)
    if rebind is not None:
        if not rebind(hwnd):
            return False
    else:
        bot.hwnd = hwnd
        intervention = getattr(bot, "intervention", None)
        if intervention is not None:
            intervention.hwnd = int(hwnd)
            intervention.reset()
    if maximize:
        maximize_game_window(hwnd)
    logger.info("已重新找到并绑定游戏窗口: %s (hwnd=%s)", title, hwnd)
    return True


def operation_paused(bot: BattleBot) -> bool:
    """Poll the keyboard/Web pause before any capture, focus or game action."""
    intervention = getattr(bot, "intervention", None)
    if intervention is None:
        return False
    paused = bool(
        intervention.poll(getattr(bot, "gamepad", None), time.monotonic())
    )
    if paused:
        mark_pause = getattr(bot, "mark_manual_pause", None)
        if mark_pause is not None:
            mark_pause()
    return paused


def ensure_bound_game_foreground(bot: BattleBot) -> bool:
    """Reacquire stale HWNDs, then foreground the verified game window."""
    # This is the final common gate for capture and input paths.  Callers may
    # already have checked pause, but checking again here closes the race where
    # the user presses a key between scene logic and SetForegroundWindow.
    if operation_paused(bot):
        logger.info("[USER] 暂停期间禁止切换或最大化游戏窗口")
        return False
    current = int(getattr(bot, "hwnd", 0) or 0)
    if ensure_game_window_foreground(current):
        # Returning from another app uses the game's normal maximized state;
        # ShowWindow does not drag or reposition the window.
        maximize_game_window(current)
        return True
    if not refresh_game_window(bot):
        return False
    return ensure_game_window_foreground(bot.hwnd)


def restore_game_foreground_after_pause(bot: BattleBot, source: str) -> bool:
    """Bring the game back only after the pause gate is positively clear."""
    if bot is None:
        return True
    # This first poll catches a key pressed at the exact end of the five-second
    # quiet window. ensure_bound_game_foreground polls again immediately before
    # the Windows focus call, closing the remaining race without stealing focus.
    if operation_paused(bot):
        logger.info("[USER] 暂停解除时检测到新的用户操作，刷新暂停时间，不切前台")
        return False
    logger.info("[SYSTEM] %s，尝试恢复《战舰世界》前台", source)
    if ensure_bound_game_foreground(bot):
        logger.info("[SYSTEM] 游戏前台已恢复，下一步重新识别当前场景")
        return True
    if operation_paused(bot):
        logger.info("[USER] 恢复前台前出现新的用户操作，本次恢复已取消")
    else:
        logger.warning("自动暂停已解除，但本次恢复游戏前台失败；后续安全重试")
    return False


def ensure_capture_foreground(bot) -> bool:
    """Foreground real runtime bots before screen capture.

    Lightweight vision fixtures intentionally have no input controller and no
    Windows window. Skipping activation for those adapters keeps observation
    functions testable without weakening the production rule.
    """
    if not hasattr(bot, "gamepad"):
        return True
    return ensure_bound_game_foreground(bot)


def classify_runtime_screen(bot, image) -> ScreenState:
    """Classify normal game pages plus OCR-confirmed first-login rewards."""
    state = bot.vision.classify_screen(image)
    if state != ScreenState.UNKNOWN:
        return state
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    detector = getattr(bot.vision, "is_daily_reward_page", None)
    try:
        if detector is not None:
            try:
                if detector(image, backend):
                    return ScreenState.DAILY_REWARD
            except TypeError:
                # Lightweight adapters may expose a one-argument detector.
                if detector(image):
                    return ScreenState.DAILY_REWARD
        elif is_daily_reward_page(image, backend):
            return ScreenState.DAILY_REWARD
    except Exception:
        # Reward detection is an optional refinement of UNKNOWN.  An OCR
        # provider fault must not abort global Esc recovery or the run loop.
        logger.debug("每日奖励页面 OCR 检查失败", exc_info=True)
    return state


def wait_for_recognized_screen(bot: BattleBot, timeout: float = 300.0):
    """Wait through login/splash/loading until an actionable screen is visible."""
    deadline = time.monotonic() + max(1.0, float(timeout))
    last_state = ScreenState.UNKNOWN
    previous_state = ScreenState.UNKNOWN
    consecutive = 0
    selector_consecutive = 0
    survey_consecutive = 0
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    while time.monotonic() < deadline:
        if operation_paused(bot):
            # Keyboard/Web pause applies before the first recognized page too.
            # Do not focus or capture until the quiet period (or Web resume).
            time.sleep(0.15)
            continue
        if not ensure_capture_foreground(bot):
            time.sleep(0.5)
            continue
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        except CaptureFault as error:
            logger.info("游戏仍在启动，画面暂不可用: %s", error)
            time.sleep(2)
            continue
        last_state = classify_runtime_screen(bot, image)

        # The optional post-battle satisfaction survey is neither a result
        # page nor a port.  Exact OCR lets us dismiss only this known dialog;
        # unknown pages continue through the ordinary bounded recovery path.
        try:
            survey_open = is_battle_survey_page(image, backend=backend)
        except Exception:
            logger.debug("战斗评价页面识别失败", exc_info=True)
            survey_open = False
        survey_consecutive = survey_consecutive + 1 if survey_open else 0
        if survey_consecutive >= 2:
            if operation_paused(bot):
                time.sleep(0.15)
                continue
            escape = getattr(getattr(bot, "gamepad", None), "escape", None)
            if escape is not None:
                logger.info("已确认战斗评价页面，按 Esc 关闭后重新判断场景")
                escape()
            survey_consecutive = 0
            previous_state = ScreenState.UNKNOWN
            consecutive = 0
            time.sleep(0.8)
            continue

        # Starting/restarting the bot while the battle-type picker is already
        # open used to leave the lifecycle observer waiting for five minutes:
        # the picker is intentionally neither PORT nor BATTLE.  Recover it as
        # its own stable scene and click only an OCR-located supported card.
        # This branch remains behind the global pause gate above, so it cannot
        # steal focus or send input during user intervention.
        try:
            selector_open = in_battle_type_selector(image, backend=backend)
        except Exception:
            logger.debug("战斗模式选择页识别失败", exc_info=True)
            selector_open = False
        selector_consecutive = (
            selector_consecutive + 1 if selector_open else 0
        )
        if selector_consecutive >= 2:
            requested_mode = (
                os.environ.get("WOWS_MODE", "asymmetric").strip().lower()
                or "asymmetric"
            )
            if operation_paused(bot):
                time.sleep(0.15)
                continue
            if select_mode_from_screen(
                bot.hwnd,
                requested_mode,
                image=image,
                backend=backend,
            ):
                logger.info(
                    "启动恢复：已在模式选择页点击目标模式 %s，等待返回港口复核",
                    requested_mode,
                )
            else:
                logger.warning(
                    "启动恢复：模式页未定位到目标卡片，按 Esc 返回港口后重试"
                )
                if not operation_paused(bot):
                    escape = getattr(getattr(bot, "gamepad", None), "escape", None)
                    if escape is not None:
                        escape()
            selector_consecutive = 0
            previous_state = ScreenState.UNKNOWN
            consecutive = 0
            time.sleep(0.8)
            continue

        if last_state == previous_state:
            consecutive += 1
        else:
            previous_state = last_state
            consecutive = 1
        if last_state in {
            ScreenState.PORT,
            ScreenState.DAILY_REWARD,
            ScreenState.BATTLE,
            ScreenState.RESULTS,
        } and consecutive >= 2:
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
    if not ensure_capture_foreground(bot):
        raise SafetyFault("无法激活游戏窗口，输入自检未通过")
    time.sleep(0.4)
    # Entering the controller while a battle is already under way must not
    # change throttle or cancel an in-game autopilot.  The old universal
    # "stop" probe was a safe port check, but it was a real brake command in
    # battle and is the direct cause of resumed battles starting at STOP.
    if screen_state == ScreenState.BATTLE:
        input_check = "battle_controls_preserved"
    else:
        bot.gamepad.stop()
        input_check = "safe_release_dispatched"
    verification_frame = bot.vision.grab(bot.hwnd)
    verified_state = bot.vision.classify_screen(verification_frame)
    if verified_state == ScreenState.UNKNOWN:
        raise SafetyFault("输入释放后无法确认游戏画面，自动自检未通过")

    # Steam first exposes a tiny transient WoWS surface (for example 202x56)
    # before the DirectX client finishes maximizing. Persist the live client
    # size observed at calibration time, never that stale launch rectangle.
    try:
        live_rect = get_client_rect(bot.hwnd)
        window_size = [live_rect["width"], live_rect["height"]]
    except Exception:
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
                "input_check": input_check,
                "battle_feedback": "required",
                "source": "automatic_runtime_preflight",
            }
        },
    )
    status = store.save(record)
    if not status.valid:
        raise SafetyFault(f"自动自检记录校验失败: {status.reason}")
    return status


def wait_while_loading(
    bot: BattleBot,
    timeout: float = 120.0,
    should_stop=None,
    should_abort=None,
):
    deadline = time.monotonic() + timeout
    if should_abort and should_abort():
        return None
    # Real runtime bots carry an input controller and must own the verified
    # game foreground before capture. Observation-only adapters (including
    # offline screenshot tests) deliberately have no controller and must not
    # be forced through Win32 window activation.
    if not ensure_capture_foreground(bot):
        return None
    image = bot.vision.grab(bot.hwnd, allow_stale=True)
    while (
        bot.vision.classify_screen(image) == ScreenState.LOADING
        and time.monotonic() < deadline
    ):
        if should_stop and should_stop():
            break
        if should_abort and should_abort():
            return None
        time.sleep(2)
        if should_abort and should_abort():
            return None
        if not ensure_capture_foreground(bot):
            time.sleep(0.5)
            continue
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
    return image


def prepare_battle(bot: BattleBot, should_stop=None, configure_port=True):
    """Normalize the current menu state and request matchmaking."""
    def port_action_paused():
        return operation_paused(bot)

    if port_action_paused():
        return False
    logger.info("检查当前界面")
    time.sleep(1)
    if port_action_paused():
        return False
    image = wait_while_loading(
        bot,
        should_stop=should_stop,
        should_abort=port_action_paused,
    )
    if image is None:
        return False
    if should_stop and should_stop():
        return False
    state = classify_runtime_screen(bot, image)

    if state == ScreenState.DAILY_REWARD:
        if port_action_paused():
            return False
        backend = getattr(
            getattr(bot, "distance_reader", None), "backend", None
        )
        if claim_daily_reward(
            bot.hwnd,
            image,
            backend=backend,
            should_abort=port_action_paused,
        ):
            logger.info("每日奖励领取操作已派发，重新识别页面后继续港口流程")
            time.sleep(1.0)
        return False

    if state == ScreenState.BATTLE:
        # A direct takeover is safe only after a stable HUD confirmation.
        # A single broad visual match must never skip ship/mode/matchmaking.
        confirmed_state = recover_current_scene(
            bot,
            attempts=3,
            stable_frames=2,
            poll_interval=0.20,
        )
        if confirmed_state == ScreenState.BATTLE:
            logger.info("已连续确认战斗 HUD，直接接管当前战斗")
            return True
        logger.warning(
            "战斗 HUD 未连续确认（复核=%s），不进入驾驶逻辑",
            confirmed_state.value,
        )
        state = confirmed_state

    if state == ScreenState.RESULTS:
        if port_action_paused():
            return False
        logger.info("检测到结算界面，按状态导航返回港口")
        handle_post_battle(
            bot.hwnd,
            vision=bot.vision,
            should_abort=port_action_paused,
        )
        time.sleep(2)
        # Result and port screens can legitimately remain pixel-identical for
        # several seconds; freshness is only a combat safety requirement.
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
        state = classify_runtime_screen(bot, image)

    if state != ScreenState.PORT:
        logger.warning("当前界面为 %s，优先尝试恢复到港口", state.value)
        return_to_port(bot, attempts=2)
        return False

    # Port actions are destructive to an active match (carousel scrolling and
    # clicks). Require a second fresh port frame and give a battle HUD absolute
    # priority. This catches transitions and any one-frame classifier error
    # before the ship-selection workflow is allowed to run.
    time.sleep(0.25)
    if port_action_paused():
        return False
    confirmation = bot.vision.grab(bot.hwnd, allow_stale=True)
    confirmed_state = classify_runtime_screen(bot, confirmation)
    if confirmed_state == ScreenState.BATTLE:
        logger.info("港口复核帧检测到战斗 HUD，取消选船并直接接管当前战斗")
        return True
    if confirmed_state != ScreenState.PORT:
        logger.warning(
            "港口状态未连续确认: first=%s second=%s；本轮不执行选船",
            state.value,
            confirmed_state.value,
        )
        return False

    logger.info("已连续确认港口，准备点击“加入战斗”")
    if configure_port:
        ship_key = os.environ.get("WOWS_SHIP", "pommern")
        mode = os.environ.get("WOWS_MODE", "asymmetric")
        if not select_requested_ship(
            bot.hwnd,
            ship_key,
            vision=bot.vision,
            should_abort=port_action_paused,
        ):
            logger.warning("未能安全选择目标舰船")
            return False
        if not ensure_selected_ship_commander(
            bot.hwnd,
            ship_key,
            custom_name=os.environ.get("WOWS_CUSTOM_SHIP_NAME", ""),
            backend=getattr(getattr(bot, "distance_reader", None), "backend", None),
            should_abort=port_action_paused,
        ):
            logger.warning("目标舰船缺少指挥官且自动召回失败")
            return False
        if not ensure_requested_mode(
            bot.hwnd,
            mode,
            vision=bot.vision,
            backend=getattr(getattr(bot, "distance_reader", None), "backend", None),
            should_abort=port_action_paused,
        ):
            logger.warning("未能安全选择目标战斗模式")
            return False
    if not enter_battle(
        bot.hwnd,
        vision=bot.vision,
        configure_port=False,
        should_abort=port_action_paused,
    ):
        logger.warning("“加入战斗”请求未能派发或未通过港口复核")
        return False
    # Return immediately. The lifecycle observer must see the loading screen;
    # sleeping here used to miss that transition and later mistake the current
    # battle for a new port workflow.
    time.sleep(0.1)
    return True


def wait_for_battle(
    bot: BattleBot,
    timeout: float = 180.0,
    should_stop=None,
    *,
    require_new_round: bool = False,
    loading_already_seen: bool = False,
):
    logger.info("等待战斗 HUD")
    deadline = time.monotonic() + timeout
    last_state = None
    result_frames = 0
    battle_frames = 0
    loading_seen = bool(loading_already_seen or not require_new_round)
    clock_frames = 0
    last_clock = None
    clock_backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    opening_attempted = False
    while time.monotonic() < deadline:
        if should_stop and should_stop():
            return False
        if operation_paused(bot):
            return False
        time.sleep(0.20)
        if operation_paused(bot):
            return False
        if not ensure_capture_foreground(bot):
            time.sleep(0.5)
            continue
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        except CaptureFault as error:
            logger.info("等待战斗时画面暂不可用: %s", error)
            if not refresh_game_window(bot):
                time.sleep(0.5)
            continue
        state = classify_runtime_screen(bot, image)
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
        if state == ScreenState.LOADING:
            loading_seen = True
            battle_frames = 0
            clock_frames = 0
            continue
        if state == ScreenState.BATTLE:
            if require_new_round and not loading_seen:
                battle_frames = 0
                clock_frames = 0
                continue
            if (
                require_new_round
                and not opening_attempted
                and hasattr(bot, "gamepad")
            ):
                # Start movement on the first reliable post-loading HUD frame.
                # Clock OCR can take several seconds on its first GPU call, so
                # start the engine immediately, then establish native
                # autopilot while lifecycle identity is confirmed in the
                # background.  A failed map click must never leave a ship at
                # STOP while the clock/OCR checks continue.
                opening_attempted = True
                reassert = getattr(bot.gamepad, "reassert_full_speed", None)
                try:
                    if reassert is not None:
                        reassert()
                    else:
                        bot.gamepad.full_speed()
                    setattr(bot, "_opening_motion_prestarted", True)
                    logger.info("新一局 HUD 已出现，已立即下发全速前进")
                except RuntimeError as error:
                    logger.info("开局全速指令被暂停/前台互锁撤销: %s", error)
                if configure_opening_autopilot(bot):
                    setattr(bot, "_opening_autopilot_preconfigured", True)
                    logger.info("新一局 HUD 已出现，已先行建立自动航线")
            battle_frames += 1
            if require_new_round:
                clock = bot.vision.read_battle_clock_seconds(image, clock_backend)
                if clock is not None and clock >= 15 * 60:
                    clock_frames += 1
                    last_clock = clock
                else:
                    clock_frames = 0
            else:
                clock_frames = battle_frames
        else:
            battle_frames = 0
            clock_frames = 0
        if battle_frames >= 2 and clock_frames >= 1:
            if require_new_round:
                logger.info(
                    "新一局证据已确认: 已经历加载画面，战斗计时=%02d:%02d",
                    int(last_clock or 0) // 60,
                    int(last_clock or 0) % 60,
                )
            logger.info("战斗 HUD 已连续确认，开始接管移动")
            return True
        result_frames = result_frames + 1 if state == ScreenState.RESULTS else 0
        if result_frames >= 3:
            logger.warning("等待新战斗时持续停留结算页，交回港口恢复流程")
            return False
        if state == ScreenState.UNKNOWN and bot.vision.in_no_commander_confirmation(
            image
        ):
            logger.warning(
                "等待战斗时检测到无指挥官拦截页；禁止继续，交回港口复核指定舰船"
            )
            return False
    logger.warning("等待战斗开始超时")
    return False


def run_battle(
    bot: BattleBot,
    should_stop=None,
    progress=None,
    *,
    resume_existing=False,
    quick_battle=False,
    quick_seconds=300.0,
):
    intervention = getattr(bot, "intervention", None)
    resume_motion_reasserted = False
    while intervention is not None:
        # Poll before the first focus/capture. A player who is already using
        # the keyboard must not have the game raised behind their input.
        if not intervention.poll(bot.gamepad, time.monotonic()):
            break
        mark_pause = getattr(bot, "mark_manual_pause", None)
        if mark_pause is not None:
            mark_pause()
        if should_stop and should_stop():
            return False
        time.sleep(0.15)
    # Focusing is coupled to an imminent game command. Merely observing a
    # paused game must never steal focus from the Web panel or another app.
    if not ensure_bound_game_foreground(bot):
        logger.warning("无法切换《战舰世界》到前台，跳过本次战斗控制")
        return False
    if hasattr(bot, "vision"):
        # Scene-to-action interlock: no reset, M, W, Q/E or tactical-map click
        # is legal until a fresh frame still carries the multi-anchor battle
        # HUD.  This catches a login/port transition that occurred after the
        # outer observer made its decision.
        if operation_paused(bot):
            return "resume_state"
        try:
            control_frame = bot.vision.grab(bot.hwnd, allow_stale=True)
        except TypeError:
            control_frame = bot.vision.grab(bot.hwnd)
        if classify_runtime_screen(bot, control_frame) != ScreenState.BATTLE:
            logger.warning("战斗动作互锁：最新画面已不是战斗，撤销驾驶并重新分流")
            return "resume_state"
        # Resuming an already-active battle bypasses ``wait_for_battle``, so it
        # also bypassed that function's first-HUD full-speed command. This was
        # especially visible after scene recovery: the bot spent several
        # seconds trying M-map autopilot while the telegraph remained STOP.
        # Reassert propulsion immediately unless the game already confirms a
        # native autopilot route; W would cancel that route.
        if resume_existing:
            autopilot_visible = False
            detector = getattr(bot.vision, "is_autopilot_enabled", None)
            if detector is not None:
                try:
                    autopilot_visible = bool(detector(control_frame))
                except Exception:
                    autopilot_visible = False
            if not autopilot_visible:
                reassert = getattr(bot.gamepad, "reassert_full_speed", None)
                if reassert is not None:
                    reassert()
                else:
                    bot.gamepad.full_speed()
                resume_motion_reasserted = True
                logger.info("恢复战斗 HUD 已确认，先立即重发全速前进，再配置航线")
    preconfigured_autopilot = bool(
        getattr(bot, "_opening_autopilot_preconfigured", False)
    )
    prestarted_motion = bool(
        getattr(bot, "_opening_motion_prestarted", False)
    )
    setattr(bot, "_opening_autopilot_preconfigured", False)
    setattr(bot, "_opening_motion_prestarted", False)
    try:
        bot.reset(
            preserve_movement=(
                resume_existing or preconfigured_autopilot or prestarted_motion
            )
        )
    except TypeError:
        # Compatibility for small test doubles and third-party adapters.
        bot.reset()
    autopilot_set = False
    if resume_existing or preconfigured_autopilot:
        # Re-read the live HUD before issuing any command. A stale workflow flag
        # must never send W/Q/E and cancel an already active game-native route.
        try:
            resume_frame = bot.vision.grab(bot.hwnd, allow_stale=True)
            autopilot_set = bool(
                bot.vision.classify_screen(resume_frame) == ScreenState.BATTLE
                and bot.vision.is_autopilot_enabled(resume_frame)
            )
        except (AttributeError, CaptureFault):
            autopilot_set = False
        if autopilot_set:
            enable = getattr(bot, "enable_opening_autopilot", None)
            if enable is not None:
                enable(
                    "新一局预先设置的自动航线"
                    if preconfigured_autopilot
                    else "恢复的游戏自动航线"
                )
        else:
            # A recovered battle must use the same opening rule as a freshly
            # detected battle: establish native autopilot first, then let the
            # Q/E controller take over only after the game route ends.
            autopilot_set = configure_opening_autopilot(bot)
    else:
        autopilot_set = configure_opening_autopilot(bot)

    if not autopilot_set and intervention is not None and (
        intervention.command_generation_paused()
        or intervention.poll(bot.gamepad, time.monotonic())
    ):
        mark_pause = getattr(bot, "mark_manual_pause", None)
        if mark_pause is not None:
            mark_pause()
        logger.info("[USER] 自动航行配置期间用户介入；不启用后备驾驶，等待重新判定场景")
        return "resume_state"

    if not autopilot_set:
        enable_center_route = getattr(bot, "enable_generic_center_route", None)
        if enable_center_route is not None:
            enable_center_route(
                "战术地图自动航行设置失败，通用驾驶向地图中央接管"
            )
        if not resume_motion_reasserted:
            reassert = getattr(bot.gamepad, "reassert_full_speed", None)
            if reassert is not None:
                reassert()
            else:
                bot.gamepad.full_speed()
    # Do not reset intervention state here. The monitor already distinguishes
    # injected keyboard ticks, while reset() would erase a real user keypress
    # that arrived during tactical-map setup and leak later commands.
    intervention = getattr(bot, "intervention", None)
    if preconfigured_autopilot and autopilot_set:
        bot.last_movement_reason = "加载结束即建立自动航线，游戏自动航行已开启，禁止Q/E"
        logger.info("进入战斗前已确认原生自动航行开启，不重复设置航点")
    elif resume_existing and autopilot_set:
        bot.last_movement_reason = "已重新识别当前战斗，游戏自动航行仍开启，禁止Q/E"
        logger.info("恢复当前战斗，确认原生自动航行仍开启，不发送W/Q/E")
    elif resume_existing:
        bot.last_movement_reason = "恢复战斗的自动航行设置失败，通用驾驶向点位/地图中心接管"
        logger.info("恢复当前战斗，自动航行未生效，通用驾驶按小地图接管")
    elif autopilot_set:
        logger.info("进入战斗，已交由游戏自动航行驶向地图中心")
    else:
        bot.last_movement_reason = "自动航行设置失败，通用驾驶向地图中央接管"
        logger.warning("战术地图自动航行设置失败，通用驾驶向地图中央接管")
    last_progress = 0.0
    quick_deadline = (
        time.monotonic() + max(30.0, float(quick_seconds))
        if quick_battle
        else None
    )
    non_battle_frames = 0
    pause_observed = False
    while True:
        if should_stop and should_stop():
            logger.info("收到停止请求，终止当前控制")
            return False
        now = time.monotonic()
        if quick_deadline is not None and now >= quick_deadline:
            logger.info("快速战斗已运行五分钟，退出本局回港；不统计收益")
            return "quick_timeout"
        intervention = getattr(bot, "intervention", None)
        # Observe real keyboard input before any foreground switch. Otherwise
        # the first user keystroke can race with SetForegroundWindow and the
        # game appears to ignore the requested pause.
        paused = bool(
            intervention is not None
            and intervention.poll(bot.gamepad, now)
        )
        if paused:
            # A keyboard/Web pause is stronger than any observation or focus
            # maintenance: do not grab a frame, raise the game, or generate a
            # command.  The next resume iteration immediately re-reads the
            # current scene before sending anything.
            mark_pause = getattr(bot, "mark_manual_pause", None)
            if mark_pause is not None:
                mark_pause()
            pause_observed = True
            if progress and now - last_progress >= 1.0:
                progress(bot)
                last_progress = now
            time.sleep(0.15)
            continue
        if pause_observed:
            if not restore_game_foreground_after_pause(
                bot, "自动暂停已解除"
            ):
                time.sleep(0.15)
                continue
            pause_observed = False
        if (
            not paused
            and int(ctypes.windll.user32.GetForegroundWindow() or 0) != bot.hwnd
        ):
            # The next combat tick may issue a command, so focus immediately
            # beforehand and mark the focus event as automation-generated.
            if not ensure_bound_game_foreground(bot):
                logger.warning("无法切换《战舰世界》到前台，本轮不发送控制指令")
                time.sleep(0.5)
                continue
            # Focus activation is not a game command.  Do not mark it as an
            # injected key event, otherwise a player's first keyboard input
            # immediately after switching windows can be mistaken for ours.
        try:
            tick_result = bot.combat_tick()
        except RuntimeError as error:
            if "游戏窗口不在前台" not in str(error):
                raise
            # A transient focus denial (for example while Windows finishes
            # switching a full-screen game) must not invalidate the whole
            # battle.  No key was sent; retry the next control frame.
            logger.warning("游戏前台切换尚未完成；跳过本帧控制并重试")
            time.sleep(0.5)
            continue
        if bool(getattr(bot, "autopilot_retry_pending", False)):
            # A confirmed loss before reaching the stored destination gets
            # one bounded native-map recovery cycle. The helper itself clicks
            # three progressively farther, enemy-biased destinations. Q/E is
            # forbidden throughout this cycle and begins only next frame if
            # all three verification checks fail.
            if operation_paused(bot):
                continue
            logger.warning(
                "原生自动航行提前失效，开始三次敌方方向航点重试；期间禁止Q/E"
            )
            if configure_opening_autopilot(bot, retrying=True):
                logger.info("原生自动航行重建成功，继续保持Q/E互锁")
            elif operation_paused(bot):
                # The retry request remains pending and will resume only after
                # the user releases the keyboard/Web pause.
                continue
            else:
                setattr(bot, "autopilot_retry_pending", False)
                enable_center_route = getattr(
                    bot, "enable_generic_center_route", None
                )
                if enable_center_route is not None:
                    enable_center_route(
                        "原生自动航行三次敌方偏移重试均失败，Q/E小地图驾驶接管"
                    )
                logger.warning(
                    "三次自动航点重试均未生效；下一控制帧启用Q/E小地图驾驶"
                )
            continue
        if tick_result == "ended":
            if quick_battle:
                # Quick battles never wait on or OCR the result page.  Whether
                # the five-minute limit ended the battle or the ship was sunk,
                # Esc immediately returns to port and the next loop queues a
                # fresh battle.  The lifecycle counts it toward plan progress
                # after this positively identified battle loop returns.
                logger.info("快速战斗已离开战斗 HUD，立即回港；计入计划局数但不统计收益")
                return "quick_ended"
            break
        last_analysis = getattr(bot, "last_analysis", None)
        if (
            tick_result == "waiting"
            and last_analysis is not None
            and not bool(getattr(last_analysis, "in_battle", False))
            and float(getattr(last_analysis, "health", 1.0)) > 0.01
        ):
            non_battle_frames += 1
            if non_battle_frames >= 3:
                logger.warning(
                    "连续三帧确认当前并非战斗 HUD，撤销战斗循环并按当前场景恢复"
                )
                return "resume_state"
        else:
            non_battle_frames = 0
        if (
            quick_battle
            and getattr(last_analysis, "health", 1.0) <= 0.01
        ):
            logger.info("快速战斗检测到舰船已沉没，退出本局回港；不统计收益")
            return "quick_death"
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
    # The tactical grid uses about 81% of the vertical viewport. The former
    # 94% estimate clicked outside the actual 10x10 grid on 2560x1600 and the
    # game silently ignored the autopilot destination.
    map_size = min(float(width), float(height)) * 0.81
    left = (float(width) - map_size) / 2.0
    top = (float(height) - map_size) / 2.0
    target_x = max(0.05, min(float(normalized_target[0]), 0.95))
    target_y = max(0.05, min(float(normalized_target[1]), 0.95))
    return (
        int(round(left + target_x * map_size)),
        int(round(top + target_y * map_size)),
    )


def configure_opening_autopilot(bot: BattleBot, *, retrying: bool = False) -> bool:
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
        # A/B/C/D circle detection is retained for the radar only.  Maps have
        # different layouts and a false circle must never redirect the opening
        # route.  The stable opening vector runs from the player's spawn,
        # through the map centre, and onward into the enemy half.
        normalized_target = (0.5, 0.5)
        player_normalized = None
        # A far-side target is meaningful only relative to the live white
        # player arrow.  Sample a few fresh frames because the marker can be
        # briefly covered by loading/friendly labels.  If it still cannot be
        # found, fail over to closed-loop minimap steering instead of clicking
        # the geometric centre and creating the short route reported by users.
        for pose_attempt in range(3):
            minimap = bot.vision.find_minimap(image)
            if minimap is not None:
                pose = bot.vision.find_player_pose_on_minimap(minimap)
                if pose is not None:
                    player_normalized = (
                        pose.position[0] / max(minimap.shape[1], 1),
                        pose.position[1] / max(minimap.shape[0], 1),
                    )
                    break
            if pose_attempt < 2:
                time.sleep(0.15)
                image = bot.vision.grab(bot.hwnd, allow_stale=True)
                if classify_runtime_screen(bot, image) != ScreenState.BATTLE:
                    return False
                height, width = image.shape[:2]
        if player_normalized is None:
            logger.warning(
                "连续三帧未定位小地图白色玩家箭头；拒绝设置错误短航点，交由通用驾驶接管"
            )
            return False
        enemy_bias = None
        if retrying:
            minimap_analyzer = getattr(bot.vision, "analyze_minimap", None)
            if minimap_analyzer is not None and minimap is not None:
                try:
                    enemies, _torpedoes = minimap_analyzer(minimap)
                except Exception:
                    enemies = []
                if 1 <= len(enemies) <= 16:
                    px, py = player_normalized
                    direction = (0.5 - px, 0.5 - py)
                    # Prefer the red contact farthest along the player's
                    # spawn-to-centre attack axis. It nudges the route toward
                    # the enemy half without following a contact behind us.
                    enemy = max(
                        enemies,
                        key=lambda point: (
                            (point[0] / max(minimap.shape[1], 1) - px) * direction[0]
                            + (point[1] / max(minimap.shape[0], 1) - py) * direction[1]
                        ),
                    )
                    enemy_bias = (
                        enemy[0] / max(minimap.shape[1], 1),
                        enemy[1] / max(minimap.shape[0], 1),
                    )
        target_label = "敌方方向重试航点" if retrying else "地图中心敌方远端"
        rect = get_client_rect(bot.hwnd)
        verify_autopilot = getattr(bot.vision, "is_autopilot_enabled", None)
        accepted = False
        local_x, local_y = tactical_map_local_point(width, height, normalized_target)
        selected_target = normalized_target
        map_open = bool(getattr(bot, "_tactical_map_left_open", False))
        for attempt in range(3):
            intervention = getattr(bot, "intervention", None)
            if intervention is not None and intervention.poll(bot.gamepad):
                mark_pause = getattr(bot, "mark_manual_pause", None)
                if mark_pause is not None:
                    mark_pause()
                logger.info("用户键盘介入，取消本次自动航行设置")
                return False
            # Every attempt is beyond the centre on the spawn-to-centre ray.
            # Retrying advances still farther into the enemy half, matching
            # the game's native route semantics without trusting unstable
            # capture-circle or main-viewport detections.
            attempt_target = normalized_target
            if player_normalized is not None:
                progress = (
                    (1.55, 1.75, 1.90)
                    if retrying
                    else (1.45, 1.65, 1.85)
                )[attempt]
                dx = 0.5 - player_normalized[0]
                dy = 0.5 - player_normalized[1]
                lateral = (
                    0.0
                    if enemy_bias is not None
                    else random.uniform(-0.018, 0.018) if attempt < 2 else 0.0
                )
                attempt_target = (
                    max(0.08, min(0.92, player_normalized[0] + dx * progress - dy * lateral)),
                    max(0.08, min(0.92, player_normalized[1] + dy * progress + dx * lateral)),
                )
                if enemy_bias is not None:
                    weight = (0.16, 0.22, 0.28)[attempt]
                    attempt_target = (
                        max(0.08, min(0.92, attempt_target[0] * (1.0 - weight) + enemy_bias[0] * weight)),
                        max(0.08, min(0.92, attempt_target[1] * (1.0 - weight) + enemy_bias[1] * weight)),
                    )
            local_x, local_y = tactical_map_local_point(
                width, height, attempt_target
            )
            selected_target = attempt_target
            if not map_open:
                toggle_map()
                map_open = True
                setattr(bot, "_tactical_map_left_open", True)
                time.sleep(0.65)
            else:
                logger.info("恢复此前暂停的战术地图落点操作")
            if intervention is not None and intervention.poll(bot.gamepad):
                mark_pause = getattr(bot, "mark_manual_pause", None)
                if mark_pause is not None:
                    mark_pause()
                logger.info("用户键盘介入，战术地图已停止继续操作")
                return False
            if not physical_click(
                rect["left"] + local_x,
                rect["top"] + local_y,
                extra_delay=0.1,
                hwnd=bot.hwnd,
            ):
                if not window_message_click(
                    bot.hwnd,
                    rect["left"] + local_x,
                    rect["top"] + local_y,
                    extra_delay=0.1,
                ):
                    toggle_map()
                    map_open = False
                    setattr(bot, "_tactical_map_left_open", False)
                    continue
            time.sleep(0.35)
            toggle_map()
            map_open = False
            setattr(bot, "_tactical_map_left_open", False)
            time.sleep(0.55)
            if verify_autopilot is None:
                accepted = True
                break
            verification = bot.vision.grab(bot.hwnd, allow_stale=True)
            if classify_runtime_screen(bot, verification) != ScreenState.BATTLE:
                logger.warning("自动航行复核时场景已离开战斗，立即撤销后续驾驶")
                return False
            if verify_autopilot(verification):
                accepted = True
                break
            logger.warning(
                "战术地图落点未出现自动驾驶标识，向敌方远端继续延伸 %s/3",
                attempt + 1,
            )
        if not accepted:
            logger.warning("战术地图三次渐进落点均未生效，交由通用驾驶接管")
            return False
        try:
            enable(target_label, target_normalized=selected_target)
        except TypeError:
            # Compatibility for light-weight test/custom bot adapters.
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
            intervention = getattr(bot, "intervention", None)
            if (
                getattr(bot, "_tactical_map_left_open", False)
                and (
                    intervention is None
                    or not intervention.command_generation_paused()
                )
            ):
                toggle_map()
                setattr(bot, "_tactical_map_left_open", False)
        except Exception:
            pass
        return False


def collect_battle_rewards(bot, reader: ResultRewardReader, attempts: int = 18):
    """Confirm the result page, then read rewards before navigation.

    Returns ``(page_confirmed, rewards, last_state)``.  A battle only counts
    when the result page is positively observed on consecutive frames.
    """
    if bot.distance_ocr_service is not None:
        bot.distance_ocr_service.close()
    fallback = BattleRewards(
        provider=str(getattr(reader.backend, "execution_provider", "custom"))
    )
    result_frames = 0
    page_confirmed = False
    last_state = ScreenState.UNKNOWN
    # OCR can finish one numeric column a frame later than the others while
    # the result panel animates. Vote per resource instead of requiring one
    # whole three-column tuple to match byte-for-byte; every value still needs
    # agreement from two independent result frames.
    field_votes: dict[str, dict[int, int]] = {
        "credits": {},
        "ship_xp": {},
        "free_xp": {},
    }
    for attempt in range(max(1, attempts)):
        if operation_paused(bot):
            logger.info("[USER] 结算读取暂停，不切窗口、不继续 OCR")
            return False, fallback, ScreenState.UNKNOWN
        if not ensure_capture_foreground(bot):
            time.sleep(0.5)
            continue
        if attempt == 0 and bot.last_analysis is not None:
            image = bot.last_analysis.image
        else:
            time.sleep(0.5)
            if operation_paused(bot):
                return False, fallback, ScreenState.UNKNOWN
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        last_state = bot.vision.classify_screen(image)
        port_card_detector = getattr(reader, "_looks_like_port_reward_card", None)
        port_reward_card = bool(
            port_card_detector is not None and port_card_detector(image)
        )
        battle_hud_detector = getattr(bot.vision, "_has_battle_hud", None)
        battle_hud_visible = bool(
            callable(battle_hud_detector) and battle_hud_detector(image)
        )
        # Never OCR a live battle as settlement even if transient button
        # colours fooled the coarse scene classifier.  This was the source of
        # enemy names/distances being parsed as XP and premature round counts.
        reward_surface = (
            last_state == ScreenState.RESULTS and not battle_hud_visible
        ) or port_reward_card
        if battle_hud_visible:
            last_state = ScreenState.BATTLE
        if not reward_surface:
            result_frames = 0
            if attempt >= 1 and last_state in {ScreenState.BATTLE, ScreenState.PORT}:
                break
            continue
        result_frames += 1
        page_confirmed = page_confirmed or result_frames >= 2
        rewards = reader.read(image)
        if rewards.recognized or not fallback.recognized:
            fallback = rewards
        if page_confirmed:
            values = rewards.resource_values()
            minimum_credits = int(
                getattr(reader, "MINIMUM_CREDITS", ResultRewardReader.MINIMUM_CREDITS)
            )
            if values["credits"] >= minimum_credits:
                field_votes["credits"][values["credits"]] = (
                    field_votes["credits"].get(values["credits"], 0) + 1
                )
            for field in ("ship_xp", "free_xp"):
                if values[field] > 0:
                    field_votes[field][values[field]] = (
                        field_votes[field].get(values[field], 0) + 1
                    )
            consensus = {}
            for field, votes in field_votes.items():
                confirmed = [
                    (count, value)
                    for value, count in votes.items()
                    if count >= 2
                ]
                if confirmed:
                    consensus[field] = max(confirmed)[1]
            if len(consensus) == 3:
                return (
                    True,
                    BattleRewards(
                        credits=consensus["credits"],
                        ship_xp=consensus["ship_xp"],
                        free_xp=consensus["free_xp"],
                        recognized=True,
                        provider=rewards.provider,
                        confidence=rewards.confidence,
                        raw_text=rewards.raw_text,
                        outcome=rewards.outcome,
                    ),
                    last_state,
                )
    if page_confirmed and not fallback.recognized:
        logger.warning(
            "结算页已确认但资源 OCR 未识别: raw=%s confidence=%s",
            fallback.raw_text,
            fallback.confidence,
        )
        debug_dir = getattr(bot, "_debug_dir", None)
        save_debug = getattr(bot.vision, "save_debug_frame", None)
        if debug_dir is not None and save_debug is not None:
            save_debug(
                Path(debug_dir)
                / f"result_unrecognized_{time.strftime('%H%M%S')}.png",
                image,
            )
    return page_confirmed, fallback, last_state


def return_to_port(bot: BattleBot, attempts: int = 5):
    logger.info("等待结算并返回港口")
    for attempt in range(1, attempts + 1):
        if operation_paused(bot):
            logger.info("[USER] 回港流程暂停，不切窗口、不发送 Esc")
            return False
        if not ensure_capture_foreground(bot):
            time.sleep(0.5)
            continue
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        except CaptureFault as error:
            logger.info("回港检查画面暂不可用: %s", error)
            refresh_game_window(bot)
            time.sleep(0.5)
            continue
        state = classify_runtime_screen(bot, image)
        backend = getattr(
            getattr(bot, "distance_reader", None), "backend", None
        )
        if is_battle_survey_page(image, backend=backend):
            if operation_paused(bot):
                return False
            escape = getattr(getattr(bot, "gamepad", None), "escape", None)
            if escape is not None:
                logger.info("检测到战斗评价页面，按 Esc 关闭并继续回港检查")
                escape()
            time.sleep(0.8)
            continue
        if state == ScreenState.PORT:
            logger.info("已返回港口")
            return True
        if state == ScreenState.DAILY_REWARD:
            backend = getattr(
                getattr(bot, "distance_reader", None), "backend", None
            )
            if claim_daily_reward(
                bot.hwnd,
                image,
                backend=backend,
                should_abort=lambda: operation_paused(bot),
            ):
                logger.info("每日登录奖励已领取，继续确认港口")
                time.sleep(1.0)
                continue
            logger.warning("每日奖励页面已识别，但领取按钮未能安全点击")
            return False
        if state == ScreenState.BATTLE:
            # A live match is not an unknown dialog.  Do not press Esc here:
            # the caller must hand control back to the battle loop.
            logger.info("仍在战斗中，取消回港操作并恢复战斗控制")
            return False
        if operation_paused(bot):
            return False
        logger.info("返回港口检查 (%s/%s): %s", attempt, attempts, state.value)
        if state == ScreenState.RESULTS:
            handle_post_battle(
                bot.hwnd,
                vision=bot.vision,
                should_abort=lambda: operation_paused(bot),
            )
        elif state in {ScreenState.ESCAPE_MENU, ScreenState.EXIT_CONFIRMATION}:
            handle_post_battle(
                bot.hwnd,
                vision=bot.vision,
                max_steps=1,
                should_abort=lambda: operation_paused(bot),
            )
            return False
        else:
            # Unknown dialogs must never receive a blind click.  Esc is the
            # game's universal back action and keeps recovery scoped to the
            # already-verified game client.
            escape = getattr(bot.gamepad, "escape", None)
            if escape is not None:
                logger.info("未知页面，按 Esc 尝试返回港口")
                escape()
        time.sleep(3)
    logger.warning("未能确认已返回港口；未执行盲点操作")
    return False


def recover_current_scene(
    bot: BattleBot,
    *,
    attempts: int = 6,
    stable_frames: int = 2,
    poll_interval: float = 0.25,
) -> ScreenState:
    """Observe the current page until one scene is stable on multiple frames.

    Recovery classification must be side-effect free.  In particular, a
    transient result/menu classification is not permission to click anything:
    the lifecycle caller decides what to do only after this observer returns a
    stable scene.
    """
    previous_state = ScreenState.UNKNOWN
    consecutive = 0
    sample_count = max(1, int(attempts))
    required = max(1, int(stable_frames))

    for attempt in range(sample_count):
        if operation_paused(bot):
            logger.info("[USER] 场景恢复暂停，不切窗口、不截取画面")
            return ScreenState.UNKNOWN
        if not ensure_capture_foreground(bot):
            logger.info(
                "恢复检查暂时无法激活游戏窗口 (%s/%s)",
                attempt + 1,
                sample_count,
            )
            time.sleep(max(0.0, float(poll_interval)))
            continue
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
            state = classify_runtime_screen(bot, image)
        except CaptureFault as error:
            logger.info(
                "恢复检查暂时无法取得画面 (%s/%s): %s",
                attempt + 1,
                sample_count,
                error,
            )
            state = ScreenState.UNKNOWN
            if not operation_paused(bot):
                refresh_game_window(bot)

        if state == ScreenState.UNKNOWN:
            previous_state = ScreenState.UNKNOWN
            consecutive = 0
        elif state == previous_state:
            consecutive += 1
        else:
            previous_state = state
            consecutive = 1

        if state != ScreenState.UNKNOWN and consecutive >= required:
            logger.info(
                "恢复检查连续 %s 帧确认当前场景: %s",
                consecutive,
                state.value,
            )
            return state

        if attempt + 1 < sample_count:
            time.sleep(max(0.0, float(poll_interval)))

    logger.warning("恢复检查未取得稳定场景；保持原状态且不发送任何操作")
    return ScreenState.UNKNOWN


def recover_after_battle_fault(
    bot: BattleBot,
    *,
    should_stop=None,
    attempts: int = 3,
    loading_timeout: float = 45.0,
) -> ScreenState:
    """Retry scene recognition after a battle-loop safety/capture fault.

    Loading is handled by waiting for the battle HUD.  Port, battle and result
    pages are returned to the lifecycle for phase-specific handling.  Unknown
    and interruption pages are observed again and never clicked blindly.
    """
    retry_count = max(1, int(attempts))
    last_state = ScreenState.UNKNOWN
    for attempt in range(1, retry_count + 1):
        if should_stop and should_stop():
            return ScreenState.UNKNOWN
        last_state = recover_current_scene(bot)
        logger.info(
            "战斗故障后场景恢复 (%s/%s): %s",
            attempt,
            retry_count,
            last_state.value,
        )
        if last_state in {
            ScreenState.BATTLE,
            ScreenState.RESULTS,
            ScreenState.PORT,
        }:
            return last_state
        if last_state == ScreenState.LOADING:
            try:
                if wait_for_battle(
                    bot,
                    timeout=max(1.0, float(loading_timeout)),
                    should_stop=should_stop,
                ):
                    return ScreenState.BATTLE
            except (SafetyFault, CaptureFault) as error:
                logger.info("等待加载恢复时仍无法取得稳定画面: %s", error)
        if attempt < retry_count:
            time.sleep(min(2.0 * attempt, 5.0))

    # An unresolved loading screen is useful to the outer preparation loop.
    # Every other unfamiliar page follows the global recovery rule: Esc is
    # sent only to the verified game client, then the ordinary port detector
    # decides whether recovery succeeded.  This is deliberately bounded so a
    # login/disconnect prompt cannot spin forever or receive blind clicks.
    if last_state == ScreenState.LOADING:
        return last_state
    logger.warning("场景连续未知，执行全局 Esc 回港恢复")
    if return_to_port(bot, attempts=3):
        return ScreenState.PORT
    return ScreenState.UNKNOWN


def wait_for_web_resume(
    limits,
    reporter,
    bot=None,
    *,
    resume_state="preparing",
    window_missing_timeout=5.0,
    poll_interval=0.15,
):
    """Freeze lifecycle actions for Web pause or real keyboard intervention."""
    intervention = getattr(bot, "intervention", None)

    def keyboard_paused():
        return bool(
            intervention is not None
            and intervention.poll(getattr(bot, "gamepad", None))
        )

    web_paused = limits.pause_requested()
    key_paused = keyboard_paused()
    if not web_paused and not key_paused:
        return True
    source = "网页手动暂停" if web_paused else "用户键盘介入"
    logger.info("[USER] %s；保留当前流程位置和舰船操纵状态", source)
    reporter.update(
        "paused",
        f"{source}，暂停下发新系统指令",
        paused_by_user=True,
        manual_intervention_latched=bool(
            web_paused or (intervention is not None and intervention.latched)
        ),
        movement_mode="manual_pause",
        movement_reason="保持现有船速与舵位；5秒静默后自动恢复，持续20秒则等待网页继续",
    )
    window_missing_since = None
    latch_reported = bool(
        web_paused or (intervention is not None and intervention.latched)
    )
    while True:
        if limits.stop_requested():
            return False
        web_paused = limits.pause_requested()
        key_paused = keyboard_paused()
        if not web_paused and not key_paused:
            break
        if (
            not latch_reported
            and intervention is not None
            and intervention.latched
        ):
            latch_reported = True
            logger.warning("[USER] 持续键盘介入达到20秒，已锁定暂停并等待网页继续")
            reporter.update(
                "paused",
                "持续键盘介入达到20秒，需在网页点击继续",
                paused_by_user=True,
                manual_intervention_latched=True,
                movement_mode="manual_pause",
                movement_reason="保持既有操纵状态，等待网页手动继续",
            )
        now = time.monotonic()
        if bot is not None and not is_game_window_alive(getattr(bot, "hwnd", 0)):
            if window_missing_since is None:
                window_missing_since = now
                logger.warning("暂停期间游戏窗口句柄失效，开始无操作确认")
            elif now - window_missing_since >= max(
                0.0, float(window_missing_timeout)
            ):
                reporter.update(
                    "failed",
                    "暂停期间游戏窗口已关闭，任务已安全退出",
                    paused_by_user=False,
                    manual_intervention_latched=False,
                    movement_mode="idle",
                    movement_reason="未切换前台、未向失效窗口发送控制指令",
                    error="game_window_unavailable_while_paused",
                    safety_state="blocked",
                )
                raise GameWindowUnavailableWhilePaused(
                    "暂停期间连续无法找到原游戏窗口"
                )
        else:
            # Minimize, foreground changes and one-off handle inspection
            # failures do not trip the worker; only one stale HWND sustained
            # for the full grace period does.
            window_missing_since = None
        time.sleep(max(0.01, float(poll_interval)))
    resumed_by_web = bool(
        web_paused
        or (
            intervention is not None
            and getattr(intervention, "resumed_from_web", False)
        )
    )
    resume_source = "网页点击继续" if resumed_by_web else "5秒内无新的键盘输入"
    logger.info("[SYSTEM] %s，开始重新识别当前画面并接续原流程", resume_source)
    reporter.update(
        resume_state,
        "暂停已解除，正在恢复游戏前台并判断当前状态",
        paused_by_user=False,
        manual_intervention_latched=False,
    )
    if bot is not None:
        restore_game_foreground_after_pause(bot, resume_source)
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
    reporter = RuntimeReporter(
        limits,
        ship=ship_key,
        mode=mode,
        ship_display_name=ship_config.get("display_name", ship_config["name"]),
    )
    logger.info(
        "舰船: %s | 副炮射程: %skm",
        ship_config["name"],
        ship_config["secondary"]["range"],
    )
    if limits.quick_battle:
        logger.info("快速战斗已开启：单局最多五分钟或沉没即 Esc 回港，不统计收益")
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
    bot = BattleBot(hwnd, ship_config)
    # Close the race between an upper-level pause check and a lower-level
    # multi-attempt focus/click operation.  Every side effect in core.window
    # now consults the live keyboard/Web intervention state itself.
    set_interaction_pause_guard(lambda: operation_paused(bot))
    reward_reader = ResultRewardReader(bot.distance_reader.backend)
    # The pause monitor must exist before the first maximize/foreground action.
    # Web pause or keyboard intervention during Steam/login therefore leaves
    # the user's current page untouched until they resume.
    if not wait_for_web_resume(
        limits,
        reporter,
        bot,
        resume_state="starting",
    ):
        bot.stop(release_input=False)
        return 0
    reporter.update("starting", "已找到游戏窗口，正在默认最大化")
    if ensure_bound_game_foreground(bot):
        logger.info("已在启动时最大化游戏窗口；后续切换不再改动窗口位置")
    else:
        logger.info("暂停或窗口暂不可用，未切换/最大化游戏窗口")
    # Steam may expose the game window before login and port loading finish.
    reporter.update("entering_game", "游戏已启动，正在等待港口界面")
    screen_timeout = float(os.environ.get("WOWS_GAME_SCREEN_TIMEOUT", "300"))
    initial_frame, initial_state = wait_for_recognized_screen(
        bot,
        timeout=screen_timeout,
    )
    if initial_state == ScreenState.DAILY_REWARD:
        reporter.update(
            "preparing",
            "检测到每日登录奖励，正在领取后返回港口",
        )
        backend = getattr(
            getattr(bot, "distance_reader", None), "backend", None
        )
        if claim_daily_reward(
            bot.hwnd,
            initial_frame,
            backend=backend,
            should_abort=lambda: operation_paused(bot),
        ):
            time.sleep(1.0)
        # Re-enter the scene loop even if the first click was not confirmed:
        # return_to_port re-captures the page, retries a still-visible reward,
        # and uses the global Esc rule for the post-claim/unknown overlay.
        return_to_port(bot, attempts=4)
        initial_frame, initial_state = wait_for_recognized_screen(
            bot,
            timeout=min(screen_timeout, 30.0),
        )
    if initial_state == ScreenState.UNKNOWN:
        reporter.update(
            "recovering",
            "启动时页面未知，正在按 Esc 返回港口并重新识别",
            safety_state="armed",
            calibration_valid=True,
        )
        if return_to_port(bot, attempts=3):
            initial_frame, initial_state = wait_for_recognized_screen(
                bot,
                timeout=min(screen_timeout, 30.0),
            )
    if initial_state == ScreenState.UNKNOWN:
        reporter.update(
            "failed",
            "启动前无法可靠识别游戏画面",
            error="preflight_screen_unknown",
            safety_state="blocked",
            calibration_valid=True,
        )
        shutdown_bot(bot)
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
        shutdown_bot(bot)
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
    preparation_failures = 0
    battle_recovery_failures = 0
    battle_observed_in_run = False

    def should_stop():
        return limits.reached(completed_rounds, started_at)

    def user_stop_requested():
        return limits.stop_requested()

    try:
        while not should_stop():
            if not wait_for_web_resume(limits, reporter, bot):
                break
            if not refresh_game_window(bot):
                reporter.update(
                    "recovering",
                    "游戏窗口暂不可用，正在重新查找",
                    current_round=completed_rounds + 1,
                    completed_rounds=completed_rounds,
                )
                time.sleep(1.0)
                continue
            current_round = completed_rounds + 1
            logger.info("=== 第 %s 局 ===", current_round)
            reporter.update(
                "preparing",
                "正在准备下一局",
                current_round=current_round,
                completed_rounds=completed_rounds,
            )
            # Never reuse the previous loop's workflow flag.  A pause,
            # capture retry, or a player action may have moved the game to a
            # different page.  Confirm the live scene first and only then
            # invoke the matching continuation path.
            current_scene = recover_current_scene(
                bot,
                attempts=5,
                stable_frames=2,
                poll_interval=0.2,
            )
            prepared = False
            resuming_this_battle = False
            result_ready = False
            if current_scene == ScreenState.BATTLE:
                prepared = True
                resuming_this_battle = True
                logger.info("当前已在战斗中：跳过选船，直接配置自动航行并接管")
            elif current_scene == ScreenState.LOADING:
                logger.info("当前处于加载中：等待 HUD 后进入战斗控制")
                prepared = wait_for_battle(
                    bot,
                    should_stop=should_stop,
                    require_new_round=True,
                    loading_already_seen=True,
                )
            elif current_scene == ScreenState.DAILY_REWARD:
                logger.info("当前处于每日奖励页：领取后回港并重新开始准备")
                backend = getattr(
                    getattr(bot, "distance_reader", None), "backend", None
                )
                if claim_daily_reward(
                    bot.hwnd,
                    backend=backend,
                    should_abort=lambda: operation_paused(bot),
                ):
                    time.sleep(1.0)
                return_to_port(bot, attempts=4)
                port_configured = False
                continue
            elif current_scene == ScreenState.RESULTS:
                if not battle_observed_in_run:
                    logger.info("启动时发现上次任务遗留的结算页：不计入本组，返回港口")
                    return_to_port(bot, attempts=3)
                    port_configured = False
                    continue
                logger.info("已确认本组战斗的结算页：保留页面，先执行收益 OCR")
                prepared = True
                result_ready = True
            elif current_scene == ScreenState.PORT:
                # In the port we must always validate the selected ship and
                # battle mode before joining.  Do not retain a stale success
                # flag from a prior round.
                configured_this_attempt = True
                battle_queued = prepare_battle(
                    bot,
                    should_stop=should_stop,
                    configure_port=configured_this_attempt,
                )
                if battle_queued and configured_this_attempt:
                    port_configured = True
                prepared = battle_queued and wait_for_battle(
                    bot,
                    should_stop=should_stop,
                    require_new_round=True,
                )
            else:
                logger.warning("当前场景仍未知，按全局规则尝试 Esc 返回港口")
                return_to_port(bot, attempts=3)
                port_configured = False
            if not prepared:
                if should_stop():
                    break
                intervention = getattr(bot, "intervention", None)
                preparation_paused = bool(
                    intervention is not None
                    and (
                        intervention.command_generation_paused()
                        or intervention.poll(bot.gamepad, time.monotonic())
                    )
                )
                if preparation_paused:
                    logger.info(
                        "[USER] 准备流程已暂停；不执行场景恢复、不切窗口，等待网页/键盘暂停结束"
                    )
                    if not wait_for_web_resume(limits, reporter, bot):
                        break
                    # Resume is always live-state based.  Do not continue from
                    # a half-finished carousel/mode-selector operation.
                    continue
                preparation_failures += 1
                recovered_state = recover_current_scene(bot)
                reporter.update(
                    "recovering",
                    "正在按当前场景恢复流程，"
                    f"重试 {preparation_failures}/5（{recovered_state.value}）",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                )
                if recovered_state == ScreenState.BATTLE:
                    logger.info("准备恢复确认已在战斗中，下一循环直接接管")
                    continue
                if recovered_state == ScreenState.LOADING:
                    logger.info("准备恢复确认正在加载，继续等待战斗 HUD")
                    try:
                        if wait_for_battle(
                            bot,
                            timeout=45.0,
                            should_stop=should_stop,
                            require_new_round=True,
                            loading_already_seen=True,
                        ):
                            logger.info("加载恢复已确认 HUD，下一循环直接进入战斗")
                            continue
                    except (SafetyFault, CaptureFault) as error:
                        logger.info("准备恢复等待 HUD 时画面仍不稳定: %s", error)
                elif recovered_state == ScreenState.RESULTS:
                    # Preserve this page.  The next lifecycle pass recognizes
                    # that this run already observed a battle and routes it to
                    # reward OCR before any navigation click.
                    logger.info("准备恢复已确认结算页，保留页面等待收益 OCR")
                    preparation_failures = 0
                    continue
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
            if not result_ready:
                battle_observed_in_run = True
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
                intervention_active = bool(
                    getattr(active_bot, "manual_intervention_active", False)
                )
                intervention_latched = bool(
                    getattr(active_bot, "manual_intervention_latched", False)
                )
                intervention_remaining = float(
                    getattr(
                        active_bot,
                        "manual_intervention_remaining_seconds",
                        0.0,
                    )
                )
                if intervention_latched:
                    progress_state = "paused"
                    progress_message = "用户持续操作已满20秒，永久暂停；等待网页点击继续"
                elif intervention_active:
                    progress_state = "paused"
                    progress_message = (
                        f"用户介入暂停；静默 {max(0, math.ceil(intervention_remaining))} 秒后自动恢复，"
                        "持续操作满20秒将永久暂停"
                    )
                else:
                    progress_state = "battle"
                    progress_message = (
                        "闭环控制已确认"
                        if active_bot.movement_verified
                        else "等待舰船位移反馈"
                    )
                reporter.update(
                    progress_state,
                    progress_message,
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
                    minimap_player=None
                    if analysis is None
                    else analysis.minimap_player_normalized,
                    minimap_heading=None
                    if analysis is None
                    else analysis.minimap_heading,
                    navigation_target=None
                    if analysis is None
                    else analysis.navigation_target_normalized,
                    capture_zone_center=None
                    if analysis is None
                    else analysis.capture_zone_center_normalized,
                    capture_zone_radius=None
                    if analysis is None
                    else analysis.capture_zone_radius_normalized,
                    capture_zone_label=""
                    if analysis is None
                    else analysis.capture_zone_label,
                    nearest_enemy=None
                    if analysis is None
                    else analysis.nearest_enemy_normalized,
                    minimap_enemy_count=0
                    if analysis is None
                    else analysis.minimap_enemy_count,
                    minimap_contacts=[]
                    if analysis is None
                    else analysis.minimap_contacts,
                    capture_zones=[]
                    if analysis is None
                    else analysis.capture_zones,
                    minimap_islands=[]
                    if analysis is None
                    else analysis.minimap_islands,
                    navigation_source="unknown"
                    if analysis is None
                    else analysis.navigation_source,
                    autopilot_enabled=False
                    if analysis is None
                    else analysis.autopilot_enabled,
                    rudder_indicator="neutral"
                    if analysis is None
                    else analysis.rudder_indicator,
                    commanded_rudder=None
                    if active_bot.last_movement_command is None
                    else active_bot.last_movement_command.rudder,
                    island_distance=None
                    if analysis is None
                    else analysis.island_distance,
                    health_percent=None
                    if analysis is None or not analysis.health_recognized
                    else round(analysis.health * 100.0, 1),
                    speed_knots=None
                    if analysis is None
                    else analysis.speed_knots,
                    on_fire=False if analysis is None else analysis.on_fire,
                    flooding=False if analysis is None else analysis.flooding,
                    damage_control_ready=bool(
                        getattr(active_bot, "damage_control_ready", False)
                    ),
                    heal_ready=bool(getattr(active_bot, "heal_ready", False)),
                    manual_intervention_active=bool(
                        getattr(active_bot, "manual_intervention_active", False)
                    ),
                    manual_intervention_latched=(
                        active_bot.manual_intervention_latched
                    ),
                    manual_intervention_seconds=(
                        active_bot.manual_intervention_seconds
                    ),
                    manual_intervention_remaining_seconds=(
                        active_bot.manual_intervention_remaining_seconds
                    ),
                    stop_after_current=bool(
                        limits.duration_seconds
                        and time.monotonic() - started_at
                        >= limits.duration_seconds
                    ),
                )

            battle_finished = False
            try:
                battle_finished = (
                    True
                    if result_ready
                    else run_battle(
                        bot,
                        # A time limit is a soft boundary: finish the active battle.
                        # Only an explicit user stop interrupts combat immediately.
                        should_stop=user_stop_requested,
                        progress=report_battle_progress,
                        resume_existing=resuming_this_battle,
                        quick_battle=limits.quick_battle,
                    )
                )
                if battle_finished == "resume_state":
                    if not wait_for_web_resume(
                        limits,
                        reporter,
                        bot,
                        resume_state="recovering",
                    ):
                        break
                    # The next lifecycle iteration reclassifies port/loading/
                    # battle/results before sending another command.
                    continue
            except (SafetyFault, CaptureFault) as error:
                battle_recovery_failures += 1
                logger.warning(
                    "战斗控制异常，先重新判断当前场景再续接 (%s/3): %s",
                    battle_recovery_failures,
                    error,
                )
                recovered_state = recover_after_battle_fault(
                    bot,
                    should_stop=user_stop_requested,
                )
                reporter.update(
                    "recovering",
                    "战斗控制异常，已重新识别当前场景并准备续接",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    error=str(error),
                    recovery_attempt=battle_recovery_failures,
                    recovered_scene=recovered_state.value,
                    safety_state="armed",
                )
                if recovered_state == ScreenState.BATTLE:
                    if battle_recovery_failures >= 3:
                        logger.error("战斗控制连续恢复失败，需要人工检查")
                        reporter.update(
                            "failed",
                            "战斗控制连续失败，请人工检查后从网页继续",
                            current_round=current_round,
                            completed_rounds=completed_rounds,
                            error="battle_recovery_retry_limit_reached",
                            safety_state="blocked",
                        )
                        return 2
                    logger.info("恢复场景仍为战斗，下一循环继续当前驾驶")
                    continue
                if recovered_state == ScreenState.RESULTS:
                    # The controller can fault during the transition out of a
                    # battle.  Preserve the result page for OCR; never click it
                    # away during recovery.
                    battle_finished = True
                    battle_recovery_failures = 0
                    logger.info("恢复场景已是结算页，直接进入收益统计")
                elif recovered_state == ScreenState.PORT:
                    battle_recovery_failures = 0
                    port_configured = False
                    logger.info("恢复场景已回港，本局不计数并重走港口准备")
                    continue
                elif recovered_state == ScreenState.LOADING:
                    logger.info("恢复场景仍在加载，交由下一轮加载/HUD检查续接")
                    continue
                else:
                    logger.error("反复识别后仍无法确认当前场景，需要人工介入")
                    reporter.update(
                        "failed",
                        "反复重试后仍无法识别当前页面，请人工检查后重试",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        error="battle_scene_unknown_after_retries",
                        safety_state="blocked",
                    )
                    return 2
            else:
                if battle_finished:
                    battle_recovery_failures = 0

            if battle_finished in QUICK_BATTLE_COMPLETION_REASONS:
                # A quick battle deliberately leaves before the ordinary
                # settlement lifecycle, so its plan progress cannot depend on
                # a results page.  Reaching this branch already requires a
                # positively identified battle HUD and one of three independent
                # completion signals: five-minute timeout, numeric HP=0, or a
                # stable natural battle-end surface.  Count the round for run
                # scheduling, but explicitly skip reward/history persistence.
                completed_rounds = count_quick_battle_for_plan(
                    completed_rounds,
                    battle_finished,
                )
                battle_observed_in_run = False
                reporter.update(
                    "returning",
                    "快速战斗已计入计划局数，正在 Esc 返回港口；本局不统计收益",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    rewards_status="skipped",
                    rewards_round=0,
                    last_rewards={},
                    last_outcome="unknown",
                )
                escape = getattr(bot.gamepad, "escape", None)
                if escape is not None:
                    escape()
                    time.sleep(0.8)
                return_to_port(bot, attempts=5)
                port_configured = False
                if should_stop():
                    reporter.update(
                        "completed",
                        "快速战斗计划已完成",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        rewards_status="skipped",
                        rewards_round=0,
                        last_rewards={},
                        movement_mode="idle",
                        movement_reason="计划已完成，控制已安全释放",
                        route_phase="unplanned",
                        route_progress=0.0,
                        route_waypoint=0,
                        route_arrived=False,
                        inside_capture_point=False,
                        paused_by_user=False,
                        manual_intervention_active=False,
                        manual_intervention_latched=False,
                    )
                    return 0
                continue
            if not battle_finished:
                if user_stop_requested():
                    reporter.update(
                        "stopped",
                        "已按用户要求安全停止",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        paused_by_user=False,
                        manual_intervention_active=False,
                        manual_intervention_latched=False,
                    )
                    return 0
                break
            if not wait_for_web_resume(
                limits,
                reporter,
                bot,
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
            result_confirmed, rewards, post_battle_state = collect_battle_rewards(
                bot,
                reward_reader,
            )
            if not result_confirmed:
                logger.warning(
                    "未连续确认结算页，本次不增加局数；当前界面=%s",
                    post_battle_state.value,
                )
                reporter.update(
                    "recovering",
                    "未确认结算页，本局不计数，正在恢复当前流程",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    rewards_status="pending",
                    last_rewards={},
                )
                if post_battle_state == ScreenState.BATTLE:
                    logger.info("战斗仍在进行，下一循环继续当前战斗")
                else:
                    logger.warning("异常/未知页面优先尝试返回港口")
                    return_to_port(bot)
                    port_configured = False
                continue
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
                    last_outcome=rewards.outcome,
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
                    last_outcome=rewards.outcome,
                )
            completed_rounds += 1
            battle_observed_in_run = False
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
                    paused_by_user=False,
                    manual_intervention_active=False,
                    manual_intervention_latched=False,
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
                bot,
                resume_state="requeueing",
            ):
                break
            if queue_next_battle(
                bot.hwnd,
                vision=bot.vision,
                should_abort=lambda: operation_paused(bot),
            ):
                if wait_for_battle(
                    bot,
                    should_stop=should_stop,
                    require_new_round=True,
                ):
                    continue
                if should_stop():
                    break
                logger.warning("已点击继续战斗，但未确认下一局 HUD；重新识别场景后重试")
                reporter.update(
                    "recovering",
                    "下一局状态未确认，正在重新识别场景并恢复",
                    current_round=current_round + 1,
                    completed_rounds=completed_rounds,
                    error="requeue_battle_not_confirmed",
                    safety_state="armed",
                )
                recovered_state = recover_current_scene(bot)
                if recovered_state == ScreenState.BATTLE:
                    logger.info("下一局 HUD 已确认，下一循环重新确认场景后接管")
                elif recovered_state == ScreenState.RESULTS:
                    return_to_port(bot, attempts=2)
                    port_configured = False
                elif recovered_state == ScreenState.PORT:
                    port_configured = False
                elif recovered_state == ScreenState.UNKNOWN:
                    logger.warning("下一局场景仍未知，按全局规则 Esc 返回港口")
                    return_to_port(bot, attempts=3)
                    port_configured = False
                continue
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
            paused_by_user=False,
            manual_intervention_active=False,
            manual_intervention_latched=False,
        )
        return 0
    except GameWindowUnavailableWhilePaused as error:
        logger.error("暂停任务安全退出: %s", error)
        # wait_for_web_resume has already written the precise failure state.
        # Returning lets the control server reconcile and release the stale run.
        return 2
    except KeyboardInterrupt:
        logger.info("用户中断")
        reporter.update(
            "stopped",
            "已停止",
            current_round=current_round,
            completed_rounds=completed_rounds,
            paused_by_user=False,
            manual_intervention_active=False,
            manual_intervention_latched=False,
        )
        return 0
    except ShipSelectionError as error:
        logger.error("自定义舰船选择失败: %s", error)
        reporter.update(
            "failed",
            "未找到指定舰船，请返回网页重新选择",
            current_round=current_round,
            completed_rounds=completed_rounds,
            error=str(error),
            safety_state="blocked",
        )
        return 2
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
        # A stale HWND must never receive focus-management/key-up input. The
        # shutdown helper still closes internal OCR and event resources.
        shutdown_bot(bot)
        logger.info("Bot 已停止")


if __name__ == "__main__":
    raise SystemExit(run())
