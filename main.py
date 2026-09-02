"""World of Warships bot entry point and high-level lifecycle."""

import ctypes
import logging
import math
import os
import time
from dataclasses import dataclass
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
    set_automation_input_observer,
    set_interaction_pause_guard,
    window_message_click,
)
from port_navigator import (
    claim_daily_reward,
    dismiss_port_exit_confirmation,
    enter_battle,
    ensure_selected_ship_commander,
    ensure_requested_mode,
    force_quick_battle_return_to_port,
    dismiss_battle_survey,
    handle_post_battle,
    in_battle_type_selector,
    is_early_exit_confirmation_page,
    is_battle_survey_page,
    is_port_exit_confirmation_page,
    queue_next_battle,
    is_daily_reward_page,
    select_mode_from_screen,
    select_requested_ship,
    ShipSelectionError,
)
from runtime_control import RunLimits, RuntimeReporter

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("runner")


@dataclass(frozen=True)
class ResumeGateResult:
    """Outcome of the cooperative pause gate.

    ``resumed`` is deliberately separate from ``allowed``.  A caller that
    actually waited for the player must discard its phase-local assumptions
    and return to live scene classification before issuing another action.
    """

    allowed: bool
    resumed: bool = False

    def __bool__(self) -> bool:
        return self.allowed


QUICK_BATTLE_COMPLETION_REASONS = frozenset(
    {"quick_timeout", "quick_death", "quick_ended"}
)


def count_quick_battle_for_plan(
    completed_rounds: int,
    battle_finished,
    *,
    closure_confirmed: bool = False,
) -> int:
    """Advance quick-battle progress only after leaving the active match.

    A five-minute/death signal requests an exit, but it is not itself proof
    that Esc actually returned to port.  Counting before that confirmation can
    count the same still-running battle again on the next lifecycle pass.
    """

    if closure_confirmed and battle_finished in QUICK_BATTLE_COMPLETION_REASONS:
        return max(0, int(completed_rounds)) + 1
    return max(0, int(completed_rounds))


def count_settled_battle_for_plan(
    completed_rounds: int,
    *,
    settlement_confirmed: bool,
) -> int:
    """Count a normal battle only after its result boundary was confirmed."""

    if settlement_confirmed:
        return max(0, int(completed_rounds)) + 1
    return max(0, int(completed_rounds))


def finalize_round_diagnostics(bot, round_number: int, *, outcome="unknown"):
    """Seal the matching frames/events/log and prune older round evidence."""
    finalize = getattr(bot, "complete_round_diagnostics", None)
    if finalize is None:
        return None
    try:
        return finalize(int(round_number), outcome=str(outcome or "unknown"))
    except Exception:
        logger.exception("第 %s 局诊断证据封存失败；保留现有文件", round_number)
        return None


def lifecycle_stop_requested(
    limits,
    completed_rounds: int,
    started_at: float,
    *,
    round_active: bool,
) -> bool:
    """Apply plan limits only between complete battle cycles.

    The Stop button is a hard interrupt. Round/time limits are soft while a
    matchmaking, loading, battle, or settlement cycle is active, so a pause
    cannot make the outer loop exit before that same round reaches results.
    """

    if limits.stop_requested():
        return True
    if round_active:
        return False
    return limits.schedule_reached(completed_rounds, started_at)


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


def close_game_window_after_plan(hwnd) -> bool:
    """Gracefully close only the verified game process window.

    This is called solely after a configured run reaches its plan target.  It
    never runs for Web stop, keyboard pause, recovery failure, or an unknown
    HWND, so the opt-in cannot close a browser/video window by title mistake.
    """
    try:
        if not hwnd or not is_game_window(int(hwnd)):
            logger.warning("完成后关闭游戏已勾选，但当前句柄不是已验证游戏窗口")
            return False
        posted = bool(
            ctypes.windll.user32.PostMessageW(
                int(hwnd),
                0x0010,  # WM_CLOSE
                0,
                0,
            )
        )
        if posted:
            logger.info("运行计划完成，已向《战舰世界》发送关闭窗口请求")
        else:
            logger.warning("运行计划完成，但关闭游戏窗口请求未被系统接受")
        return posted
    except (AttributeError, OSError, TypeError, ValueError):
        logger.exception("运行计划完成，但关闭游戏窗口失败")
        return False


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
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    # The visual exit-confirmation heuristic deliberately has broad colour
    # tolerance for different maps and UI scales. It is never sufficient to
    # authorize Esc on its own: a port animation can put a similarly solid
    # teal rectangle in the same region. Require exact dialog text first.
    if state == ScreenState.EXIT_CONFIRMATION:
        if is_port_exit_confirmation_page(image, backend):
            return ScreenState.PORT_EXIT_CONFIRMATION
        if is_early_exit_confirmation_page(image, backend):
            return state
        return ScreenState.UNKNOWN
    if state != ScreenState.UNKNOWN:
        return state
    try:
        if is_port_exit_confirmation_page(image, backend):
            return ScreenState.PORT_EXIT_CONFIRMATION
    except Exception:
        logger.debug("港口退出确认框 OCR 检查失败", exc_info=True)
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


def classify_battle_continuity_screen(bot, image) -> ScreenState:
    """Give a positive live HUD priority while a match is being entered/resumed.

    The generic classifier deliberately lets strong port controls win visual
    conflicts. During an already committed match that ordering is unsafe: a
    bright battle scene can imitate the port header and carousel for several
    frames. The minimap plus independent HUD anchors are stronger evidence in
    this phase, so only this phase-specific classifier may override PORT.
    """
    state = classify_runtime_screen(bot, image)
    if state != ScreenState.PORT:
        return state
    detector = getattr(bot.vision, "_has_battle_hud", None)
    if not callable(detector):
        return state
    try:
        battle_hud_visible = bool(detector(image))
    except Exception:
        logger.debug("战斗连续性 HUD 复核失败", exc_info=True)
        return state
    if battle_hud_visible:
        logger.info("战斗连续性复核：小地图与 HUD 锚点成立，覆盖本帧港口误判")
        return ScreenState.BATTLE
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
            # Do not focus or capture until the quiet period (or Web resume),
            # and do not spend the startup timeout while intentionally paused.
            paused_at = time.monotonic()
            time.sleep(0.15)
            deadline += max(0.0, time.monotonic() - paused_at)
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
            dismissed = dismiss_battle_survey(
                bot.hwnd,
                image,
                backend=backend,
                should_abort=lambda: operation_paused(bot),
                escape_action=getattr(getattr(bot, "gamepad", None), "escape", None),
            )
            if not dismissed:
                # Two consecutive positive observations already established
                # this exact overlay. Keep Esc as a bounded fallback when a
                # fresh OCR pass cannot reproduce the button glyphs.
                escape = getattr(getattr(bot, "gamepad", None), "escape", None)
                if escape is not None and not operation_paused(bot):
                    logger.info("评价页按钮复核未定位，按 Esc 跳过")
                    escape()
            survey_consecutive = 0
            previous_state = ScreenState.UNKNOWN
            consecutive = 0
            time.sleep(0.8)
            continue

        if last_state == ScreenState.PORT_EXIT_CONFIRMATION:
            # Esc in port opens this dialog and does not reliably close it.
            # Cancel only through the OCR-located ``否`` action.
            if operation_paused(bot):
                time.sleep(0.15)
                continue
            if dismiss_port_exit_confirmation(
                bot.hwnd,
                image,
                backend=backend,
                should_abort=lambda: operation_paused(bot),
            ):
                logger.info("启动恢复：已取消港口退出游戏确认框")
            else:
                logger.warning("港口退出确认框已识别，但“否”点击暂未派发")
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
    # Port and result pages can be legitimately pixel-identical for minutes.
    # OCR model warm-up may also take longer than FrameGuard's stale timeout.
    # Requiring visual motion here made a static result page fail preflight
    # before the lifecycle could click ``继续战斗``. Battle HUDs remain strict:
    # a motionless combat frame is still treated as a frozen capture backend.
    static_screen = screen_state in {ScreenState.PORT, ScreenState.RESULTS}
    verification_frame = bot.vision.grab(
        bot.hwnd,
        allow_stale=static_screen,
    )
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
    try:
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
    except CaptureFault as error:
        logger.info("加载检查画面暂不可用，交回生命周期重试: %s", error)
        return None
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
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        except CaptureFault as error:
            logger.info("加载等待画面暂不可用，交回生命周期重试: %s", error)
            return None
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
            confirm_action=getattr(bot.gamepad, "confirm", None),
            close_action=getattr(bot.gamepad, "escape", None),
        ):
            logger.info("每日奖励领取及关闭操作已派发，重新识别后继续港口流程")
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
            backend=getattr(getattr(bot, "distance_reader", None), "backend", None),
            should_abort=port_action_paused,
        )
        time.sleep(2)
        # Result and port screens can legitimately remain pixel-identical for
        # several seconds; freshness is only a combat safety requirement.
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        except CaptureFault as error:
            logger.info("结算返回后的画面暂不可用，稍后重新识别: %s", error)
            return False
        state = classify_battle_continuity_screen(bot, image)

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
    try:
        confirmation = bot.vision.grab(bot.hwnd, allow_stale=True)
    except CaptureFault as error:
        logger.info("港口复核画面暂不可用，本轮不执行选船: %s", error)
        return False
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
            ocr_backend=getattr(
                getattr(bot, "distance_reader", None), "backend", None
            ),
            should_abort=port_action_paused,
            require_port_action=True,
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
        backend=getattr(getattr(bot, "distance_reader", None), "backend", None),
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
    unknown_frames = 0
    last_clock = None
    clock_backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    opening_attempted = False
    new_round_state_reset = bool(require_new_round and loading_already_seen)
    if new_round_state_reset:
        setattr(bot, "_tactical_map_attempted_this_battle", False)
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
        state = classify_battle_continuity_screen(bot, image)
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
            unknown_frames = 0
            if require_new_round and not new_round_state_reset:
                # Loading is the definitive boundary between matches. Only
                # here may the next battle receive its one tactical-map setup.
                setattr(bot, "_tactical_map_attempted_this_battle", False)
                new_round_state_reset = True
            loading_seen = True
            battle_frames = 0
            clock_frames = 0
            continue
        if state == ScreenState.BATTLE:
            unknown_frames = 0
            if require_new_round and not loading_seen:
                battle_frames = 0
                clock_frames = 0
                continue
            if (
                require_new_round
                and not opening_attempted
                and hasattr(bot, "gamepad")
            ):
                find_minimap = getattr(bot.vision, "find_minimap", None)
                find_player_pose = getattr(
                    bot.vision,
                    "find_player_pose_on_minimap",
                    None,
                )
                opening_pose_ready = True
                if callable(find_minimap) and callable(find_player_pose):
                    try:
                        minimap = find_minimap(image)
                        opening_pose_ready = bool(
                            minimap is not None
                            and find_player_pose(minimap) is not None
                        )
                    except Exception:
                        opening_pose_ready = False
                        logger.debug(
                            "开局小地图玩家箭头预检失败，等待后续帧",
                            exc_info=True,
                        )
                if not opening_pose_ready:
                    # A broad HUD colour match during roster/loading must not
                    # consume this round's only M-map setup.  The real player
                    # arrow is the positive evidence that ship controls and
                    # the tactical map are ready to accept input.
                    logger.debug("战斗外观已出现但玩家箭头未就绪，暂缓开局输入")
                    battle_frames = 0
                    clock_frames = 0
                    continue
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
                setattr(bot, "_opening_autopilot_attempted", True)
                opening_configured = (
                    configure_opening_autopilot(bot)
                    if should_stop is None
                    else configure_opening_autopilot(
                        bot,
                        should_stop=should_stop,
                    )
                )
                if opening_configured:
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
        if state == ScreenState.UNKNOWN:
            unknown_frames += 1
            if unknown_frames >= 3:
                logger.warning(
                    "等待战斗时场景连续未知；交回生命周期识别，避免加载锁死"
                )
                return False
        elif state not in {ScreenState.LOADING, ScreenState.BATTLE}:
            unknown_frames = 0
        if state in {
            ScreenState.PORT,
            ScreenState.DAILY_REWARD,
            ScreenState.PORT_EXIT_CONFIRMATION,
        }:
            logger.info(
                "等待战斗时回到可处理页面 %s；交回生命周期分流",
                state.value,
            )
            return False
        # The loading boundary (or a caller that has just positively observed
        # the join/requeue transition) is the round identity. Battle-clock OCR
        # is useful telemetry but must not hold a visible HUD for three minutes
        # merely because tiny top-right text was unreadable at one UI scale.
        if battle_frames >= 2 and loading_seen:
            if require_new_round:
                if clock_frames >= 1:
                    logger.info(
                        "新一局证据已确认: 已经历加载/续局边界，战斗计时=%02d:%02d",
                        int(last_clock or 0) // 60,
                        int(last_clock or 0) % 60,
                    )
                else:
                    logger.info(
                        "新一局证据已确认: 已经历加载/续局边界且战斗 HUD 连续出现；"
                        "计时 OCR 暂不可用"
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
    dead_at_start = False
    abandoned_native_route = bool(
        resume_existing
        and getattr(bot, "native_autopilot_abandoned", False)
    )
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
    if not normalize_tactical_map_overlay(bot):
        logger.info("战术地图仍在安全收尾，暂不进入旧的战斗控制步骤")
        return "resume_state"
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
        if (
            classify_battle_continuity_screen(bot, control_frame)
            != ScreenState.BATTLE
        ):
            logger.warning("战斗动作互锁：最新画面已不是战斗，撤销驾驶并重新分流")
            return "resume_state"
        # A run can be restarted while the previous match is still in the
        # spectator/death HUD.  That screen retains enough battle anchors to
        # classify as BATTLE, but it is not a live ship and must never trigger
        # M-map/autopilot setup or a full-speed reassertion.  Probe the numeric
        # HP before any command; the regular combat tick will continue to wait
        # for the actual settlement boundary.
        health_reader = getattr(bot.vision, "read_health_fraction", None)
        if health_reader is not None:
            try:
                health_probe = health_reader(
                    control_frame,
                    getattr(getattr(bot, "distance_reader", None), "backend", None),
                )
                dead_at_start = health_probe is not None and float(health_probe) <= 0.0
            except Exception:
                logger.debug("战斗接管前生命值探测失败", exc_info=True)
        if dead_at_start:
            logger.info("接管前已确认舰船生命值为0；跳过全速重发和自动驾驶配置，等待结算")
        # The first HUD frame can appear before the game accepts movement
        # input. wait_for_battle may have sent W during that narrow transition
        # and advanced the controller cache even though the in-game telegraph
        # stayed at STOP. Reassert FULL at the actual control hand-off for both
        # fresh and resumed battles. An already-visible native route is the
        # only exception because W would cancel that valid autopilot.
        autopilot_visible = bool(
            getattr(bot, "opening_autopilot_active", False)
            and not abandoned_native_route
        )
        if not autopilot_visible and not dead_at_start:
            resynchronize = getattr(
                bot.gamepad,
                "resynchronize_forward_controls",
                None,
            )
            reassert = getattr(bot.gamepad, "reassert_full_speed", None)
            full_speed = getattr(bot.gamepad, "full_speed", None)
            if resynchronize is not None:
                resynchronize()
                resume_motion_reasserted = True
            elif reassert is not None:
                reassert()
                resume_motion_reasserted = True
            elif full_speed is not None:
                full_speed()
                resume_motion_reasserted = True
            if resume_motion_reasserted:
                logger.info("战斗 HUD 已确认，立即重发全速前进，再配置自动航线")
    preconfigured_autopilot = bool(
        getattr(bot, "_opening_autopilot_preconfigured", False)
    )
    opening_autopilot_attempted = bool(
        getattr(bot, "_opening_autopilot_attempted", False)
    )
    prestarted_motion = bool(
        getattr(bot, "_opening_motion_prestarted", False)
    )
    preconfigured_target = getattr(bot, "opening_autopilot_target", "")
    preconfigured_target_normalized = getattr(
        bot, "opening_autopilot_target_normalized", None
    )
    setattr(bot, "_opening_autopilot_preconfigured", False)
    setattr(bot, "_opening_autopilot_attempted", False)
    setattr(bot, "_opening_motion_prestarted", False)
    if resume_existing:
        # A focus/capture pause is not a round boundary. Full reset() starts a
        # new event stream and clears battle timers, consumable cooldowns,
        # tracking filters and route state. Preserve all of that until a
        # settlement page positively closes this match.
        logger.info("接续同一局战斗：保留局内计时、地图、航线与消耗品状态")
    else:
        try:
            bot.reset(
                preserve_movement=(preconfigured_autopilot or prestarted_motion),
                preserve_static_map=preconfigured_autopilot,
            )
        except TypeError:
            # Compatibility for small test doubles and third-party adapters.
            bot.reset()
        setattr(bot, "_round_control_initialized", True)
        if preconfigured_autopilot:
            enable = getattr(bot, "enable_opening_autopilot", None)
            if enable is not None:
                try:
                    enable(
                        preconfigured_target or "新一局预先设置的自动航线",
                        target_normalized=preconfigured_target_normalized,
                    )
                except TypeError:
                    enable(preconfigured_target or "新一局预先设置的自动航线")
    if abandoned_native_route:
        # ``reset(preserve_movement=True)`` rebuilds vision/route state during
        # same-battle recovery. Preserve the decision to ignore a stale green
        # autopilot HUD, otherwise each pause/fault can re-arm the route that
        # already stalled against terrain.
        setattr(bot, "native_autopilot_abandoned", True)
    autopilot_set = False
    if dead_at_start:
        autopilot_set = False
        logger.info("当前为沉船/观战阶段；不打开 M 地图，保持只读等待结算")
    elif resume_existing or preconfigured_autopilot:
        autopilot_set = bool(
            not abandoned_native_route
            and not getattr(bot, "native_autopilot_abandoned", False)
            and getattr(bot, "opening_autopilot_active", False)
        )
        if (
            not autopilot_set
            and not opening_autopilot_attempted
            and not abandoned_native_route
        ):
            # A recovered battle must use the same opening rule as a freshly
            # detected battle: establish native autopilot first, then let the
            # Q/E controller take over only after the game route ends.
            autopilot_set = (
                configure_opening_autopilot(bot)
                if should_stop is None
                else configure_opening_autopilot(
                    bot,
                    should_stop=should_stop,
                )
            )
    else:
        autopilot_set = (
            False
            if opening_autopilot_attempted
            else (
                configure_opening_autopilot(bot)
                if should_stop is None
                else configure_opening_autopilot(
                    bot,
                    should_stop=should_stop,
                )
            )
        )

    if not autopilot_set and intervention is not None and (
        intervention.command_generation_paused()
        or intervention.poll(bot.gamepad, time.monotonic())
    ):
        mark_pause = getattr(bot, "mark_manual_pause", None)
        if mark_pause is not None:
            mark_pause()
        logger.info("[USER] 自动航行配置期间用户介入；不启用后备驾驶，等待重新判定场景")
        return "resume_state"

    if not autopilot_set and not normalize_tactical_map_overlay(bot):
        logger.info("自动航行未完成且战术地图仍未收尾，交回场景路由")
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
    quick_started_at = (
        float(getattr(bot, "battle_start_time", time.monotonic()))
        if resume_existing
        else time.monotonic()
    )
    quick_deadline = (
        quick_started_at + max(30.0, float(quick_seconds))
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
            # The game can move from battle to death/results/port while the
            # controller is paused.  Never resume the old combat loop and wait
            # for several failed HUD frames: hand control back to the outer
            # lifecycle, which performs stable multi-frame scene routing before
            # any further command.
            logger.info(
                "[SYSTEM] 战斗暂停已解除；废弃暂停前战斗步骤并重新识别当前场景"
            )
            return "resume_state"
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
            # Never reopen the tactical map after the opening setup. Its green
            # indicator can disappear transiently and repeated M-map retries
            # both obscure the battle and create fragile overlay state. The
            # minimap controller already has live player/centre/island data, so
            # it is the deterministic recovery path for the rest of the match.
            if operation_paused(bot):
                continue
            setattr(bot, "autopilot_retry_pending", False)
            enable_center_route = getattr(
                bot, "enable_generic_center_route", None
            )
            if enable_center_route is not None:
                enable_center_route(
                    "原生自动航行结束，小地图闭环驾驶接管；本局不再打开M地图"
                )
            logger.info(
                "原生自动航行已结束；本局不重开M地图，下一控制帧由Q/E接管"
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


def acknowledge_automation_input(bot: BattleBot) -> None:
    """Tell intervention monitoring that the latest key came from the bot."""
    intervention = getattr(bot, "intervention", None)
    acknowledge = getattr(intervention, "acknowledge_automation", None)
    if acknowledge is not None:
        acknowledge(getattr(bot, "gamepad", None))


def tactical_map_is_open(bot: BattleBot, image) -> bool:
    """Confirm the tactical overlay from instructions or its large grid axes."""
    if image is None or image.size == 0:
        return False
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    if backend is None:
        return False
    height, width = image.shape[:2]
    regions = (
        image[
            int(height * 0.10) : int(height * 0.68),
            int(width * 0.55) : int(width * 0.99),
        ],
        image[
            int(height * 0.66) : int(height * 0.99),
            int(width * 0.02) : int(width * 0.72),
        ],
    )
    try:
        for region in regions:
            if region.size == 0:
                continue
            combined = "".join(
                "".join(str(token.text or "").split())
                for token in backend.recognize(region)
                if float(getattr(token, "confidence", 0.0)) >= 0.55
            )
            if (
                "自动驾驶控制" in combined
                or "离开战术地图模式" in combined
                or "设置自动航行" in combined
            ):
                return True

        # Some UI scales omit or blur the help paragraph. The full tactical
        # map still has A-J on its left axis and 1-10 on its top axis. Require
        # both axes near the expected centred 0.81H square so the small
        # bottom-right minimap cannot satisfy this fallback.
        tokens = list(backend.recognize(image) or [])
    except Exception:
        logger.debug("战术地图说明/网格 OCR 失败", exc_info=True)
        return False
    map_size = min(float(width), float(height)) * 0.81
    left = (float(width) - map_size) / 2.0
    top = (float(height) - map_size) / 2.0
    row_labels = set()
    column_labels = set()
    for token in tokens:
        if float(getattr(token, "confidence", 0.0)) < 0.60:
            continue
        text = "".join(str(getattr(token, "text", "") or "").split()).upper()
        box = getattr(token, "box", ()) or ()
        if len(box) < 3:
            continue
        center_x = sum(float(point[0]) for point in box) / len(box)
        center_y = sum(float(point[1]) for point in box) / len(box)
        if (
            text in tuple("ABCDEFGHIJ")
            and abs(center_x - left) <= map_size * 0.07
            and top <= center_y <= top + map_size
        ):
            row_labels.add(text)
        if (
            text in {str(value) for value in range(1, 11)}
            and abs(center_y - top) <= map_size * 0.07
            and left <= center_x <= left + map_size
        ):
            column_labels.add(text)
    return len(row_labels) >= 3 and len(column_labels) >= 3


def normalize_tactical_map_overlay(bot: BattleBot) -> bool:
    """Close a map left open by an interrupted waypoint operation exactly once."""
    if not bool(getattr(bot, "_tactical_map_left_open", False)):
        return True
    if operation_paused(bot):
        return False
    try:
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
    except (AttributeError, CaptureFault):
        return False
    if not tactical_map_is_open(bot, image):
        logger.info("战术地图残留标志与当前画面不符，已清除；不发送 M")
        setattr(bot, "_tactical_map_left_open", False)
        return True
    toggle_map = getattr(getattr(bot, "gamepad", None), "toggle_tactical_map", None)
    if toggle_map is None or operation_paused(bot):
        return False
    try:
        toggle_map()
        acknowledge_automation_input(bot)
    except RuntimeError as error:
        logger.info("战术地图收尾暂未派发，稍后按当前场景重试: %s", error)
        return False
    setattr(bot, "_tactical_map_left_open", False)
    logger.info("已关闭上次中断遗留的战术地图，再重新识别战斗 HUD")
    time.sleep(0.35)
    return True


def opening_autopilot_target(
    player_normalized: tuple[float, float],
    *,
    retrying: bool = False,
    attempt_index: int = 0,
) -> tuple[float, float]:
    """Choose a central waypoint shifted into the enemy half.

    The enemy side is the direction from spawn through map centre.  Keeping
    the waypoint a fixed distance beyond centre avoids both the old short
    centre click and far-edge routes that were prone to islands.
    """
    px, py = (float(player_normalized[0]), float(player_normalized[1]))
    dx, dy = 0.5 - px, 0.5 - py
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0.5, 0.5)
    attempt = max(0, min(2, int(attempt_index)))
    forward_offsets = (0.17, 0.20, 0.23)
    lateral_offsets = (0.0, 0.035, -0.035)
    forward = forward_offsets[attempt] + (0.02 if retrying else 0.0)
    lateral = lateral_offsets[attempt]
    unit_x, unit_y = dx / length, dy / length
    return (
        max(0.10, min(0.90, 0.5 + unit_x * forward - unit_y * lateral)),
        max(0.10, min(0.90, 0.5 + unit_y * forward + unit_x * lateral)),
    )


def _save_navigation_debug(bot: BattleBot, label: str, image) -> None:
    debug_dir = getattr(bot, "_debug_dir", None)
    save_debug = getattr(getattr(bot, "vision", None), "save_debug_frame", None)
    if debug_dir is None or save_debug is None or image is None:
        return
    try:
        save_debug(
            Path(debug_dir) / f"{label}_{time.strftime('%H%M%S')}.png",
            image,
        )
    except Exception:
        logger.debug("保存自动航行诊断截图失败", exc_info=True)


def configure_opening_autopilot(
    bot: BattleBot,
    *,
    retrying: bool = False,
    should_stop=None,
) -> bool:
    """Set and positively verify one native tactical-map waypoint."""
    toggle_map = getattr(bot.gamepad, "toggle_tactical_map", None)
    enable = getattr(bot, "enable_opening_autopilot", None)
    if toggle_map is None or enable is None or not hasattr(bot, "vision"):
        return False
    if should_stop is not None and should_stop():
        logger.info("终止请求已到达，取消开局自动航行配置")
        return False
    if operation_paused(bot):
        return False
    if not normalize_tactical_map_overlay(bot):
        return False
    try:
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
        if classify_battle_continuity_screen(bot, image) != ScreenState.BATTLE:
            return False
        height, width = image.shape[:2]
        player_normalized = None
        # A far-side target is meaningful only relative to the live white
        # player arrow.  Sample a few fresh frames because the marker can be
        # briefly covered by loading/friendly labels.  If it still cannot be
        # found, fail over to closed-loop minimap steering instead of clicking
        # the geometric centre and creating the short route reported by users.
        # The first HUD frame often arrives before the minimap ship marker is
        # rendered (especially after a port/roster transition).  Sampling for
        # a few seconds avoids falling back to Q/E just because the marker was
        # one frame late, while the surrounding battle interlock still aborts
        # immediately on a scene change or user pause.
        # Loading/roster transitions can leave the lower-right minimap
        # rendered before the white ship arrow. At 2K the real capture/OCR
        # cycle is ~0.6 s, so 28 samples cover roughly 15-18 s without ever
        # clicking a guessed centre point. This is still an opening-only wait;
        # no M-map retry is performed after the route has been handed off.
        pose_attempts = max(
            8,
            int(
                getattr(bot, "strategy", {}).get(
                    "opening_autopilot_pose_attempts", 28
                )
            ),
        )
        for pose_attempt in range(pose_attempts):
            if should_stop is not None and should_stop():
                logger.info("终止请求已到达，不再等待小地图玩家箭头")
                return False
            if operation_paused(bot):
                return False
            minimap = bot.vision.find_minimap(image)
            if minimap is not None:
                pose = bot.vision.find_player_pose_on_minimap(minimap)
                if pose is not None:
                    player_normalized = (
                        pose.position[0] / max(minimap.shape[1], 1),
                        pose.position[1] / max(minimap.shape[0], 1),
                    )
                    break
            if pose_attempt < pose_attempts - 1:
                time.sleep(0.25)
                image = bot.vision.grab(bot.hwnd, allow_stale=True)
                if (
                    classify_battle_continuity_screen(bot, image)
                    != ScreenState.BATTLE
                ):
                    return False
                height, width = image.shape[:2]
        if player_normalized is None:
            logger.warning(
                "连续%s帧未定位小地图白色玩家箭头；拒绝设置错误短航点，交由通用驾驶接管",
                pose_attempts,
            )
            return False
        if bool(getattr(bot, "_tactical_map_attempted_this_battle", False)):
            logger.info("本局已经完成三次战术地图落点尝试；不再重复打开 M 地图")
            return False

        rect = get_client_rect(bot.hwnd)
        intervention = getattr(bot, "intervention", None)
        begin_static_capture = getattr(bot, "begin_tactical_map_static_capture", None)
        static_sampler = getattr(bot, "capture_tactical_map_static_layer", None)
        autopilot_reader = getattr(bot.vision, "read_autopilot_enabled_text", None)
        backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
        static_capture_started = False
        static_complete = False

        for route_attempt in range(3):
            attempt_number = route_attempt + 1
            if should_stop is not None and should_stop():
                logger.info("终止请求已到达，取消剩余自动航行尝试")
                return False
            if operation_paused(bot):
                return False
            if route_attempt and not normalize_tactical_map_overlay(bot):
                logger.warning("第 %s/3 次自动驾驶重试前无法安全关闭 M 图", attempt_number)
                return False
            if intervention is not None and intervention.poll(bot.gamepad):
                mark_pause = getattr(bot, "mark_manual_pause", None)
                if mark_pause is not None:
                    mark_pause()
                logger.info("用户键盘介入，取消剩余自动航行尝试")
                return False

            normalized_target = opening_autopilot_target(
                player_normalized,
                retrying=retrying,
                attempt_index=route_attempt,
            )
            local_x, local_y = tactical_map_local_point(
                width,
                height,
                normalized_target,
            )
            toggle_map()
            acknowledge_automation_input(bot)
            # An accepted first M starts this battle's bounded three-attempt
            # sequence. Rejected keyboard dispatch raises before this flag.
            setattr(bot, "_tactical_map_attempted_this_battle", True)
            setattr(bot, "_tactical_map_left_open", True)
            time.sleep(0.90)

            tactical_static_frames = []
            open_confirmations = 0
            for sample_index in range(5):
                if should_stop is not None and should_stop():
                    return False
                if operation_paused(bot):
                    return False
                tactical_frame = bot.vision.grab(bot.hwnd, allow_stale=True)
                if tactical_map_is_open(bot, tactical_frame):
                    open_confirmations += 1
                tactical_static_frames.append(tactical_frame)
                if sample_index < 4:
                    time.sleep(0.12)
            if open_confirmations < 2:
                _save_navigation_debug(
                    bot,
                    f"tactical_open_unconfirmed_{attempt_number}",
                    tactical_static_frames[-1],
                )
                logger.warning(
                    "第 %s/3 次仅 %s/5 帧确认 M 图打开；不点击并更换偏移重试",
                    attempt_number,
                    open_confirmations,
                )
                continue

            if not static_capture_started and begin_static_capture is not None:
                begin_static_capture()
                static_capture_started = True

            if intervention is not None and intervention.poll(bot.gamepad):
                mark_pause = getattr(bot, "mark_manual_pause", None)
                if mark_pause is not None:
                    mark_pause()
                logger.info("用户键盘介入，战术地图停止落点并保留给统一恢复关闭")
                return False

            clicked = physical_click(
                rect["left"] + local_x,
                rect["top"] + local_y,
                extra_delay=0.1,
                hwnd=bot.hwnd,
            ) or window_message_click(
                bot.hwnd,
                rect["left"] + local_x,
                rect["top"] + local_y,
                extra_delay=0.1,
            )
            if not clicked:
                logger.warning("第 %s/3 次航点点击未派发；更换偏移重试", attempt_number)
                continue

            time.sleep(0.35)
            toggle_map()
            acknowledge_automation_input(bot)
            time.sleep(0.65)
            verification = bot.vision.grab(bot.hwnd, allow_stale=True)
            if tactical_map_is_open(bot, verification):
                setattr(bot, "_tactical_map_left_open", True)
                logger.warning("第 %s/3 次 M 图关闭未生效；安全收尾后重试", attempt_number)
                continue
            setattr(bot, "_tactical_map_left_open", False)
            if (
                classify_battle_continuity_screen(bot, verification)
                != ScreenState.BATTLE
            ):
                logger.warning("自动航行复核时已离开战斗，撤销剩余尝试")
                return False

            if static_sampler is not None and not static_complete:
                for tactical_frame in tactical_static_frames:
                    static_complete = (
                        bool(static_sampler(tactical_frame)) or static_complete
                    )

            # In production, each click succeeds only when the game's exact
            # lower-left green status confirms it. Test adapters without the
            # OCR reader retain compatibility with the public enable hook.
            confirmed = autopilot_reader is None or backend is None
            confirmation_frame = verification
            if not confirmed:
                for confirmation_attempt in range(4):
                    if should_stop is not None and should_stop():
                        return False
                    if operation_paused(bot):
                        return False
                    if autopilot_reader(confirmation_frame, backend):
                        confirmed = True
                        break
                    if confirmation_attempt < 3:
                        time.sleep(0.25)
                        confirmation_frame = bot.vision.grab(
                            bot.hwnd,
                            allow_stale=True,
                        )
            if not confirmed:
                _save_navigation_debug(
                    bot,
                    f"autopilot_unconfirmed_{attempt_number}",
                    confirmation_frame,
                )
                logger.warning(
                    "第 %s/3 次未确认“自动驾驶启用”；更换航点偏移重试",
                    attempt_number,
                )
                continue

            if static_sampler is not None and not static_complete:
                logger.warning("M 大地图静态层未完整确认；缺失项使用小地图多帧兜底")
            target_label = f"地图中心偏敌方航点（第{attempt_number}次）"
            try:
                enable(target_label, target_normalized=normalized_target)
            except TypeError:
                enable(target_label)
            logger.info(
                "[SYSTEM] 战术地图自动航行成功: %s | local=(%s,%s)",
                target_label,
                local_x,
                local_y,
            )
            return True

        if not normalize_tactical_map_overlay(bot):
            logger.warning("三次自动驾驶均失败且 M 图尚未安全收尾")
            return False
        logger.warning("自动驾驶三次尝试均未确认成功；切换 Q/E 驾驶兜底")
        return False
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
                acknowledge_automation_input(bot)
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
        try:
            if attempt == 0 and bot.last_analysis is not None:
                image = bot.last_analysis.image
            else:
                time.sleep(0.5)
                if operation_paused(bot):
                    return False, fallback, ScreenState.UNKNOWN
                image = bot.vision.grab(bot.hwnd, allow_stale=True)
            last_state = bot.vision.classify_screen(image)
        except CaptureFault as error:
            logger.info("结算读取画面暂不可用，保留结算流程并重试: %s", error)
            continue
        port_card_detector = getattr(reader, "_looks_like_port_reward_card", None)
        try:
            port_reward_card = bool(
                port_card_detector is not None and port_card_detector(image)
            )
        except Exception:
            logger.debug("港口收益卡片检测失败", exc_info=True)
            port_reward_card = False
        battle_hud_detector = getattr(bot.vision, "_has_battle_hud", None)
        try:
            battle_hud_visible = bool(
                callable(battle_hud_detector) and battle_hud_detector(image)
            )
        except Exception:
            logger.debug("结算页战斗 HUD 互锁检测失败", exc_info=True)
            battle_hud_visible = False
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
        try:
            rewards = reader.read(image)
        except Exception as error:
            # OCR/provider failures are data-quality failures, not lifecycle
            # failures. Keep the positively identified result page and retry
            # later frames; the round may still complete with unrecognized
            # rewards instead of stopping the whole automation run.
            logger.warning("结算收益 OCR 暂时失败，继续读取后续帧: %s", error)
            continue
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


def dismiss_battle_overlay(
    bot: BattleBot,
    initial_state: ScreenState,
    *,
    attempts: int = 3,
) -> ScreenState:
    """Dismiss only positively identified battle menus with the Esc key.

    Escape-menu and early-exit confirmation pages are overlays on a live
    battle, not failed port preparation.  Clicking guessed modal coordinates
    repeatedly can leave the same dialog open forever.  Esc is deterministic
    for both pages and is reissued only after a fresh frame confirms that one
    of those two overlays is still present.
    """
    state = initial_state
    overlay_states = {ScreenState.ESCAPE_MENU, ScreenState.EXIT_CONFIRMATION}
    for attempt in range(max(1, int(attempts))):
        if state not in overlay_states:
            return state
        if operation_paused(bot) or not ensure_capture_foreground(bot):
            return ScreenState.UNKNOWN
        escape = getattr(getattr(bot, "gamepad", None), "escape", None)
        if escape is None:
            return state
        try:
            logger.info(
                "已确认战斗覆盖层 %s，按 Esc 恢复战斗 (%s/%s)",
                state.value,
                attempt + 1,
                max(1, int(attempts)),
            )
            escape()
            acknowledge_automation_input(bot)
        except RuntimeError as error:
            logger.info("战斗覆盖层恢复键暂未派发: %s", error)
            return state
        time.sleep(0.45)
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
            state = classify_runtime_screen(bot, image)
        except CaptureFault:
            return ScreenState.UNKNOWN
    return state


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
        try:
            survey_open = is_battle_survey_page(image, backend=backend)
        except Exception:
            logger.debug("回港流程的战斗评价页面识别失败", exc_info=True)
            survey_open = False
        if survey_open:
            if operation_paused(bot):
                return False
            dismiss_battle_survey(
                bot.hwnd,
                image,
                backend=backend,
                should_abort=lambda: operation_paused(bot),
                escape_action=getattr(getattr(bot, "gamepad", None), "escape", None),
            )
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
                confirm_action=getattr(bot.gamepad, "confirm", None),
                close_action=getattr(bot.gamepad, "escape", None),
            ):
                logger.info("每日登录奖励已领取并关闭，继续确认港口")
                time.sleep(1.0)
                continue
            logger.warning("每日奖励页面已识别，但领取按钮未能安全点击")
            return False
        if state == ScreenState.PORT_EXIT_CONFIRMATION:
            if dismiss_port_exit_confirmation(
                bot.hwnd,
                image,
                backend=backend,
                should_abort=lambda: operation_paused(bot),
            ):
                logger.info("已取消港口退出游戏确认框，继续确认港口")
                time.sleep(0.8)
                continue
            logger.warning("港口退出确认框未能安全取消；保留页面重试")
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
                backend=backend,
                should_abort=lambda: operation_paused(bot),
            )
        elif state in {ScreenState.ESCAPE_MENU, ScreenState.EXIT_CONFIRMATION}:
            dismiss_battle_overlay(bot, state)
            return False
        else:
            # Esc in a port-looking transition opens the client's quit-game
            # confirmation. Unknown pages remain observation-only; every
            # supported overlay has its own positive OCR/visual gate.
            logger.info("未知页面缺少可验证操作，不发送 Esc；稍后重新识别")
        time.sleep(3)
    logger.warning("未能确认已返回港口；未执行盲点操作")
    return False


def recover_current_scene(
    bot: BattleBot,
    *,
    attempts: int = 6,
    stable_frames: int = 2,
    poll_interval: float = 0.25,
    round_in_progress: bool = False,
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
            state = (
                classify_battle_continuity_screen(bot, image)
                if round_in_progress
                else classify_runtime_screen(bot, image)
            )
            backend = getattr(
                getattr(bot, "distance_reader", None), "backend", None
            )
            if is_battle_survey_page(image, backend=backend):
                # The survey sits between battle/results and port. Preserve it
                # as an actionable overlay instead of letting the active-round
                # port lock collapse it into UNKNOWN forever.
                state = ScreenState.SURVEY
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

        if round_in_progress and state in {
            ScreenState.PORT,
            ScreenState.DAILY_REWARD,
        }:
            # A port-looking frame cannot close an active round. The game may
            # have transitioned or the bright battle HUD may have fooled the
            # broad port detector, but neither authorizes carousel clicks.
            # Only a confirmed settlement page clears the round lock.
            logger.warning(
                "本局尚未确认结算，忽略 %s 判定并禁止进入港口/选船流程",
                state.value,
            )
            state = ScreenState.UNKNOWN

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


def dismiss_current_battle_survey(bot: BattleBot) -> bool:
    """Revalidate and skip the currently visible post-battle survey."""
    if operation_paused(bot) or not ensure_capture_foreground(bot):
        return False
    try:
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
    except CaptureFault as error:
        logger.info("评价页关闭前暂时无法取得画面: %s", error)
        return False
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    return dismiss_battle_survey(
        bot.hwnd,
        image,
        backend=backend,
        should_abort=lambda: operation_paused(bot),
        escape_action=getattr(getattr(bot, "gamepad", None), "escape", None),
    )


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
        last_state = recover_current_scene(bot, round_in_progress=True)
        logger.info(
            "战斗故障后场景恢复 (%s/%s): %s",
            attempt,
            retry_count,
            last_state.value,
        )
        if last_state in {
            ScreenState.BATTLE,
            ScreenState.RESULTS,
            ScreenState.SURVEY,
            ScreenState.ESCAPE_MENU,
            ScreenState.EXIT_CONFIRMATION,
            ScreenState.PORT_EXIT_CONFIRMATION,
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
    # Every other unfamiliar page remains observation-only. Sending Esc on a
    # port transition opens the quit-game confirmation and creates a loop.
    if last_state == ScreenState.LOADING:
        return last_state
    logger.warning("本局尚未确认结算，场景连续未知；保持识别且不执行回港/选船操作")
    return ScreenState.UNKNOWN


def wait_for_web_resume(
    limits,
    reporter,
    bot=None,
    *,
    resume_state="preparing",
    poll_interval=0.15,
):
    """Freeze lifecycle actions and report whether old phase state is stale.

    Only an explicit Stop request closes this gate permanently. A missing or
    recreated game window remains recoverable: after Continue, foreground
    restoration keeps retrying until the window can be rebound, while callers
    always discard their old phase-local assumptions and classify the live
    screen again.
    """
    intervention = getattr(bot, "intervention", None)

    def keyboard_paused():
        return bool(
            intervention is not None
            and intervention.poll(getattr(bot, "gamepad", None))
        )

    web_paused = limits.pause_requested()
    key_paused = keyboard_paused()
    if not web_paused and not key_paused:
        return ResumeGateResult(allowed=True, resumed=False)
    initially_web_paused = web_paused
    trigger = str(getattr(intervention, "last_trigger", ""))
    source = (
        "网页手动暂停"
        if web_paused
        else "用户切屏"
        if trigger == "window_switch"
        else "用户键盘介入"
    )
    logger.info("[USER] %s；冻结自动流程并保留舰船操纵状态", source)
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
    window_missing_reported = False
    foreground_retry_reported = False
    latch_reported = bool(
        web_paused or (intervention is not None and intervention.latched)
    )
    while True:
        if limits.stop_requested():
            return ResumeGateResult(allowed=False, resumed=False)
        web_paused = limits.pause_requested()
        key_paused = keyboard_paused()
        if not web_paused and not key_paused:
            resumed_by_web = bool(
                initially_web_paused
                or (
                    intervention is not None
                    and getattr(intervention, "resumed_from_web", False)
                )
            )
            resume_source = (
                "网页点击继续" if resumed_by_web else "连续5秒无新的切屏或键盘操作"
            )
            if (
                bot is not None
                and intervention is not None
                and not restore_game_foreground_after_pause(bot, resume_source)
            ):
                # A new user event can land between the quiet-period poll and
                # SetForegroundWindow. A recreated game window can also make
                # one restore attempt fail. Neither condition terminates the
                # task: stay here until Stop or a verified foreground exists.
                web_paused = limits.pause_requested()
                key_paused = keyboard_paused()
                if web_paused or key_paused:
                    foreground_retry_reported = False
                    reporter.update(
                        "paused",
                        "检测到新的用户操作，继续等待5秒静默",
                        paused_by_user=True,
                        manual_intervention_latched=bool(
                            web_paused
                            or (
                                intervention is not None
                                and intervention.latched
                            )
                        ),
                        movement_mode="manual_pause",
                    )
                    time.sleep(max(0.01, float(poll_interval)))
                else:
                    if not foreground_retry_reported:
                        logger.warning(
                            "暂停已解除但游戏窗口暂不可用；保留任务并等待重新绑定"
                        )
                        reporter.update(
                            "recovering",
                            "暂停已解除，游戏窗口暂不可用；任务保持运行并等待恢复",
                            paused_by_user=False,
                            manual_intervention_latched=False,
                            movement_mode="idle",
                            movement_reason="等待重新找到游戏窗口后识别当前阶段",
                            error="game_window_rebind_pending",
                        )
                        foreground_retry_reported = True
                    time.sleep(max(0.25, float(poll_interval)))
                continue
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
        if bot is not None and not is_game_window_alive(getattr(bot, "hwnd", 0)):
            if not window_missing_reported:
                window_missing_reported = True
                logger.warning("暂停期间游戏窗口暂不可用；保留任务，继续后重新查找")
                reporter.update(
                    "paused",
                    "暂停期间游戏窗口暂不可用；任务已保留，继续后会重新查找",
                    paused_by_user=True,
                    manual_intervention_latched=bool(
                        web_paused
                        or (intervention is not None and intervention.latched)
                    ),
                    movement_mode="manual_pause",
                    movement_reason="暂停期间不切窗口、不发送指令",
                )
        else:
            window_missing_reported = False
        time.sleep(max(0.01, float(poll_interval)))
    logger.info("[SYSTEM] %s，开始重新识别当前画面并接续原流程", resume_source)
    reporter.update(
        resume_state,
        "暂停已解除，正在恢复游戏前台并判断当前状态",
        paused_by_user=False,
        manual_intervention_latched=False,
    )
    reporter.update(
        resume_state,
        "暂停前流程位置已作废，正在根据当前画面重新分流",
        paused_by_user=False,
        manual_intervention_latched=False,
    )
    return ResumeGateResult(allowed=True, resumed=True)


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
        launcher_client = (
            os.environ.get("WOWS_LAUNCHER_CLIENT", "steam").strip().lower()
            or "steam"
        )
        launcher_name = "WG Game Center" if launcher_client == "wgc" else "Steam"
        logger.info("游戏未运行，使用 %s 自动检索路径并启动", launcher_name)
        result = launch_game(client=launcher_client)
        if not result.started:
            logger.error("无法自动启动游戏: %s", result.detail)
            reporter.update(
                "failed",
                "无法自动启动游戏",
                error="game_launch_failed",
            )
            return 1
        logger.info("已请求自动启动游戏: %s (%s)", result.method, result.detail)
        reporter.update("launching_game", f"正在通过 {launcher_name} 启动战舰世界")
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
    set_automation_input_observer(
        lambda: bot.intervention.acknowledge_automation()
    )
    reward_reader = ResultRewardReader(bot.distance_reader.backend)
    # The pause monitor must exist before the first maximize/foreground action.
    # Web pause or keyboard intervention during Steam/login therefore leaves
    # the user's current page untouched until they resume.
    startup_resume_gate = wait_for_web_resume(
        limits,
        reporter,
        bot,
        resume_state="starting",
    )
    if not startup_resume_gate:
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
            confirm_action=getattr(bot.gamepad, "confirm", None),
            close_action=getattr(bot.gamepad, "escape", None),
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
            "启动时页面未知，正在安全重识别已知界面",
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
    while True:
        preflight_resume_gate = wait_for_web_resume(
            limits,
            reporter,
            bot,
            resume_state="preparing",
        )
        if not preflight_resume_gate:
            bot.stop(release_input=False)
            return 0
        if preflight_resume_gate.resumed:
            # The page may have moved from port to loading/battle/results while
            # paused. Never run a stale port preflight (which releases throttle)
            # against a live battle; classify the current page again first.
            reporter.update(
                "recovering",
                "启动自检暂停已解除，正在重新识别当前阶段",
            )
            initial_frame, initial_state = wait_for_recognized_screen(
                bot,
                timeout=min(screen_timeout, 30.0),
            )
            if initial_state == ScreenState.DAILY_REWARD:
                backend = getattr(
                    getattr(bot, "distance_reader", None), "backend", None
                )
                if claim_daily_reward(
                    bot.hwnd,
                    initial_frame,
                    backend=backend,
                    should_abort=lambda: operation_paused(bot),
                    confirm_action=getattr(bot.gamepad, "confirm", None),
                    close_action=getattr(bot.gamepad, "escape", None),
                ):
                    time.sleep(1.0)
                return_to_port(bot, attempts=4)
                # Re-enter the same pause-aware recognition path. No stale
                # DAILY_REWARD state is passed into the input preflight.
                continue
            if initial_state == ScreenState.UNKNOWN:
                if operation_paused(bot):
                    continue
                logger.warning("暂停恢复后页面仍未知；保持任务并继续识别")
                reporter.update(
                    "recovering",
                    "暂停恢复后页面暂未识别，任务保持运行并继续重试",
                    error="preflight_scene_retrying",
                )
                time.sleep(1.0)
                continue
        try:
            calibration = automatic_input_preflight(
                bot,
                title,
                rect,
                initial_state,
            )
            break
        except (SafetyFault, CaptureFault) as error:
            if operation_paused(bot) or limits.pause_requested():
                logger.info("[USER] 启动自检期间发生暂停；保留任务，恢复后重识别")
                continue
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
    # Once matchmaking/loading/battle has been committed, PORT is not allowed
    # to start selection again until settlement positively closes this round.
    round_in_progress = False
    round_entry_pending = False
    # A stable result page is the only boundary that permits a later PORT to
    # close the round. Keep this evidence across pauses because the player may
    # press Return to Port before reward OCR resumes.
    round_result_seen = False
    plan_completed = False

    def should_stop():
        return lifecycle_stop_requested(
            limits,
            completed_rounds,
            started_at,
            round_active=(
                round_in_progress
                or round_entry_pending
                or round_result_seen
            ),
        )

    def user_stop_requested():
        return limits.stop_requested()

    try:
        while not should_stop():
            lifecycle_resume_gate = wait_for_web_resume(limits, reporter, bot)
            if not lifecycle_resume_gate:
                break
            if lifecycle_resume_gate.resumed:
                logger.info(
                    "[SYSTEM] 已回到生命周期入口；重新判定当前画面，"
                    "未确认结算则继续同一局"
                )
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
            round_locked = (
                round_in_progress
                or round_entry_pending
                or round_result_seen
            )
            logger.info(
                "=== 第 %s 局%s ===",
                current_round,
                "（接续）" if round_locked else "",
            )
            reporter.update(
                "preparing",
                "正在恢复当前战斗" if round_locked else "正在准备下一局",
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
                # Before results, a port-looking frame can be a dangerous HUD
                # false positive. Once results were stably seen, PORT is a
                # valid continuation and may close this exact round once.
                round_in_progress=(round_locked and not round_result_seen),
            )
            if round_result_seen and current_scene in {
                ScreenState.LOADING,
                ScreenState.BATTLE,
            }:
                # The player may click Continue Battle while automation is
                # paused. A previously confirmed result followed by loading or
                # a battle HUD proves that exact old cycle closed and a new one
                # has begun. Commit the old round once and initialize the live
                # one as fresh control state.
                completed_rounds = count_settled_battle_for_plan(
                    completed_rounds,
                    settlement_confirmed=round_result_seen,
                )
                closed_round = current_round
                finalize_round_diagnostics(
                    bot,
                    closed_round,
                    outcome="unknown",
                )
                round_result_seen = False
                round_entry_pending = False
                round_in_progress = True
                setattr(bot, "_round_control_initialized", False)
                current_round = completed_rounds + 1
                logger.info(
                    "暂停期间已从结算进入新战斗；第 %s 局计数完成，接续第 %s 局",
                    closed_round,
                    current_round,
                )
                reporter.update(
                    "recovering",
                    "已确认上一局结算并进入新战斗；上一局已计数，正在接续当前局",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    rewards_status=(
                        "skipped" if limits.quick_battle else "unrecognized"
                    ),
                    rewards_round=(0 if limits.quick_battle else closed_round),
                    last_rewards={},
                    last_outcome="unknown",
                )
                if user_stop_requested() or limits.schedule_reached(
                    completed_rounds,
                    started_at,
                ):
                    manually_stopped = user_stop_requested()
                    plan_completed = not manually_stopped
                    reporter.update(
                        "stopped" if manually_stopped else "completed",
                        "已按用户要求安全停止"
                        if manually_stopped
                        else "运行计划已完成；不接管暂停期间新进入的战斗",
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
            prepared = False
            resuming_this_battle = False
            result_ready = False
            if current_scene == ScreenState.SURVEY:
                if dismiss_current_battle_survey(bot):
                    logger.info("战斗评价页已跳过；下一循环重新识别结算/港口")
                else:
                    logger.warning("战斗评价页暂未关闭；保持任务并重新识别")
                time.sleep(0.8)
                continue
            if current_scene == ScreenState.PORT_EXIT_CONFIRMATION:
                try:
                    image = bot.vision.grab(bot.hwnd, allow_stale=True)
                except CaptureFault as error:
                    logger.info("取消港口退出确认框前画面暂不可用: %s", error)
                    continue
                if dismiss_port_exit_confirmation(
                    bot.hwnd,
                    image,
                    backend=getattr(
                        getattr(bot, "distance_reader", None), "backend", None
                    ),
                    should_abort=lambda: operation_paused(bot),
                ):
                    logger.info("已取消港口退出游戏确认框；下一循环重新识别")
                else:
                    logger.warning("港口退出确认框暂未取消；保持任务并重试")
                time.sleep(0.8)
                continue
            if current_scene == ScreenState.BATTLE:
                if round_entry_pending:
                    round_entry_pending = False
                    round_in_progress = True
                prepared = True
                resuming_this_battle = bool(
                    getattr(bot, "_round_control_initialized", False)
                )
                logger.info(
                    "当前已在战斗中：跳过选船，%s",
                    "接续本局控制" if resuming_this_battle else "建立本局控制",
                )
            elif current_scene == ScreenState.LOADING:
                resuming_this_battle = bool(
                    getattr(bot, "_round_control_initialized", False)
                )
                logger.info(
                    "当前处于加载中：%s",
                    "等待同一局 HUD 恢复" if resuming_this_battle else "等待本局 HUD 出现",
                )
                prepared = wait_for_battle(
                    bot,
                    should_stop=should_stop,
                    require_new_round=not resuming_this_battle,
                    loading_already_seen=not resuming_this_battle,
                )
                # A loading-looking frame is not by itself round identity: a
                # dimmed port dialog briefly satisfies the same colour test.
                # Commit the lock only after two live HUD frames appear. A
                # pause preserves round_entry_pending, so resume still cannot
                # fall into ship selection during a real matchmaking load.
                if prepared:
                    round_entry_pending = False
                    round_in_progress = True
            elif current_scene == ScreenState.DAILY_REWARD:
                logger.info("当前处于每日奖励页：领取后回港并重新开始准备")
                backend = getattr(
                    getattr(bot, "distance_reader", None), "backend", None
                )
                if claim_daily_reward(
                    bot.hwnd,
                    backend=backend,
                    should_abort=lambda: operation_paused(bot),
                    confirm_action=getattr(bot.gamepad, "confirm", None),
                    close_action=getattr(bot.gamepad, "escape", None),
                ):
                    time.sleep(1.0)
                return_to_port(bot, attempts=4)
                port_configured = False
                continue
            elif current_scene in {
                ScreenState.ESCAPE_MENU,
                ScreenState.EXIT_CONFIRMATION,
            }:
                restored_state = dismiss_battle_overlay(bot, current_scene)
                logger.info(
                    "战斗覆盖层恢复结果: %s；下一循环重新识别",
                    restored_state.value,
                )
                preparation_failures = 0
                continue
            elif current_scene == ScreenState.RESULTS:
                if not (
                    round_in_progress
                    or round_entry_pending
                    or round_result_seen
                ):
                    round_entry_pending = False
                    logger.info("当前结算页不属于已进入的新战斗：不重复计数，返回港口")
                    return_to_port(bot, attempts=3)
                    port_configured = False
                    continue
                round_entry_pending = False
                round_in_progress = True
                round_result_seen = True
                logger.info("已确认本组战斗的结算页：保留页面，先执行收益 OCR")
                prepared = True
                result_ready = True
            elif current_scene == ScreenState.PORT:
                if round_result_seen and round_in_progress:
                    # Results were already stably observed before a pause or
                    # manual Return-to-Port action. The full cycle is closed,
                    # but reward numbers are no longer available. Count once,
                    # clear every round lock, then begin the next lifecycle.
                    completed_rounds = count_settled_battle_for_plan(
                        completed_rounds,
                        settlement_confirmed=round_result_seen,
                    )
                    finalize_round_diagnostics(
                        bot,
                        current_round,
                        outcome="unknown",
                    )
                    round_in_progress = False
                    round_entry_pending = False
                    round_result_seen = False
                    setattr(bot, "_round_control_initialized", False)
                    logger.info(
                        "已凭暂停前确认的结算证据在港口闭合本局，计入计划进度: %s",
                        completed_rounds,
                    )
                    reporter.update(
                        "returning",
                        "已确认本局结算并回到港口；本局已计数，收益记为未识别",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        rewards_status=(
                            "skipped" if limits.quick_battle else "unrecognized"
                        ),
                        rewards_round=(
                            0 if limits.quick_battle else current_round
                        ),
                        last_rewards={},
                        last_outcome="unknown",
                    )
                    if should_stop():
                        manually_stopped = user_stop_requested()
                        plan_completed = not manually_stopped
                        reporter.update(
                            "stopped" if manually_stopped else "completed",
                            "已按用户要求安全停止"
                            if manually_stopped
                            else "运行计划已完成",
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
                    continue
                # In the port we must always validate the selected ship and
                # battle mode before joining.  Do not retain a stale success
                # flag from a prior round.
                configured_this_attempt = True
                try:
                    battle_queued = prepare_battle(
                        bot,
                        should_stop=should_stop,
                        configure_port=configured_this_attempt,
                    )
                except CaptureFault as error:
                    logger.info("港口准备画面暂不可用，保留任务并重试: %s", error)
                    battle_queued = False
                if battle_queued and configured_this_attempt:
                    port_configured = True
                if battle_queued:
                    # From this moment until a confirmed settlement, any PORT
                    # classification is treated as unsafe. This specifically
                    # covers a user pause during wait_for_battle().
                    round_entry_pending = True
                prepared = battle_queued and wait_for_battle(
                    bot,
                    should_stop=should_stop,
                    require_new_round=True,
                    loading_already_seen=True,
                )
                if prepared:
                    round_entry_pending = False
                    round_in_progress = True
            else:
                if round_in_progress or round_entry_pending:
                    logger.warning(
                        "本局尚未确认结算且当前场景未知；保持识别，禁止回港和选船"
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
                    preparation_resume_gate = wait_for_web_resume(
                        limits, reporter, bot
                    )
                    if not preparation_resume_gate:
                        break
                    # Resume is always live-state based.  Do not continue from
                    # a half-finished carousel/mode-selector operation.
                    continue
                recovered_state = recover_current_scene(
                    bot,
                    round_in_progress=(round_in_progress or round_entry_pending),
                )
                if recovered_state == ScreenState.BATTLE:
                    round_entry_pending = False
                    round_in_progress = True
                    round_result_seen = False
                    logger.info("准备恢复确认已在战斗中，下一循环直接接管")
                    preparation_failures = 0
                    continue
                if recovered_state == ScreenState.LOADING:
                    round_entry_pending = False
                    round_in_progress = True
                    logger.info("准备恢复确认正在加载，继续等待战斗 HUD")
                    preparation_failures = 0
                    reporter.update(
                        "recovering",
                        "游戏仍在加载，保持任务运行并等待战斗 HUD",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                    )
                    try:
                        if wait_for_battle(
                            bot,
                            timeout=45.0,
                            should_stop=should_stop,
                            require_new_round=not bool(
                                getattr(bot, "_round_control_initialized", False)
                            ),
                            loading_already_seen=not bool(
                                getattr(bot, "_round_control_initialized", False)
                            ),
                        ):
                            logger.info("加载恢复已确认 HUD，下一循环直接进入战斗")
                            continue
                    except (SafetyFault, CaptureFault) as error:
                        logger.info("准备恢复等待 HUD 时画面仍不稳定: %s", error)
                    continue
                elif recovered_state == ScreenState.RESULTS:
                    # Preserve this page.  The next lifecycle pass recognizes
                    # that this run already observed a battle and routes it to
                    # reward OCR before any navigation click.
                    logger.info("准备恢复已确认结算页，保留页面等待收益 OCR")
                    preparation_failures = 0
                    continue
                elif recovered_state == ScreenState.SURVEY:
                    dismiss_current_battle_survey(bot)
                    preparation_failures = 0
                    time.sleep(0.8)
                    continue
                elif recovered_state in {
                    ScreenState.ESCAPE_MENU,
                    ScreenState.EXIT_CONFIRMATION,
                }:
                    dismiss_battle_overlay(bot, recovered_state)
                    preparation_failures = 0
                    continue
                preparation_failures = min(preparation_failures + 1, 5)
                reporter.update(
                    "recovering",
                    "正在按当前场景安全恢复流程，"
                    f"连续未就绪 {preparation_failures} 次（{recovered_state.value}）",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                )
                if preparation_failures >= 5:
                    logger.warning(
                        "连续 %s 次未完成准备；保持任务运行，仅继续识别，不盲目操作",
                        preparation_failures,
                    )
                    reporter.update(
                        "recovering",
                        "当前页面暂未恢复，任务保持运行并继续安全重试",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        error="prepare_recovery_continuing",
                    )
                time.sleep(min(2 * preparation_failures, 8))
                continue
            preparation_failures = 0
            port_configured = True
            if not result_ready:
                round_in_progress = True
                round_entry_pending = False
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
                    progress_message = "用户持续操作已满20秒，已锁定暂停；等待网页点击继续"
                elif intervention_active:
                    progress_state = "paused"
                    progress_message = (
                        f"用户介入暂停；静默 {max(0, math.ceil(intervention_remaining))} 秒后自动恢复，"
                        "持续操作满20秒将锁定暂停"
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
                    autopilot_confirmed=False
                    if analysis is None
                    else analysis.autopilot_confirmed,
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
                    other_consumables_ready=bool(
                        getattr(active_bot, "other_consumables_ready", False)
                    ),
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
                    "quick_ended"
                    if result_ready and limits.quick_battle
                    else True
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
                    battle_resume_gate = wait_for_web_resume(
                        limits,
                        reporter,
                        bot,
                        resume_state="recovering",
                    )
                    if not battle_resume_gate:
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
                        logger.warning(
                            "战斗控制已连续恢复 %s 次；保持任务运行并重新进入场景路由",
                            battle_recovery_failures,
                        )
                        reporter.update(
                            "recovering",
                            "战斗控制暂不稳定，已停止下发指令并继续重新识别",
                            current_round=current_round,
                            completed_rounds=completed_rounds,
                            error="battle_recovery_continuing",
                            safety_state="armed",
                        )
                        battle_recovery_failures = 2
                        time.sleep(2.0)
                    logger.info("恢复场景仍为战斗，下一循环继续当前驾驶")
                    continue
                if recovered_state == ScreenState.RESULTS:
                    # The controller can fault during the transition out of a
                    # battle.  Preserve the result page for OCR; never click it
                    # away during recovery.
                    battle_finished = (
                        "quick_ended" if limits.quick_battle else True
                    )
                    battle_recovery_failures = 0
                    logger.info("恢复场景已是结算页，直接进入收益统计")
                elif recovered_state == ScreenState.SURVEY:
                    dismiss_current_battle_survey(bot)
                    battle_recovery_failures = 0
                    logger.info("已处理战斗评价页，下一循环重新识别实际场景")
                    continue
                elif recovered_state == ScreenState.PORT:
                    battle_recovery_failures = 0
                    port_configured = False
                    logger.info("恢复场景已回港，本局不计数并重走港口准备")
                    continue
                elif recovered_state == ScreenState.LOADING:
                    logger.info("恢复场景仍在加载，交由下一轮加载/HUD检查续接")
                    continue
                elif recovered_state in {
                    ScreenState.ESCAPE_MENU,
                    ScreenState.EXIT_CONFIRMATION,
                }:
                    dismiss_battle_overlay(bot, recovered_state)
                    logger.info("已处理战斗覆盖层，下一循环重新识别并接管")
                    continue
                else:
                    logger.warning("反复识别后仍无法确认当前场景，保持安全重试")
                    reporter.update(
                        "recovering",
                        "当前页面仍无法确认，任务未停止，将继续安全识别",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        error="battle_scene_unknown_continuing",
                        safety_state="armed",
                    )
                    battle_recovery_failures = min(battle_recovery_failures, 2)
                    time.sleep(2.0)
                    continue
            else:
                if battle_finished:
                    battle_recovery_failures = 0

            if battle_finished in QUICK_BATTLE_COMPLETION_REASONS:
                # Quick mode has explicit closure paths and never enters reward
                # OCR: timeout/HP=0 -> two-step forced port exit; natural result
                # -> result-page continuation. Count only once a path positively
                # proves the old match is over.
                reporter.update(
                    "returning",
                    "快速战斗已触发退出，正在确认已离开本局；本局不统计收益",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    rewards_status="skipped",
                    rewards_round=0,
                    last_rewards={},
                    last_outcome="unknown",
                )
                escape = getattr(bot.gamepad, "escape", None)
                backend = getattr(
                    getattr(bot, "distance_reader", None), "backend", None
                )
                closure_scene = ScreenState.UNKNOWN
                if battle_finished in {"quick_timeout", "quick_death"}:
                    closure_scene = force_quick_battle_return_to_port(
                        bot.hwnd,
                        vision=bot.vision,
                        backend=backend,
                        open_menu=escape,
                        should_abort=lambda: operation_paused(bot),
                    )
                    # A match can naturally finish while the two-step exit is
                    # being handled. Timeout and sunk-ship semantics both
                    # require a positively confirmed port before the round is
                    # counted. Clicking the death dialog's "Continue Battle"
                    # merely returns to spectating the same match and must
                    # never be mistaken for a new round.
                    closure_confirmed = closure_scene == ScreenState.PORT
                    if closure_scene == ScreenState.RESULTS:
                        closure_confirmed = return_to_port(bot, attempts=5)
                        if closure_confirmed:
                            closure_scene = ScreenState.PORT
                    port_configured = False
                else:
                    closure_scene = (
                        ScreenState.RESULTS
                        if result_ready
                        else recover_current_scene(
                            bot,
                            attempts=24,
                            stable_frames=2,
                            poll_interval=0.25,
                        )
                    )
                    closure_confirmed = closure_scene in {
                        ScreenState.RESULTS,
                        ScreenState.PORT,
                    }
                if not closure_confirmed:
                    logger.warning(
                        "快速战斗尚未确认离开当前对局，本局不计数并重新判断场景"
                    )
                    reporter.update(
                        "recovering",
                        "尚未确认离开当前快速战斗，本局不计数；正在重新判断场景",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        rewards_status="skipped",
                        rewards_round=0,
                        last_rewards={},
                    )
                    continue
                completed_rounds = count_quick_battle_for_plan(
                    completed_rounds,
                    battle_finished,
                    closure_confirmed=True,
                )
                finalize_round_diagnostics(
                    bot,
                    current_round,
                    outcome="quick_battle",
                )
                round_in_progress = False
                round_entry_pending = False
                round_result_seen = False
                setattr(bot, "_round_control_initialized", False)
                logger.info(
                    "快速战斗已确认离开，本局计入计划进度: %s",
                    completed_rounds,
                )
                reporter.update(
                    "returning",
                    "快速战斗已确认结束并计入计划局数；本局不统计收益",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    rewards_status="skipped",
                    rewards_round=0,
                    last_rewards={},
                    last_outcome="unknown",
                )
                if should_stop():
                    manually_stopped = user_stop_requested()
                    plan_completed = not manually_stopped
                    reporter.update(
                        "stopped" if manually_stopped else "completed",
                        "已按用户要求安全停止"
                        if manually_stopped
                        else "快速战斗计划已完成",
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

                # Start the next match through the surface that actually
                # closed this one. Any failure falls back to live scene
                # recovery/port preparation without changing the count again.
                queued = False
                if closure_scene == ScreenState.RESULTS:
                    queued = queue_next_battle(
                        bot.hwnd,
                        vision=bot.vision,
                        backend=backend,
                        should_abort=lambda: operation_paused(bot),
                    )
                if queued:
                    round_entry_pending = True
                    reporter.update(
                        "requeueing",
                        "快速战斗不统计收益，已点击继续战斗进入下一局",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        rewards_status="skipped",
                    )
                    if wait_for_battle(
                        bot,
                        should_stop=should_stop,
                        require_new_round=True,
                        loading_already_seen=True,
                    ):
                        round_entry_pending = False
                        round_in_progress = True
                        continue
                    logger.warning("快速续局已点击，但新一局 HUD 尚未确认；重新识别")
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
                # A transient foreground/focus denial returns False without
                # ending the battle. Preserve the round lock and let the next
                # lifecycle pass classify battle/loading/results again.
                logger.warning(
                    "战斗控制本轮未启动，但未收到终止请求；保留当前局并重新识别"
                )
                reporter.update(
                    "recovering",
                    "战斗控制暂未接管，任务保持运行并重新识别当前阶段",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    error="battle_control_retrying",
                    safety_state="armed",
                )
                continue
            reward_resume_gate = wait_for_web_resume(
                limits,
                reporter,
                bot,
                resume_state="collecting_rewards",
            )
            if not reward_resume_gate:
                break
            if reward_resume_gate.resumed:
                logger.info(
                    "[SYSTEM] 收益统计前发生过暂停；不沿用旧结算状态，重新识别当前场景"
                )
                reporter.update(
                    "recovering",
                    "暂停期间页面可能变化，正在重新识别后恢复流程",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                )
                continue
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
                    round_result_seen = False
                    logger.info("战斗仍在进行，下一循环继续当前战斗")
                else:
                    logger.warning("异常/未知页面优先尝试返回港口")
                    return_to_port(bot)
                    port_configured = False
                continue
            # ``run_battle`` normally returns immediately after the HUD
            # disappears, before the outer lifecycle has had a chance to
            # classify the results page as ``RESULTS``.  Reward OCR is itself
            # gated by consecutive result-page evidence, so promote that
            # evidence to the round boundary here.  Without this hand-off a
            # normal battle was treated as an unclosed round (completed_rounds
            # stayed at zero) and the 1-round plan clicked "继续战斗" into a
            # second match.
            if not round_result_seen:
                round_result_seen = True
                logger.info("结算页已由收益 OCR 稳定确认；闭合当前战斗周期")
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
            completed_rounds = count_settled_battle_for_plan(
                completed_rounds,
                settlement_confirmed=(result_confirmed and round_result_seen),
            )
            finalize_round_diagnostics(
                bot,
                current_round,
                outcome=rewards.outcome,
            )
            round_in_progress = False
            round_entry_pending = False
            round_result_seen = False
            setattr(bot, "_round_control_initialized", False)
            if should_stop():
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
                plan_completed = True
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
            requeue_resume_gate = wait_for_web_resume(
                limits,
                reporter,
                bot,
                resume_state="requeueing",
            )
            if not requeue_resume_gate:
                break
            if requeue_resume_gate.resumed:
                logger.info(
                    "[SYSTEM] 续局前发生过暂停；不点击旧页面，重新识别当前场景"
                )
                reporter.update(
                    "recovering",
                    "暂停期间页面可能变化，正在重新识别后恢复流程",
                    current_round=current_round + 1,
                    completed_rounds=completed_rounds,
                )
                continue
            if queue_next_battle(
                bot.hwnd,
                vision=bot.vision,
                backend=reward_reader.backend,
                should_abort=lambda: operation_paused(bot),
            ):
                round_entry_pending = True
                if wait_for_battle(
                    bot,
                    should_stop=should_stop,
                    require_new_round=True,
                    loading_already_seen=True,
                ):
                    round_entry_pending = False
                    round_in_progress = True
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
                recovered_state = recover_current_scene(
                    bot,
                    round_in_progress=True,
                )
                if recovered_state == ScreenState.BATTLE:
                    round_entry_pending = False
                    round_in_progress = True
                    logger.info("下一局 HUD 已确认，下一循环重新确认场景后接管")
                elif recovered_state == ScreenState.RESULTS:
                    round_entry_pending = False
                    return_to_port(bot, attempts=2)
                    port_configured = False
                elif recovered_state == ScreenState.SURVEY:
                    dismiss_current_battle_survey(bot)
                elif recovered_state == ScreenState.UNKNOWN:
                    logger.warning("下一局场景仍未知；保持入局锁，禁止选船并继续识别")
                continue
            logger.warning("无法直接继续战斗，开始执行已验证的回港兜底")
            returned_to_port = return_to_port(bot, attempts=6)
            port_configured = False
            if returned_to_port:
                logger.info("回港兜底已确认；下一循环从港口常规入口继续战斗")
                reporter.update(
                    "preparing",
                    "继续战斗不可用，已回到港口；正在从常规入口续战",
                    current_round=current_round + 1,
                    completed_rounds=completed_rounds,
                )
            elif not operation_paused(bot):
                logger.warning("回港兜底尚未确认，保留任务并在下一循环重识别")
            time.sleep(2)
        manually_stopped = user_stop_requested()
        plan_completed = bool(
            not manually_stopped
            and limits.schedule_reached(completed_rounds, started_at)
        )
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
        if plan_completed and limits.close_game_when_done:
            close_game_window_after_plan(bot.hwnd)
        logger.info("Bot 已停止")


if __name__ == "__main__":
    raise SystemExit(run())
