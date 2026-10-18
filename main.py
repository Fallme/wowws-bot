"""World of Warships bot entry point and high-level lifecycle."""

import ctypes
import logging
import math
import os
import time
from runtime_control import shutdown_after_plan
from dataclasses import dataclass, replace
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
from core.blockage_diagnostics import BlockageTrace
from core.overlay_hints import escape_close_hint
from core.launcher import launch_game, last_launch_client
from core.results import BattleRewards, ResultRewardReader
from core.tactical_geometry import TacticalMapGeometry, detect_tactical_map_geometry
from core.ui import ScreenState
from core.battle_phase import BattlePhase, BattlePhaseTracker
from core.window import (
    activate_window,
    ensure_game_window_foreground,
    find_game_window,
    get_client_rect,
    is_game_window,
    is_usable_game_window,
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
    quick_death_leave_hint_visible,
    is_battle_survey_page,
    is_port_exit_confirmation_page,
    click_loading_start_action,
    port_battle_action_point,
    queue_next_battle,
    is_daily_reward_page,
    select_mode_from_screen,
    select_requested_ship,
    ShipSelectionError,
)
from runtime_control import RunLimits, RuntimeReporter

BASE_DIR = Path(__file__).resolve().parent
BLOCKAGE_ROOT = Path("E:/aimemo/docs/screenshots/wows_blockages")
logger = logging.getLogger("runner")


def blockage_trace(bot, label):
    """Keep one evidence sequence across repeated calls for the same blocker."""
    traces = getattr(bot, "_blockage_traces", None)
    if traces is None:
        traces = bot._blockage_traces = {}
    if label not in traces:
        traces[label] = BlockageTrace(
            getattr(bot, "blockage_diagnostic_root", None), label,
            lossless=label == "result_rewards",
        )
    return traces[label]


def finish_blockage_trace(bot, label, image, state, *, attempt=0, outcome="resolved"):
    traces = getattr(bot, "_blockage_traces", {})
    trace = traces.get(label)
    if trace is not None and trace.folder is not None:
        trace.record(image, state=state, action=outcome, attempt=attempt, phase="result")
    if outcome == "resolved" and trace is not None and trace.folder is not None:
        traces.pop(label, None)


def recovery_escape_hint(bot, image, state):
    """Authorize generic recovery only for a visible corner close hint."""
    if state not in {ScreenState.UNKNOWN, ScreenState.LOADING}:
        return None
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    if backend is None or image is None or not getattr(image, "size", 0):
        return None
    try:
        if loading_start_action_visible(bot, image):
            return None
        numeric_hud = getattr(bot.vision, "has_numeric_battle_hud", None)
        if callable(numeric_hud) and numeric_hud(image, backend):
            return None
        return escape_close_hint(image, backend)
    except Exception:
        logger.debug("关闭提示复核失败；不派发 Esc", exc_info=True)
        return None


def dismiss_unknown_overlay(bot, *, attempts=3):
    """Close an unrecognized overlay without navigating away from a match.

    Each Esc needs a fresh unknown frame with a corner Esc close hint. Known
    battle, loading, results and port pages go back to the lifecycle untouched. The budget is shared over
    repeated recovery calls so a frozen capture cannot cause endless keys.
    """
    trace = blockage_trace(bot, "unknown_overlay")
    used = int(getattr(bot, "_unknown_overlay_escape_attempts", 0))
    pending = None
    state = ScreenState.UNKNOWN
    for attempt in range(max(1, int(attempts)) + 1):
        if operation_paused(bot) or not ensure_capture_foreground(bot):
            return ScreenState.UNKNOWN
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        except CaptureFault:
            return ScreenState.UNKNOWN
        state = classify_runtime_screen(bot, image)
        if pending is not None:
            trace.record(image, state=state, action="escape", attempt=pending, phase="after")
            pending = None
        if state != ScreenState.UNKNOWN:
            bot._unknown_overlay_escape_attempts = 0
            finish_blockage_trace(bot, "unknown_overlay", image, state, attempt=used)
            return state
        trace.record(image, state=state, action="unrecognized", attempt=used)
        hint = recovery_escape_hint(bot, image, state)
        if hint is None:
            trace.record(image, state=state, action="escape_not_authorized",
                         attempt=used, phase="result",
                         details={"reason": "no_verified_corner_close_hint"})
            logger.info("未知画面未确认角落 Esc 关闭提示，保持观察，不发送 Esc")
            return state
        if attempt >= max(1, int(attempts)) or used >= 3:
            finish_blockage_trace(bot, "unknown_overlay", image, state,
                                  attempt=used, outcome="escape_budget_exhausted")
            return state
        escape = getattr(getattr(bot, "gamepad", None), "escape", None)
        if escape is None or operation_paused(bot):
            return state
        trace.record(image, state=state, action="escape", attempt=used + 1, phase="before",
                     details={"close_hint": hint})
        if operation_paused(bot):
            return state
        try:
            escape()
        except RuntimeError as error:
            logger.info("未知遮挡 Esc 暂未派发: %s", error)
            return state
        used += 1
        bot._unknown_overlay_escape_attempts = used
        pending = used
        acknowledge_automation_input(bot)
        logger.warning("未知遮挡，按 Esc 关闭并重新识别 (%s/3)", used)
        time.sleep(0.8)
    return state


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
    """Count a quick round after death requeue, results, or port is verified.

    A raw zero-HP or timeout observation alone is not enough: the previous
    round must have an observed transition before its counter advances.
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
    # A confirmed closure ends any setup deferred earlier in this match.
    # Otherwise its remaining budget could leak into the next loading phase.
    setattr(bot, "_opening_autopilot_retry_pending", False)
    setattr(bot, "_opening_autopilot_budget_active", False)
    setattr(bot, "_opening_autopilot_attempted", False)
    setattr(bot, "_opening_autopilot_preconfigured", False)
    setattr(bot, "_opening_motion_prestarted", False)
    reset_acquisition = getattr(bot, "_reset_torpedo_acquisition", None)
    if callable(reset_acquisition):
        reset_acquisition()
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


def wait_for_game_window(timeout: float = 60.0, poll_interval: float = 1.0, should_stop=None,
                         launch_process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            return None
        code = launch_process.poll() if launch_process is not None else None
        if isinstance(code, int):
            logger.error("等待游戏窗口期间，本体已退出(%s)，停止等待", code)
            return None
        windows = find_game_window()
        if windows:
            code = launch_process.poll() if launch_process is not None else None
            if isinstance(code, int):
                logger.error("发现窗口时，本体已退出(%s)，不接受过期窗口", code)
                return None
            return windows[0]
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    return None


def refresh_game_window(bot: BattleBot, *, maximize: bool = True) -> bool:
    """Reacquire a recreated game window without sending any game input."""
    if operation_paused(bot):
        logger.info("[USER] 暂停期间不重新绑定、不最大化游戏窗口")
        return False
    current = int(getattr(bot, "hwnd", 0) or 0)
    windows = find_game_window()
    if not windows:
        if is_game_window(current):
            return True
        logger.warning("原游戏窗口已失效，暂未发现新的战舰世界窗口")
        return False
    hwnd, title, _rect = windows[0]
    if is_game_window(current) and int(hwnd) == current:
        if maximize and not is_usable_game_window(current):
            maximize_game_window(current)
        return True
    if is_game_window(current) and int(hwnd) != current:
        logger.info(
            "发现更大的战舰世界渲染窗口，重新绑定: %s -> %s",
            current,
            hwnd,
        )
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


def operation_stopped(bot) -> bool:
    limits = getattr(bot, "run_limits", None)
    return bool(limits is not None and limits.stop_requested())


def operation_paused(bot: BattleBot) -> bool:
    """Poll the keyboard/Web pause before any capture, focus or game action."""
    if operation_stopped(bot):
        return True
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
    # Recognition and foreground loops can pause without returning to the
    # lifecycle reporter. Publish transitions here so Continue stays available.
    reporter = getattr(bot, "runtime_reporter", None)
    if reporter is not None:
        status = reporter.status
        latched = paused and bool(getattr(intervention, "latched", False))
        if (
            bool(status.manual_intervention_active) != paused
            or bool(status.manual_intervention_latched) != latched
        ):
            reporter.update(
                "paused" if paused else "recovering",
                "检测到用户操作，等待静默或点击继续"
                if paused else "暂停已解除，正在接续当前流程",
                manual_intervention_active=paused,
                manual_intervention_latched=latched,
                paused_by_user=paused,
            )
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


# The pre-battle roster page renders its own countdown and enters the live HUD
# by itself; 2026-09-05..09-11 logs show thirteen successful native-route setups
# on days when that page was never clicked at all.  Its explicit action is
# therefore a rescue for a page that has visibly stopped advancing, never a
# routine per-frame click: pressing it on every confirmed frame produced an
# unbounded physical-click storm (85 clicks in one day) and the setup success
# rate fell from 81% of attempts to 10%.  Wait out a full countdown plus a wide
# margin, and give a whole page episode exactly one bounded attempt.
LOADING_START_RESCUE_DELAY = 60.0
LOADING_START_RESCUE_ATTEMPTS = 1
# A detector that flickers between frames must not re-arm the bounded rescue.
# The budget returns only after the confirmed page has really stayed away.
LOADING_START_EPISODE_QUIET = 10.0


def loading_start_action_visible(bot, image) -> bool:
    """Detect the pre-battle start action without trusting broad page colour."""
    detector = getattr(getattr(bot, "vision", None), "_has_loading_start_action", None)
    if not callable(detector):
        return False
    try:
        return bool(detector(image))
    except Exception:
        logger.debug("加载页开始按钮识别失败", exc_info=True)
        return False


def loading_start_confirmed(bot, image) -> bool:
    """Confirm the action text; missing OCR is not positive button evidence."""
    if not loading_start_action_visible(bot, image):
        return False
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    if backend is None:
        return False
    try:
        height, width = image.shape[:2]
        box_reader = getattr(bot.vision, "loading_start_action_box", None)
        box = box_reader(image) if callable(box_reader) else None
        if box is not None:
            x1, y1, x2, y2 = box
            pad = max(2, round((y2 - y1) * .15))
            action = image[max(0, y1 - pad):min(height, y2 + pad),
                           max(0, x1 - pad):min(width, x2 + pad)]
        else:
            action = image[int(height * .91):, int(width * .43):int(width * .57)]
        words = "".join(str(token.text or "") for token in (backend.recognize(action) or ()))
        return "开始战斗" in "".join(words.split())
    except Exception:
        logger.debug("加载按钮文字复核失败", exc_info=True)
        return False


def battle_overlay_text_confirmed(image, backend, state) -> bool:
    """Verify an actual menu/dialog in the centre, never the death HUD hint."""
    if backend is None or image is None or not getattr(image, "size", 0):
        return False
    try:
        height, width = image.shape[:2]
        centre = image[int(height * .25):int(height * .85),
                       int(width * .30):int(width * .70)]
        words = {
            "".join(str(token.text or "").split()).strip("?!？！，。:：")
            for token in (backend.recognize(centre) or ())
            if float(getattr(token, "confidence", 0)) >= .75
        }
        if state == ScreenState.EXIT_CONFIRMATION:
            return (
                "确认" in words and "是" in words and "否" in words
                and any(word == "离开战斗" for word in words)
                and any(word.startswith("继续战斗") for word in words)
            )
        if state == ScreenState.ESCAPE_MENU:
            return "菜单" in words and bool(words & {"返回游戏", "回到游戏", "继续游戏", "返回战斗"})
    except Exception:
        logger.debug("战斗菜单/离场框文字复核失败，不发送恢复键", exc_info=True)
    return False


def classify_runtime_screen(bot, image) -> ScreenState:
    """Classify normal game pages plus OCR-confirmed first-login rewards."""
    state = bot.vision.classify_screen(image)
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    # Some loading-tip artwork resembles result buttons. The exact current
    # start-action text is stronger proof that no battle has finished yet.
    if state == ScreenState.RESULTS and loading_start_confirmed(bot, image):
        return ScreenState.LOADING
    # Economy pages hide Join Battle but retain the dense ship carousel. Their
    # artwork passes the broad HUD texture check; confirm the page heading
    # before allowing any opening W/M commands. Do not classify it as PORT:
    # ship selection must wait until recovery closes this subpage.
    carousel = getattr(bot.vision, "_is_port_ship_bar", None)
    if state in {ScreenState.BATTLE, ScreenState.LOADING} and callable(carousel) and carousel(image):
        if backend is None:
            return ScreenState.UNKNOWN
        try:
            height, width = image.shape[:2]
            heading = image[:int(height * 0.12), int(width * 0.35):int(width * 0.65)]
            text = "".join(token.text for token in backend.recognize(heading))
            if "经济加成" in text.replace(" ", ""):
                return ScreenState.UNKNOWN
        except Exception:
            logger.debug("疑似港口子页面 OCR 失败，暂缓战斗指令", exc_info=True)
            return ScreenState.UNKNOWN
    # The visual exit-confirmation heuristic deliberately has broad colour
    # Teal ocean/consumable pixels can resemble the prebattle action bar.
    # With a competing HUD, require its text instead of endlessly recovering
    # an active match as LOADING. OCR failure remains fail-closed.
    if state == ScreenState.LOADING and backend is not None:
        # A sunk ship loses its consumable anchors; bright sea can then look
        # like loading. Its own death banner plus leave-battle hint still
        # positively identify this match. A real new-round start action wins.
        if not loading_start_confirmed(bot, image):
            try:
                if quick_death_leave_hint_visible(image, backend):
                    return ScreenState.BATTLE
            except Exception:
                logger.debug("加载色彩冲突下观战提示复核失败", exc_info=True)
        hud = getattr(bot.vision, "_has_battle_hud", None)
        start = getattr(bot.vision, "_has_loading_start_action", None)
        if callable(hud) and callable(start) and hud(image) and start(image):
            if not loading_start_confirmed(bot, image):
                return ScreenState.BATTLE
    # The visual exit-confirmation heuristic deliberately has broad colour
    # tolerance for different maps and UI scales. It is never sufficient to
    # authorize Esc on its own: a port animation can put a similarly solid
    # teal rectangle in the same region. Require exact dialog text first.
    if state == ScreenState.EXIT_CONFIRMATION:
        if is_port_exit_confirmation_page(image, backend):
            return ScreenState.PORT_EXIT_CONFIRMATION
        if is_early_exit_confirmation_page(image, backend):
            return state
        if battle_overlay_text_confirmed(image, backend, ScreenState.EXIT_CONFIRMATION):
            return ScreenState.EXIT_CONFIRMATION
        # Ocean artwork can satisfy the blue modal heuristic. Keep checking
        # for the positive sunk-ship HUD instead of immediately returning UNKNOWN.
        state = ScreenState.UNKNOWN
    if state != ScreenState.UNKNOWN:
        return state
    if battle_overlay_text_confirmed(image, backend, ScreenState.EXIT_CONFIRMATION):
        return ScreenState.EXIT_CONFIRMATION
    # A sunk ship's spectating HUD loses the usual health/consumable anchors.
    # It still belongs to the current battle: the bottom "Esc 离开战斗" hint
    # is not a dismissible overlay and must never enter generic Esc recovery.
    if backend is not None:
        try:
            if quick_death_leave_hint_visible(image, backend):
                return ScreenState.BATTLE
        except Exception:
            logger.debug("观战画面复核失败，保留当前判定", exc_info=True)
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
    # The pre-battle roster/start page can satisfy the generic reward-card and
    # HUD heuristics at the same time. Its explicit start action is a hard
    # boundary: never let it become BATTLE merely because a minimap-shaped
    # decoration is visible. Do not apply this override to RESULTS/SURVEY;
    # those pages can contain similarly coloured action buttons of their own.
    if state in {
        ScreenState.PORT,
        ScreenState.BATTLE,
        ScreenState.LOADING,
        ScreenState.UNKNOWN,
    }:
        # Economy/ship-subpage pages are intentionally UNKNOWN and may expose
        # a teal button that the loading detector mistakes for “开始战斗”.
        # The runtime classifier already performed the heading check; retain
        # that safe UNKNOWN result instead of converting it back to LOADING.
        carousel = getattr(getattr(bot, "vision", None), "_is_port_ship_bar", None)
        if state == ScreenState.UNKNOWN and callable(carousel):
            try:
                if carousel(image):
                    return state
            except Exception:
                logger.debug("战斗连续性港口子页面复核失败", exc_info=True)
        if loading_start_confirmed(bot, image):
            logger.info("检测到战斗开始按钮，保持加载态，暂不发送驾驶指令")
            return ScreenState.LOADING
    if state not in {ScreenState.PORT, ScreenState.LOADING}:
        return state
    # The ordinary HUD detector deliberately vetoes PORT, so asking it again
    # cannot resolve a false port match. Require independent numeric HUD proof
    # before overriding that veto in this already-committed battle workflow.
    numeric_hud = getattr(bot.vision, "has_numeric_battle_hud", None)
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    if callable(numeric_hud) and backend is not None:
        try:
            if numeric_hud(image, backend):
                logger.info("战斗连续性复核：本船血量、比赛计时与 HUD 成立，覆盖%s误判", state.value)
                return ScreenState.BATTLE
        except Exception:
            logger.debug("战斗连续性数值 HUD 复核失败，保留当前判定", exc_info=True)
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


def observe_battle_phase(bot, image, state):
    tracker = getattr(bot, "_battle_phase_tracker", None)
    if tracker is None:
        tracker = bot._battle_phase_tracker = BattlePhaseTracker()
    clock = None
    if state == ScreenState.BATTLE:
        backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
        reader = getattr(bot.vision, "read_battle_clock_seconds", None)
        if callable(reader) and backend is not None:
            try:
                clock = reader(image, backend)
            except Exception:
                logger.debug("战斗阶段计时 OCR 暂不可用", exc_info=True)
    active_evidence = bool(getattr(bot, "_round_control_initialized", False))
    if state == ScreenState.BATTLE and clock is not None and clock <= 60 and not active_evidence:
        try:
            speed_reader = getattr(bot.vision, "read_speed_knots", None)
            hp_reader = getattr(bot.vision, "read_health_fraction", None)
            speed = speed_reader(image, backend) if callable(speed_reader) else None
            health = hp_reader(image, backend) if callable(hp_reader) else None
            active_evidence = (speed is not None and abs(speed) > 2) or (health is not None and health < 0.99)
        except Exception:
            logger.debug("短计时战斗证据暂不可用", exc_info=True)
    phase = tracker.observe(state, clock, time.monotonic(), active_evidence=active_evidence)
    if phase != getattr(bot, "_observed_battle_phase", None):
        logger.info("战斗阶段: %s | 比赛计时=%s", phase.value, clock)
        bot._observed_battle_phase = phase
        reporter = getattr(bot, "runtime_reporter", None)
        if reporter is not None:
            reporter.update(reporter.status.state, reporter.status.message, battle_phase=phase.value)
    return phase


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
        if operation_stopped(bot):
            return None, ScreenState.UNKNOWN
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
    if screen_state not in {ScreenState.PORT, ScreenState.BATTLE, ScreenState.RESULTS,
                            ScreenState.DAILY_REWARD}:
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
    elif screen_state == ScreenState.DAILY_REWARD:
        # Startup recognizes this overlay; let the lifecycle claim/close it.
        # Movement key probes are unnecessary on a reward dialog.
        input_check = "reward_controls_preserved"
    else:
        bot.gamepad.stop()
        input_check = "safe_release_dispatched"
    # Port and result pages can be legitimately pixel-identical for minutes.
    # OCR model warm-up may also take longer than FrameGuard's stale timeout.
    # Requiring visual motion here made a static result page fail preflight
    # before the lifecycle could click ``继续战斗``. Battle HUDs remain strict:
    # a motionless combat frame is still treated as a frozen capture backend.
    static_screen = screen_state in {ScreenState.PORT, ScreenState.RESULTS,
                                    ScreenState.DAILY_REWARD}
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


def timed_port_action(label, action, *args, **kwargs):
    """Report measured operation duration on the operation's own log line."""
    started = time.perf_counter()
    outcome = "失败"
    try:
        result = action(*args, **kwargs)
        outcome = "完成" if result else "未完成"
        return result
    finally:
        logger.info("%s%s   %.0fms", label, outcome,
                    (time.perf_counter() - started) * 1000)


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
        return_to_port(bot, attempts=2, post_battle_context=True)
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

    configure_port = configure_port or bool(getattr(bot, "_port_recheck_required", False))
    logger.info("已连续确认港口，准备复核配置并进入战斗")
    if configure_port:
        ship_key = os.environ.get("WOWS_SHIP", "pommern")
        mode = os.environ.get("WOWS_MODE", "asymmetric")
        if not timed_port_action("目标舰船复核与选择", select_requested_ship,
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
        if not timed_port_action("指挥官复核", ensure_selected_ship_commander,
            bot.hwnd,
            ship_key,
            custom_name=os.environ.get("WOWS_CUSTOM_SHIP_NAME", ""),
            backend=getattr(getattr(bot, "distance_reader", None), "backend", None),
            should_abort=port_action_paused,
        ):
            logger.warning("目标舰船指挥官状态未完成复核或召回；等待重新识别")
            return False
        if not timed_port_action("战斗模式复核与选择", ensure_requested_mode,
            bot.hwnd,
            mode,
            vision=bot.vision,
            backend=getattr(getattr(bot, "distance_reader", None), "backend", None),
            should_abort=port_action_paused,
        ):
            logger.warning("未能安全选择目标战斗模式")
            return False
    if not timed_port_action("加入战斗", enter_battle,
        bot.hwnd,
        vision=bot.vision,
        configure_port=False,
        backend=getattr(getattr(bot, "distance_reader", None), "backend", None),
        should_abort=port_action_paused,
    ):
        logger.warning("“加入战斗”请求未能派发或未通过港口复核")
        return False
    bot._port_recheck_required = False
    # Return immediately. The lifecycle observer must see the loading screen;
    # sleeping here used to miss that transition and later mistake the current
    # battle for a new port workflow.
    time.sleep(0.1)
    return True


def reset_opening_autopilot_attempts(bot):
    """A confirmed new match gets its own budget; an interrupted setup keeps it."""
    if getattr(bot, "_opening_autopilot_budget_active", False):
        return
    setattr(bot, "_opening_autopilot_budget_active", True)
    setattr(bot, "_tactical_map_attempted_this_battle", False)
    if not getattr(bot, "_opening_autopilot_retry_pending", False):
        setattr(bot, "_tactical_map_attempts_used", 0)
        setattr(bot, "_tactical_map_prefer_message_click", False)
        setattr(bot, "_opening_autopilot_pending_target", None)


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
    loading_start_dwell_since = None
    loading_start_quiet_since = None
    loading_start_clicks = 0
    new_round_state_reset = bool(require_new_round and loading_already_seen)
    if new_round_state_reset:
        observe_battle_phase(bot, None, ScreenState.LOADING)
        reset_opening_autopilot_attempts(bot)
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
            observe_battle_phase(bot, image, state)
            unknown_frames = 0
            if require_new_round and not new_round_state_reset:
                # Loading is the definitive boundary between matches. Only
                # here may the next battle receive its one tactical-map setup.
                reset_opening_autopilot_attempts(bot)
                new_round_state_reset = True
            loading_seen = True
            battle_frames = 0
            clock_frames = 0
            # Never treat this page as a routine action item.  It carries its
            # own countdown and reaches the live HUD unaided, so the confirmed
            # action is only a rescue for a page that has clearly stopped
            # advancing.  The strict gate is mandatory here: the loose green-bar
            # test also fires on port sub-pages that have no start action at
            # all, and a stray physical click there is not recoverable.
            if loading_start_confirmed(bot, image):
                now = time.monotonic()
                loading_start_quiet_since = None
                if loading_start_dwell_since is None:
                    loading_start_dwell_since = now
                    logger.info(
                        "检测到已确认的“开始战斗”页面；该页自带倒计时，"
                        "先等待其自行推进（兜底确认最早在 %s 秒后）",
                        int(LOADING_START_RESCUE_DELAY),
                    )
                if (
                    hasattr(bot, "gamepad")
                    and loading_start_clicks < LOADING_START_RESCUE_ATTEMPTS
                    and now - loading_start_dwell_since >= LOADING_START_RESCUE_DELAY
                ):
                    loading_start_clicks += 1
                    dwell = now - loading_start_dwell_since
                    try:
                        clicked = click_loading_start_action(
                            bot.hwnd,
                            image=image,
                            vision=bot.vision,
                            backend=getattr(getattr(bot, "distance_reader", None), "backend", None),
                            should_abort=lambda: operation_paused(bot),
                        )
                    except (CaptureFault, SafetyFault, RuntimeError) as error:
                        clicked = False
                        logger.warning("“开始战斗”兜底确认动作异常: %s", error)
                    if clicked:
                        logger.warning(
                            "“开始战斗”页面已连续确认 %s 秒仍未自行推进，"
                            "已派发一次兜底确认 (%s/%s)；本次停留不再重复点击",
                            int(dwell),
                            loading_start_clicks,
                            LOADING_START_RESCUE_ATTEMPTS,
                        )
            else:
                # One episode is one continuous confirmed dwell.  The bounded
                # budget may only be re-armed after the page has really stayed
                # away, so a detector that flickers between frames can no
                # longer turn the attempt cap into unlimited clicking.
                loading_start_dwell_since = None
                if loading_start_clicks:
                    now = time.monotonic()
                    if loading_start_quiet_since is None:
                        loading_start_quiet_since = now
                    elif (
                        now - loading_start_quiet_since
                        >= LOADING_START_EPISODE_QUIET
                    ):
                        loading_start_clicks = 0
                        loading_start_quiet_since = None
                        logger.info("“开始战斗”页面已离开，兜底确认额度已重置")
            # This branch deliberately never returns False.  Bailing out only
            # because the pre-battle page persisted discarded a confirmed,
            # already in-flight transition: the caller then re-identified the
            # scene, re-entered the same page and pushed the round onto the
            # resume path, where the opening native route is never rebuilt.
            continue
        if state == ScreenState.BATTLE:
            unknown_frames = 0
            if callable(getattr(bot.vision, "read_battle_clock_seconds", None)):
                if observe_battle_phase(bot, image, state) != BattlePhase.ACTIVE:
                    battle_frames = 0
                    clock_frames = 0
                    continue
            if require_new_round and not loading_seen:
                battle_frames = 0
                clock_frames = 0
                continue
            if loading_start_confirmed(bot, image):
                # The roster/start-battle page can retain minimap-shaped and
                # HUD-shaped decoration. It is still a loading phase: W/M/Esc
                # are all forbidden until that action disappears on a fresh
                # frame.
                battle_frames = 0
                clock_frames = 0
                logger.debug("战斗外观仍带加载页开始按钮，暂不发送开局指令")
                continue
            battle_frames += 1
            if (
                require_new_round
                and not opening_attempted
                and hasattr(bot, "gamepad")
                and battle_frames >= 2
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
                    # Pose availability is not the battle identity. Give the
                    # marker a short grace period, then hand a stable HUD to
                    # the battle controller without consuming its M-map setup.
                    # The loading/start-action guard above remains mandatory.
                    if battle_frames < 4:
                        logger.debug("战斗 HUD 已出现，短暂等待玩家箭头，不重置场景确认")
                        continue
                    logger.warning(
                        "战斗 HUD 已连续4帧确认，但玩家箭头暂不可用；"
                        "结束入局等待，由战斗控制继续定位，不消耗 M 图尝试"
                    )
                    return True
                # Start movement on the first reliable post-loading HUD frame.
                # Clock OCR can take several seconds on its first GPU call, so
                # start the engine immediately, then establish native
                # autopilot while lifecycle identity is confirmed in the
                # background.  A failed map click must never leave a ship at
                # STOP while the clock/OCR checks continue.
                opening_attempted = True
                reassert = getattr(bot.gamepad, "reassert_full_speed", None)
                try:
                    pending_setup = bool(getattr(bot, "_opening_autopilot_retry_pending", False))
                    if pending_setup:
                        logger.info("自动航行配置待恢复：先复核已有航线，避免全速指令取消航线")
                    elif reassert is not None:
                        reassert()
                    else:
                        bot.gamepad.full_speed()
                    if not pending_setup:
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
                elif not getattr(bot, "_opening_autopilot_attempted", True):
                    opening_attempted = False
                    battle_frames = 0
                    clock_frames = 0
                    continue
                # Configuring the map may take seconds. Never confirm entry
                # using the old pre-configuration frame/clock.
                # A focus change can occur during the M-map click.  In that
                # case the capture backend cannot read the game, but the
                # round is still alive: return to the lifecycle pause gate.
                if operation_paused(bot):
                    logger.info("[USER] 自动航行复核期间已暂停；保留本局并等待恢复")
                    return False
                try:
                    image = bot.vision.grab(bot.hwnd, allow_stale=True)
                except CaptureFault as error:
                    logger.info("自动航行后画面暂不可用，保留任务重新识别: %s", error)
                    battle_frames = 0
                    clock_frames = 0
                    continue
                if classify_battle_continuity_screen(bot, image) != ScreenState.BATTLE:
                    battle_frames = 0
                    clock_frames = 0
                    continue
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
        # The loading boundary identifies the round; the independent temporal
        # phase gate above proves that the HUD is active, not preparation.
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
    bot.runtime_screen_classifier = lambda image: classify_battle_continuity_screen(bot, image)
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
        control_state = classify_battle_continuity_screen(bot, control_frame)
        if control_state != ScreenState.BATTLE:
            logger.warning("战斗动作互锁：最新画面已不是战斗，撤销驾驶并重新分流")
            bot._battle_phase_unconfirmed_since = None
            return "resume_state"
        if loading_start_confirmed(bot, control_frame):
            logger.warning("战斗动作互锁：最新画面仍带开始战斗按钮，撤销驾驶并重新分流")
            bot._battle_phase_unconfirmed_since = None
            return "resume_state"
        if callable(getattr(bot.vision, "read_battle_clock_seconds", None)):
            phase = observe_battle_phase(bot, control_frame, control_state)
            if phase == BattlePhase.UNKNOWN:
                # A port close-up / hub page can satisfy the broad battle-HUD
                # anchors while carrying no match clock. Waiting forever keeps
                # re-entering this same branch. After a bounded observation,
                # use the verified corner Esc hint (numeric-HUD veto included)
                # to back out instead of looping.
                now_monotonic = time.monotonic()
                since = getattr(bot, "_battle_phase_unconfirmed_since", None)
                if since is None:
                    bot._battle_phase_unconfirmed_since = since = now_monotonic
                wait_seconds = max(8.0, float(
                    getattr(bot, "strategy", {}).get(
                        "battle_phase_unconfirmed_escape_seconds", 12.0,
                    )
                ))
                if now_monotonic - since >= wait_seconds:
                    attempts = int(getattr(bot, "_battle_phase_escape_attempts", 0))
                    hint = None
                    if attempts < 3 and not operation_paused(bot):
                        hint = recovery_escape_hint(
                            bot, control_frame, ScreenState.UNKNOWN,
                        )
                    escape = getattr(bot.gamepad, "escape", None)
                    if hint is not None and escape is not None:
                        logger.warning(
                            "战斗 HUD 已出现但 %.0f 秒无法确认开战，画面带角落 Esc "
                            "关闭提示；发送 Esc 退回上一页",
                            now_monotonic - since,
                        )
                        escape()
                        acknowledge_automation_input(bot)
                        bot._battle_phase_escape_attempts = attempts + 1
                        bot._battle_phase_unconfirmed_since = None
                        return "resume_state"
                    logger.info(
                        "HUD 已出现但正式开战未确认，保留本局且不发送驾驶指令（Esc %s/3）",
                        attempts,
                    )
                else:
                    logger.info("HUD 已出现但正式开战未确认，保留本局且不发送驾驶指令")
                return "resume_state"
            if phase == BattlePhase.ACTIVE:
                bot._battle_phase_unconfirmed_since = None
                bot._battle_phase_escape_attempts = 0
            else:
                # PREPARING is the legitimate pre-battle countdown.
                bot._battle_phase_unconfirmed_since = None
                logger.info("HUD 已出现但正式开战未确认，保留本局且不发送驾驶指令")
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
        if ((resume_existing or getattr(bot, "_opening_autopilot_retry_pending", False)) and not dead_at_start
                and not getattr(bot, "opening_autopilot_active", False)
                and not getattr(bot, "native_autopilot_abandoned", False)):
            autopilot_reader = getattr(bot.vision, "read_autopilot_enabled_text", None)
            if callable(autopilot_reader):
                try:
                    native_route_visible = autopilot_reader(
                        control_frame,
                        getattr(getattr(bot, "distance_reader", None), "backend", None),
                    )
                except Exception:
                    logger.debug("接续时读取原生自动驾驶 HUD 失败", exc_info=True)
                    native_route_visible = None
                if native_route_visible is True:
                    bot.opening_autopilot_active = True
                    bot._native_autopilot_confirmed = True
                    bot._tactical_map_attempted_this_battle = True
                    bot.opening_autopilot_target = "接续前已存在的游戏内自动航线"
                    if getattr(bot, "_opening_autopilot_retry_pending", False):
                        bot.opening_autopilot_target_normalized = getattr(
                            bot, "_opening_autopilot_pending_target", None)
                        bot._opening_autopilot_preconfigured = True
                        bot._opening_autopilot_retry_pending = False
                    logger.info("接续时确认游戏内自动驾驶仍启用；不发送 W 或覆盖原航线")
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
            # Resuming must update the helm model as well as the key cache.
            resynchronize = (
                getattr(bot, "_resynchronize_forward_controls", None)
                if resume_existing else None
            ) or getattr(
                bot.gamepad,
                "resynchronize_forward_controls",
                None,
            )
            reassert = getattr(bot.gamepad, "reassert_full_speed", None)
            full_speed = getattr(bot.gamepad, "full_speed", None)
            if operation_paused(bot) or not ensure_bound_game_foreground(bot):
                return "resume_state"
            try:
                if resynchronize is not None:
                    resynchronize()
                    resume_motion_reasserted = True
                elif reassert is not None:
                    reassert()
                    resume_motion_reasserted = True
                elif full_speed is not None:
                    full_speed()
                    resume_motion_reasserted = True
            except RuntimeError as error:
                if "游戏窗口不在前台" not in str(error):
                    raise
                logger.info("恢复驾驶时前台已变化，撤销本次操作并重新识别场景")
                return "resume_state"
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
                preserve_static_map=(preconfigured_autopilot or opening_autopilot_attempted or bool(
                    getattr(bot, "_opening_autopilot_retry_pending", False))),
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
            and not resume_existing
            and not opening_autopilot_attempted
            and not abandoned_native_route
        ):
            # Only a fresh preconfigured entry may retry setup here. Resuming
            # an initialized round must preserve its current navigation owner.
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

    if not autopilot_set and getattr(bot, "_autopilot_waiting_for_start", False):
        setattr(bot, "_round_control_initialized", False)
        setattr(bot, "_opening_autopilot_attempted", False)
        return "resume_state"

    if not autopilot_set and getattr(bot, "_opening_autopilot_retry_pending", False):
        setattr(bot, "_round_control_initialized", False)
        setattr(bot, "_opening_autopilot_attempted", False)
        logger.info("自动航行配置中断，保留剩余尝试；重新识别后接续，不切换后备驾驶")
        return "resume_state"

    if not autopilot_set and intervention is not None and (
        intervention.command_generation_paused()
        or intervention.poll(bot.gamepad, time.monotonic())
    ):
        mark_pause = getattr(bot, "mark_manual_pause", None)
        if mark_pause is not None:
            mark_pause()
        logger.info("[USER] 自动航行配置期间用户介入；不启用后备驾驶，等待重新判定场景")
        return "resume_state"

    if not autopilot_set and not dead_at_start and not normalize_tactical_map_overlay(bot):
        logger.info("自动航行未完成且战术地图仍未收尾，交回场景路由")
        return "resume_state"

    if not autopilot_set and not dead_at_start:
        synchronize = getattr(bot, "_resynchronize_forward_controls", None)
        if synchronize is not None and (not resume_motion_reasserted or not resume_existing):
            # The failed M-map attempt may have changed the actual game helm.
            # Repair both engine and Q/E caches before any fallback command.
            synchronize()
            resume_motion_reasserted = True
        enable_center_route = getattr(bot, "enable_generic_center_route", None)
        if enable_center_route is not None and not (
            resume_existing and getattr(bot, "generic_center_route_active", False)
        ):
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
    elif resume_existing and not dead_at_start:
        bot.last_movement_reason = "接续本局小地图驾驶，保留计时与航线，不重做开局"
        logger.info("恢复当前战斗，继续原有小地图驾驶")
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
    # A normal battle must also have an absolute upper bound.  If the scene
    # classifier keeps reporting BATTLE after the match ended (result-page OCR
    # never stabilises, a sunk ship is misread as alive, or the battle-HUD
    # interlock misfires) the loop below would otherwise never return and the
    # soft round/time limits would be deferred forever.
    battle_strategy = getattr(bot, "strategy", None) or {}
    max_battle_seconds = (
        float(battle_strategy.get("max_battle_seconds", 2700.0))
        if isinstance(battle_strategy, dict)
        else 2700.0
    )
    battle_deadline = (
        None
        if quick_battle
        else quick_started_at + max(60.0, max_battle_seconds)
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
        if battle_deadline is not None and now >= battle_deadline:
            logger.warning(
                "战斗超过最长时限（%.0f 秒）仍未确认结束，强制结束本局",
                max_battle_seconds,
            )
            return "battle_timeout"
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
        last_analysis = getattr(bot, "last_analysis", None)
        if quick_battle and getattr(last_analysis, "health", 1.0) <= 0.01:
            logger.info("快速战斗确认舰船沉没；按 Esc 后直接续局，不等待结算/港口")
            return "quick_death"
        if tick_result == "ended":
            if quick_battle:
                # A naturally ended match may have a normal results page;
                # a sunk ship has already taken the distinct branch above.
                logger.info("快速战斗已自然结束；重新识别当前页面，不统计收益")
                return "quick_ended"
            break
        if (
            tick_result == "waiting"
            and last_analysis is not None
            and not bool(getattr(last_analysis, "in_battle", False))
        ):
            # Cached zero HP persists after sinking, including on the survey
            # overlay. Scene recovery must work for sunk ships as well; it
            # does not close/count the round or authorize a return-to-port.
            non_battle_frames += 1
            if non_battle_frames >= 3:
                logger.warning(
                    "连续三帧确认当前并非战斗 HUD，撤销战斗循环并按当前场景恢复"
                )
                return "resume_state"
        else:
            non_battle_frames = 0
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
    *,
    geometry: TacticalMapGeometry | None = None,
) -> tuple[int, int]:
    """Map a waypoint through the measured M-grid's perspective transform."""
    target_x = max(0.05, min(float(normalized_target[0]), 0.95))
    target_y = max(0.05, min(float(normalized_target[1]), 0.95))
    if geometry is not None:
        if geometry.frame_shape != (height, width):
            raise ValueError("Tactical grid measurement belongs to a different frame size")
        return geometry.normalized_to_pixel((target_x, target_y))
    # Keep the public estimate for callers inspecting coordinates without an
    # image. Input dispatch below always supplies a verified fresh geometry.
    # The tactical grid uses about 81% of the vertical viewport. The former
    # 94% estimate clicked outside the actual 10x10 grid on 2560x1600 and the
    # game silently ignored the autopilot destination.
    map_size = min(float(width), float(height)) * 0.81
    left = (float(width) - map_size) / 2.0
    top = (float(height) - map_size) / 2.0
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
    """Confirm M from its measured grid or explicit instructions anywhere."""
    if image is None or image.size == 0:
        return False
    if detect_tactical_map_geometry(image) is not None:
        return True
    backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
    if backend is None:
        return False
    try:
        combined = "".join(
            "".join(str(token.text or "").split())
            for token in backend.recognize(image) or []
            if float(getattr(token, "confidence", 0.0)) >= 0.55
        )
        return any(text in combined for text in (
            "自动驾驶控制", "离开战术地图模式", "设置自动航行",
        ))
    except Exception:
        logger.debug("战术地图说明/网格 OCR 失败", exc_info=True)
        return False


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
    def defer_setup():
        # A pause does not prove that the waypoint failed. Preserve the budget.
        setattr(bot, "_opening_autopilot_retry_pending", True)
        setattr(bot, "_opening_autopilot_attempted", False)
        setattr(bot, "_tactical_map_attempted_this_battle", False)
        return False

    if should_stop is not None and should_stop():
        logger.info("终止请求已到达，取消开局自动航行配置")
        return False
    if operation_paused(bot):
        return defer_setup()
    if not normalize_tactical_map_overlay(bot):
        return defer_setup()
    try:
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
        if classify_battle_continuity_screen(bot, image) != ScreenState.BATTLE:
            setattr(bot, "_opening_autopilot_attempted", False)
            return False
        # Preparation exposes the HUD and M map before navigation is usable.
        # Require the actual match timer to advance before spending attempts.
        backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
        autopilot_reader = getattr(bot.vision, "read_autopilot_enabled_text", None)
        if getattr(bot, "_opening_autopilot_retry_pending", False):
            if callable(autopilot_reader) and backend is not None and autopilot_reader(image, backend):
                target = getattr(bot, "_opening_autopilot_pending_target", None)
                try:
                    enable("暂停后已确认的原生航线", target_normalized=target)
                except TypeError:
                    enable("暂停后已确认的原生航线")
                setattr(bot, "_native_autopilot_confirmed", True)
                setattr(bot, "_tactical_map_attempted_this_battle", True)
                setattr(bot, "_opening_autopilot_retry_pending", False)
                logger.info("暂停后新截图确认自动驾驶已启用；接续已有航线")
                return True
            setattr(bot, "_opening_autopilot_retry_pending", False)
        clock_reader = getattr(bot.vision, "read_battle_clock_seconds", None)
        if callable(clock_reader):
            start_deadline = time.monotonic() + 60.0
            setattr(bot, "_autopilot_waiting_for_start", True)
            logger.info("自动航行等待正式开战：战斗计时递减后再设置航点")
            while time.monotonic() < start_deadline:
                if should_stop and should_stop():
                    return False
                if operation_paused(bot):
                    return defer_setup()
                if observe_battle_phase(bot, image, ScreenState.BATTLE) == BattlePhase.ACTIVE:
                    setattr(bot, "_autopilot_waiting_for_start", False)
                    break
                time.sleep(0.5)
                image = bot.vision.grab(bot.hwnd, allow_stale=True)
                if classify_battle_continuity_screen(bot, image) != ScreenState.BATTLE:
                    setattr(bot, "_opening_autopilot_attempted", False)
                    return False
            else:
                setattr(bot, "_opening_autopilot_attempted", False)
                logger.warning("尚未确认正式开战，保留自动航行机会并重新识别")
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
                return defer_setup()
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
                    setattr(bot, "_opening_autopilot_attempted", False)
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
        prefer_message_click = bool(getattr(bot, "_tactical_map_prefer_message_click", False))
        attempts_used = min(3, max(0, int(getattr(bot, "_tactical_map_attempts_used", 0))))

        for route_attempt in range(attempts_used, 3):
            attempt_number = route_attempt + 1
            if should_stop is not None and should_stop():
                logger.info("终止请求已到达，取消剩余自动航行尝试")
                return False
            if operation_paused(bot):
                return defer_setup()
            if route_attempt and not normalize_tactical_map_overlay(bot):
                logger.warning("第 %s/3 次自动驾驶重试前无法安全关闭 M 图", attempt_number)
                return defer_setup()
            if intervention is not None and intervention.poll(bot.gamepad):
                mark_pause = getattr(bot, "mark_manual_pause", None)
                if mark_pause is not None:
                    mark_pause()
                logger.info("用户键盘介入，取消剩余自动航行尝试")
                return defer_setup()

            normalized_target = opening_autopilot_target(
                player_normalized,
                retrying=retrying,
                attempt_index=route_attempt,
            )
            toggle_map()
            acknowledge_automation_input(bot)
            # An accepted first M starts this battle's bounded three-attempt
            # sequence. Rejected keyboard dispatch raises before this flag.
            setattr(bot, "_tactical_map_attempted_this_battle", True)
            setattr(bot, "_opening_autopilot_budget_active", True)
            setattr(bot, "_tactical_map_left_open", True)
            setattr(bot, "_tactical_map_attempts_used", attempt_number)
            setattr(bot, "_opening_autopilot_pending_target", normalized_target)
            time.sleep(0.90)

            tactical_static_frames = []
            open_confirmations = 0
            for sample_index in range(5):
                if should_stop is not None and should_stop():
                    return False
                if operation_paused(bot):
                    return defer_setup()
                # M is static while the route is being configured. grab still
                # captures a new frame; allow equal pixels across samples.
                tactical_frame = bot.vision.grab(bot.hwnd, allow_stale=True)
                if tactical_map_is_open(bot, tactical_frame):
                    open_confirmations += 1
                elif (
                    getattr(bot.vision, "_has_loading_start_action", None)
                    is not None
                    and loading_start_confirmed(bot, tactical_frame)
                ):
                    # The live HUD can appear for one transitional frame before
                    # the ship is actually spawned.  On that pre-battle roster
                    # M does not open the tactical map.  Do not burn all three
                    # waypoint offsets against the roster page.
                    setattr(bot, "_tactical_map_left_open", False)
                    setattr(bot, "_tactical_map_attempted_this_battle", False)
                    setattr(bot, "_tactical_map_attempts_used", route_attempt)
                    setattr(bot, "_opening_autopilot_attempted", False)
                    logger.info(
                        "M 图尝试遇到战前队伍/开始战斗页；等待稳定战斗 HUD 后再导航"
                    )
                    return False
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

            latest_map = tactical_static_frames[-1]
            if any(tuple(frame.shape[:2]) != (height, width)
                   for frame in tactical_static_frames):
                logger.warning("M 图采样期间窗口尺寸改变；取消航点点击并安全关闭地图")
                normalize_tactical_map_overlay(bot)
                return False
            geometry = detect_tactical_map_geometry(latest_map)
            if geometry is None:
                _save_navigation_debug(bot, f"tactical_grid_unconfirmed_{attempt_number}", latest_map)
                logger.warning("第 %s/3 次最新 M 图格线无法测量；不点击，安全关闭后重新采样", attempt_number)
                if not normalize_tactical_map_overlay(bot):
                    return defer_setup()
                continue
            local_x, local_y = tactical_map_local_point(
                width, height, normalized_target, geometry=geometry,
            )

            if not static_capture_started and begin_static_capture is not None:
                begin_static_capture()
                static_capture_started = True

            if intervention is not None and intervention.poll(bot.gamepad):
                mark_pause = getattr(bot, "mark_manual_pause", None)
                if mark_pause is not None:
                    mark_pause()
                logger.info("用户键盘介入，战术地图停止落点并保留给统一恢复关闭")
                return defer_setup()

            # Dispatch success only proves Windows accepted the event, not
            # that the game created a route. Retry through client messages
            # after an unconfirmed physical click instead of repeating the
            # same ineffective input path for all three destinations.
            rect = get_client_rect(bot.hwnd)
            client_width = rect.get("width", rect.get("right", rect["left"]) - rect["left"])
            client_height = rect.get("height", rect.get("bottom", rect["top"]) - rect["top"])
            if (client_width, client_height) != (width, height):
                logger.warning("点击前客户区尺寸与最新 M 图不一致；取消航点并安全关闭地图")
                normalize_tactical_map_overlay(bot)
                return False
            if should_stop is not None and should_stop():
                return False
            if operation_paused(bot):
                return defer_setup()
            _save_navigation_debug(
                bot, f"autopilot_map_before_click_{attempt_number}",
                tactical_static_frames[-1],
            )
            logger.info(
                "自动航行落点 (%s/3): local=(%s,%s) backend=%s",
                attempt_number, local_x, local_y,
                "window_message" if prefer_message_click else "physical",
            )
            clicked = False
            if not prefer_message_click:
                clicked = physical_click(
                    rect["left"] + local_x,
                    rect["top"] + local_y,
                    extra_delay=0.1,
                    hwnd=bot.hwnd,
                )
            if not clicked:
                clicked = window_message_click(
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
                prefer_message_click = True
                setattr(bot, "_tactical_map_prefer_message_click", True)
                for confirmation_attempt in range(4):
                    if should_stop is not None and should_stop():
                        return False
                    if operation_paused(bot):
                        return defer_setup()
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
            bot._native_spawn_position = player_normalized
            setattr(bot, "_opening_autopilot_retry_pending", False)
            if autopilot_reader is not None and backend is not None:
                # Carry the observed HUD confirmation into the battle loop.
                # A later unreadable frame must not undo this verified setup.
                bot._native_autopilot_confirmed = True
            logger.info(
                "[SYSTEM] 战术地图自动航行成功: %s | local=(%s,%s)",
                target_label,
                local_x,
                local_y,
            )
            return True

        if not normalize_tactical_map_overlay(bot):
            logger.warning("三次自动驾驶均失败且 M 图尚未安全收尾")
            return defer_setup()
        setattr(bot, "_tactical_map_attempted_this_battle", True)
        logger.warning("自动驾驶三次尝试均未确认成功；切换 Q/E 驾驶兜底")
        return False
    except Exception as error:
        logger.exception("设置战术地图自动航行失败")
        if isinstance(error, CaptureFault) or operation_paused(bot):
            return defer_setup()
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
    image = None
    trace = blockage_trace(bot, "result_rewards")
    # OCR can finish one numeric column a frame later than the others while
    # the result panel animates. Vote per resource instead of requiring one
    # whole three-column tuple to match byte-for-byte; every value still needs
    # agreement from two independent result frames.
    field_votes: dict[str, dict[int, int]] = {
        "credits": {},
        "ship_xp": {},
        "free_xp": {},
    }
    outcome_votes = {}

    def confirmed_outcome():
        confirmed = [name for name, count in outcome_votes.items() if count >= 2]
        return confirmed[0] if len(confirmed) == 1 else "unknown"

    for attempt in range(max(1, attempts)):
        if operation_paused(bot):
            logger.info("[USER] 结算读取暂停，不切窗口、不继续 OCR")
            return False, fallback, ScreenState.UNKNOWN
        if not ensure_capture_foreground(bot):
            time.sleep(0.5)
            continue
        try:
            if attempt:
                time.sleep(0.5)
            if operation_paused(bot):
                return False, fallback, ScreenState.UNKNOWN
            # last_analysis belongs to the last combat tick, not the recovery
            # observer that just confirmed RESULTS. Reusing it on attempt zero
            # made every settlement retry immediately return the same old
            # BATTLE frame, so the outer lifecycle could never reach reward OCR.
            # allow_stale permits a static page; grab still captures the window.
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
            last_state = bot.vision.classify_screen(image)
        except CaptureFault as error:
            logger.info("结算读取画面暂不可用，保留结算流程并重试: %s", error)
            continue
        if last_state == ScreenState.UNKNOWN:
            try:
                if is_battle_survey_page(image, backend=reader.backend):
                    return page_confirmed, replace(fallback, recognized=False), ScreenState.SURVEY
            except Exception:
                logger.debug("结算遮挡页 OCR 暂不可用，保留页面", exc_info=True)
        if last_state in {ScreenState.ESCAPE_MENU, ScreenState.EXIT_CONFIRMATION}:
            return page_confirmed, replace(fallback, recognized=False), last_state
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
        if last_state == ScreenState.BATTLE:
            # Earlier result-like loading frames cannot settle a live battle.
            trace.finish(image, state=last_state, action="live_battle",
                         attempt=attempt + 1, outcome="battle_in_progress",
                         details={"source": "fresh_capture", "battle_hud": battle_hud_visible})
            getattr(bot, "_blockage_traces", {}).pop("result_rewards", None)
            return False, replace(fallback, recognized=False), last_state
        trace.record(image, state=last_state, action="read_rewards", attempt=attempt + 1,
                     details={"source": "fresh_capture", "reward_surface": reward_surface,
                              "battle_hud": battle_hud_visible})
        if not reward_surface:
            result_frames = 0
            if not page_confirmed:
                outcome_votes.clear()
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
            # later frames. Unrecognized rewards must never authorize navigation.
            logger.warning("结算收益 OCR 暂时失败，继续读取后续帧: %s", error)
            continue
        if rewards.outcome in {"victory", "defeat"}:
            outcome_votes[rewards.outcome] = outcome_votes.get(rewards.outcome, 0) + 1
        if rewards.recognized or not fallback.recognized:
            fallback = rewards
        if page_confirmed:
            values = rewards.resource_values()
            minimum_credits = int(
                getattr(reader, "MINIMUM_CREDITS", ResultRewardReader.MINIMUM_CREDITS)
            )
            if values["credits"] >= minimum_credits or rewards.explicit_zero("credits"):
                field_votes["credits"][values["credits"]] = (
                    field_votes["credits"].get(values["credits"], 0) + 1
                )
            for field in ("ship_xp", "free_xp"):
                if values[field] > 0 or rewards.explicit_zero(field):
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
                debug_dir = getattr(bot, "_debug_dir", None)
                save_debug = getattr(bot.vision, "save_debug_frame", None)
                if debug_dir is not None and save_debug is not None:
                    save_debug(
                        Path(debug_dir)
                        / f"result_confirmed_{time.strftime('%H%M%S')}.png",
                        image,
                    )
                logger.info(
                    "结算 OCR 三列共识: values=%s raw=%s confidence=%s outcome=%s outcome_votes=%s",
                    consensus,
                    rewards.raw_text,
                    rewards.confidence,
                    confirmed_outcome(),
                    outcome_votes,
                )
                trace.finish(image, state=last_state, action="reward_consensus",
                             attempt=attempt + 1, outcome="recognized",
                             details={"values": consensus, "battle_outcome": confirmed_outcome()})
                getattr(bot, "_blockage_traces", {}).pop("result_rewards", None)
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
                        outcome=confirmed_outcome(),
                    ),
                    last_state,
                )
    if page_confirmed:
        logger.warning(
            "结算页已确认但资源 OCR 尚未取得三列共识: raw=%s confidence=%s",
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
    if page_confirmed:
        logger.info("结算胜负确认: outcome=%s votes=%s", confirmed_outcome(), outcome_votes)
    trace.finish(image, state=last_state, action="reward_retry",
                 attempt=max(1, attempts), outcome="consensus_missing",
                 details={"page_confirmed": page_confirmed,
                          "field_votes": field_votes, "raw_text": fallback.raw_text})
    return page_confirmed, replace(fallback, recognized=False, outcome=confirmed_outcome()), last_state


def collect_rewards_before_navigation(
    bot, reader, *, resume, on_retry, max_retries=3, attempts_per_retry=6,
):
    """Retry rewards without turning missing statistics into a failed plan.

    A currently confirmed result page can close the round with unknown
    rewards after bounded retries. Older result evidence cannot authorize
    navigation on a changed/unknown page; the lifecycle must observe it again.
    Pauses wait at the resume gate without consuming the OCR retry budget.
    """
    pending = False
    retries = 0
    outcomes = set()
    bot._settlement_recovery_pending = False
    while resume():
        confirmed, rewards, state = collect_battle_rewards(
            bot, reader, attempts=max(3, int(attempts_per_retry)),
        )
        if operation_paused(bot):
            bot.last_analysis = None
            time.sleep(0.25)
            continue
        if state == ScreenState.BATTLE:
            # A user may have started the next match during a pause after we
            # already confirmed this settlement. Preserve that boundary for
            # the lifecycle, which closes the old round before taking over.
            bot._settlement_recovery_pending = pending
            return confirmed, rewards, state
        if confirmed and rewards.recognized:
            return confirmed, rewards, state
        pending = pending or confirmed
        if confirmed and rewards.outcome in {"victory", "defeat"}:
            outcomes.add(rewards.outcome)
        if state == ScreenState.SURVEY:
            dismiss_current_battle_survey(bot)
        elif state in {ScreenState.ESCAPE_MENU, ScreenState.EXIT_CONFIRMATION}:
            dismiss_battle_overlay(bot, state)
        if not pending:
            return confirmed, rewards, state
        on_retry()
        retries += 1
        if retries >= max(1, int(max_retries)):
            unknown_rewards = replace(
                rewards, credits=0, ship_xp=0, free_xp=0, recognized=False,
                outcome=next(iter(outcomes)) if len(outcomes) == 1 else "unknown",
            )
            if confirmed and state == ScreenState.RESULTS:
                trace = getattr(bot, "_blockage_traces", {}).pop("result_rewards", None)
                logger.warning(
                    "结算页已确认，但收益在 %s 组重试后仍无共识；"
                    "收益记为未识别并继续计划，诊断截图=%s",
                    retries, getattr(trace, "folder", None),
                )
                return True, unknown_rewards, state
            bot._settlement_recovery_pending = pending
            logger.warning(
                "收益重试后当前页面=%s，保留已见结算证据并重新识别；不终止任务",
                state.value,
            )
            return False, unknown_rewards, state
        bot.last_analysis = None
        time.sleep(1)
    return None


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
    trace = blockage_trace(bot, "battle_overlay")
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
            before = bot.vision.grab(bot.hwnd, allow_stale=True)
            current = classify_runtime_screen(bot, before)
            if current not in overlay_states:
                finish_blockage_trace(bot, "battle_overlay", before, current)
                return current
            state = current
            backend = getattr(getattr(bot, "distance_reader", None), "backend", None)
            verified = battle_overlay_text_confirmed(before, backend, state)
            if state == ScreenState.EXIT_CONFIRMATION and not verified:
                verified = is_early_exit_confirmation_page(before, backend)
            if not verified:
                trace.record(before, state=state, action="overlay_text_unconfirmed",
                             attempt=attempt + 1, phase="result")
                logger.info("疑似战斗覆盖层未通过文字复核，保留画面，不发送 Esc")
                return state
            trace.record(before, state=state, action="escape", attempt=attempt + 1,
                         phase="before", details={"verified_overlay_text": True})
            if operation_paused(bot):
                return state
            logger.info(
                "已确认战斗覆盖层 %s，按 Esc 恢复战斗 (%s/%s)",
                state.value,
                attempt + 1,
                max(1, int(attempts)),
            )
            escape()
            acknowledge_automation_input(bot)
        except CaptureFault:
            return ScreenState.UNKNOWN
        except RuntimeError as error:
            logger.info("战斗覆盖层恢复键暂未派发: %s", error)
            return state
        time.sleep(0.45)
        if operation_paused(bot):
            return ScreenState.UNKNOWN
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
            state = classify_runtime_screen(bot, image)
            trace.record(image, state=state, action="escape", attempt=attempt + 1,
                         phase="after")
            if state not in overlay_states:
                finish_blockage_trace(bot, "battle_overlay", image, state, attempt=attempt + 1)
        except CaptureFault:
            return ScreenState.UNKNOWN
    return state


def return_to_port(
    bot: BattleBot,
    attempts: int = 5,
    *,
    post_battle_context: bool = False,
):
    logger.info("等待结算并返回港口")
    unknown_escape_attempts = int(getattr(bot, "_unknown_escape_attempts", 0))
    post_battle_seen = bool(post_battle_context)
    trace = blockage_trace(bot, "return_to_port")
    pending_action = None
    image = None
    state = ScreenState.UNKNOWN

    def before(action, attempt):
        nonlocal pending_action
        trace.record(image, state=state, action=action, attempt=attempt, phase="before")
        pending_action = (action, attempt)

    # The final slot only observes the last dispatched action. Previously the
    # third Esc could successfully close the blocker but still report failure.
    for attempt in range(1, max(1, int(attempts)) + 2):
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
        if state == ScreenState.BATTLE and post_battle_seen and backend is not None:
            try:
                if port_battle_action_point(image, backend) is not None:
                    logger.info("结算后画面已由“加入战斗”文字确认港口，修正宽泛 HUD 判定")
                    state = ScreenState.PORT
            except Exception:
                logger.debug("结算后港口文字复核失败", exc_info=True)
        if pending_action is not None:
            action, action_attempt = pending_action
            trace.record(image, state=state, action=action, attempt=action_attempt, phase="after")
            pending_action = None
        trace.record(image, state=state, action="inspect", attempt=attempt)
        if attempt > max(1, int(attempts)) and state not in {ScreenState.PORT, ScreenState.BATTLE}:
            break
        try:
            survey_open = is_battle_survey_page(image, backend=backend)
        except Exception:
            logger.debug("回港流程的战斗评价页面识别失败", exc_info=True)
            survey_open = False
        if survey_open:
            post_battle_seen = True
            if operation_paused(bot):
                return False
            before("dismiss_survey", attempt)
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
            bot._unknown_escape_attempts = 0
            bot._recovery_needs_user = False
            logger.info("已返回港口")
            finish_blockage_trace(bot, "return_to_port", image, state, attempt=attempt)
            return True
        if state == ScreenState.DAILY_REWARD:
            backend = getattr(
                getattr(bot, "distance_reader", None), "backend", None
            )
            before("claim_daily_reward", attempt)
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
            before("cancel_port_exit", attempt)
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
            bot._unknown_escape_attempts = 0
            bot._recovery_needs_user = False
            # A live match is not an unknown dialog.  Do not press Esc here:
            # the caller must hand control back to the battle loop.
            logger.info("仍在战斗中，取消回港操作并恢复战斗控制")
            finish_blockage_trace(bot, "return_to_port", image, state,
                                  attempt=attempt, outcome="live_battle")
            return False
        if operation_paused(bot):
            return False
        logger.info("返回港口检查 (%s/%s): %s", attempt, attempts, state.value)
        if state == ScreenState.RESULTS:
            post_battle_seen = True
            before("navigate_result", attempt)
            handle_post_battle(
                bot.hwnd,
                vision=bot.vision,
                backend=backend,
                should_abort=lambda: operation_paused(bot),
                diagnostic_dir=trace.root,
            )
        elif state in {ScreenState.ESCAPE_MENU, ScreenState.EXIT_CONFIRMATION}:
            before("dismiss_battle_overlay", attempt)
            dismiss_battle_overlay(bot, state)
            if not post_battle_seen:
                return False
            # Result overlays can briefly expose an exit-menu state while the
            # game is transitioning back to its post-battle page. Reclassify
            # on the next frame instead of abandoning recovery immediately.
            time.sleep(0.8)
            continue
        else:
            # A leftover battle-type selector page is classified UNKNOWN (it
            # has no join button), so the generic recovery never closes it and
            # the loop can spin forever on the same selector.  Detect it here
            # and Esc back to port, where selection starts over.
            try:
                if in_battle_type_selector(image, backend=backend) and recovery_escape_hint(bot, image, state) is not None:
                    logger.info("战斗模式选择页已确认角落 Esc 关闭提示，返回港口")
                    if not operation_paused(bot) and ensure_capture_foreground(bot):
                        escape = getattr(getattr(bot, "gamepad", None), "escape", None)
                        if escape is not None and unknown_escape_attempts < 3:
                            before("escape_selector", unknown_escape_attempts + 1)
                            escape()
                            unknown_escape_attempts += 1
                            bot._unknown_escape_attempts = unknown_escape_attempts
                            acknowledge_automation_input(bot)
                            time.sleep(0.8)
                            continue
            except Exception:
                logger.debug("战斗模式选择页识别失败", exc_info=True)
            # The user may land on an event/news/modal page that is not in the
            # preset classifier. Use a bounded escape ladder, reclassifying
            # after every key. If this was actually port, the resulting quit
            # confirmation is positively detected above and cancelled with
            # its verified “否” action instead of blindly pressing again.
            # Some clients leave an ad/news/loading page above the result
            # screen. It is not UNKNOWN, so the old ladder never pressed Esc
            # and the loop repeatedly observed the same blocker. Only enable
            # this extra escape path after a positive battle-end signal; an
            # ordinary startup loading page must remain untouched.
            recoverable_post_battle_page = (
                post_battle_seen and state == ScreenState.LOADING
            )
            if (
                state == ScreenState.UNKNOWN or recoverable_post_battle_page
            ) and unknown_escape_attempts < 3:
                hint = recovery_escape_hint(bot, image, state)
                if hint is None:
                    trace.record(image, state=state, action="escape_not_authorized",
                                 attempt=attempt,
                                 details={"reason": "no_verified_corner_close_hint"})
                    logger.info("回港画面未确认角落 Esc 关闭提示，保留截图并等待重新识别")
                    time.sleep(0.8)
                    continue
                escape = getattr(getattr(bot, "gamepad", None), "escape", None)
                if escape is not None and ensure_capture_foreground(bot):
                    if operation_paused(bot):
                        return False
                    before("escape_unknown", unknown_escape_attempts + 1)
                    if recoverable_post_battle_page:
                        logger.warning(
                            "结算后加载/广告页面阻塞，发送 Esc 尝试返回港口 (%s/3)；下一帧重新识别",
                            unknown_escape_attempts + 1,
                        )
                    else:
                        logger.warning(
                            "未识别页面，发送 Esc 尝试返回港口 (%s/3)；下一帧重新识别",
                            unknown_escape_attempts + 1,
                        )
                    escape()
                    unknown_escape_attempts += 1
                    bot._unknown_escape_attempts = unknown_escape_attempts
                    acknowledge_automation_input(bot)
                    time.sleep(0.8)
                    continue
            if recoverable_post_battle_page:
                logger.info("结算后阻塞页面已完成有限 Esc 兜底；保持识别且不再继续盲按")
            else:
                logger.info("未知页面已完成有限 Esc 兜底；保持识别且不再继续盲按")
            if unknown_escape_attempts >= 3:
                bot._recovery_needs_user = True
        time.sleep(3)
    logger.warning("未能确认已返回港口；未执行盲点操作")
    if unknown_escape_attempts >= 3 and state in {ScreenState.UNKNOWN, ScreenState.LOADING}:
        bot._recovery_needs_user = True
    finish_blockage_trace(bot, "return_to_port", image, state,
                          attempt=attempt, outcome="unresolved")
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
    trace = blockage_trace(bot, "scene_unknown")

    for attempt in range(sample_count):
        image = None
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
            # A stable, OCR-confirmed ``加入战斗`` button is positive evidence
            # that the game really returned to the port. This can happen when
            # the result/survey transition is missed during a pause or focus
            # loss. Keep the old fail-closed behaviour for colour-only port
            # guesses, but do not deadlock forever on a real actionable port.
            port_action = None
            if state == ScreenState.PORT and backend is not None:
                try:
                    port_action = port_battle_action_point(image, backend)
                except Exception:
                    logger.debug("恢复时港口加入战斗 OCR 检查失败", exc_info=True)
            if port_action is None:
                logger.warning(
                    "本局尚未确认结算，忽略 %s 判定并禁止进入港口/选船流程",
                    state.value,
                )
                state = ScreenState.UNKNOWN
            else:
                logger.warning(
                    "未读到结算页，但已连续恢复到 OCR 确认的港口加入战斗按钮；"
                    "允许从当前港口继续，旧局收益/胜负保持未知",
                )

        if state == ScreenState.UNKNOWN:
            trace.record(image, state=state, action="recognize",
                         attempt=attempt + 1,
                         details={"round_in_progress": round_in_progress})
            previous_state = ScreenState.UNKNOWN
            consecutive = 0
        elif state == previous_state:
            trace.remember(image, state=state, attempt=attempt + 1)
            consecutive += 1
        else:
            trace.remember(image, state=state, attempt=attempt + 1)
            previous_state = state
            consecutive = 1

        if state != ScreenState.UNKNOWN and consecutive >= required:
            finish_blockage_trace(bot, "scene_unknown", image, state, attempt=attempt + 1)
            if state in {ScreenState.BATTLE, ScreenState.PORT, ScreenState.RESULTS}:
                bot._unknown_overlay_escape_attempts = 0
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
    dispatched = dismiss_battle_survey(
        bot.hwnd,
        image,
        backend=backend,
        should_abort=lambda: operation_paused(bot),
        escape_action=getattr(getattr(bot, "gamepad", None), "escape", None),
    )
    if not dispatched:
        return False
    time.sleep(0.5)
    if operation_paused(bot) or not ensure_capture_foreground(bot):
        return False
    try:
        refreshed = bot.vision.grab(bot.hwnd, allow_stale=True)
    except CaptureFault:
        return False
    if is_battle_survey_page(refreshed, backend=backend):
        logger.info("评价页关闭指令已派发，但页面仍在；下一循环重新确认后重试")
        return False
    return classify_runtime_screen(bot, refreshed) != ScreenState.UNKNOWN


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
    if limits.stop_requested():
        return ResumeGateResult(allowed=False, resumed=False)
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
        else "切屏后的鼠标操作"
        if trigger == "background_mouse"
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
        movement_reason="保持现有船速与舵位；5秒静默后自动恢复，持续20秒后回到游戏静默恢复或点击网页继续",
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
                "网页点击继续"
                if resumed_by_web
                else "连续5秒无新的切屏、键盘或后台鼠标操作"
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
            logger.warning("[USER] 持续用户操作达到20秒，已锁定暂停并等待网页继续")
            reporter.update(
                "paused",
                "持续用户操作达到20秒，需在网页点击继续",
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
    if bot is not None:
        # A user may have changed ship/mode while paused. Keep this dirty
        # across battle/loading/result recovery until a verified port entry.
        bot._port_recheck_required = True
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
    window = wait_for_game_window(timeout=2, should_stop=limits.stop_requested)
    if limits.stop_requested():
        reporter.update("stopped", "已按用户要求停止启动")
        return 0
    if window is None:
        launcher_client = (
            os.environ.get("WOWS_LAUNCHER_CLIENT", last_launch_client()).strip().lower()
            or "steam"
        )
        launcher_name = "WG 游戏本体" if launcher_client == "wgc" else "Steam"
        logger.info("游戏未运行，使用 %s 自动检索路径并启动", launcher_name)
        reporter.update("launching_game", "正在启动游戏本体，等待后台初始化"
                        if launcher_client == "wgc"
                        else f"正在通过 {launcher_name} 启动战舰世界")
        launch_started = time.monotonic()
        result = launch_game(client=launcher_client, should_stop=limits.stop_requested)
        if limits.stop_requested():
            reporter.update("stopped", "已按用户要求停止启动")
            return 0
        if not result.started:
            logger.error("无法自动启动游戏: %s", result.detail)
            reporter.update(
                "failed",
                result.detail,
                error="game_launch_failed",
            )
            return 1
        logger.info("已请求自动启动游戏: %s (%s)", result.method, result.detail)
        reporter.update("launching_game", "启动请求已发出，正在等待战舰世界窗口"
                        if result.method == "wgc_direct"
                        else f"正在通过 {launcher_name} 启动战舰世界")
        launch_timeout = float(os.environ.get("WOWS_GAME_LAUNCH_TIMEOUT", "300"))
        remaining = max(0.0, launch_timeout - (time.monotonic() - launch_started))
        window = wait_for_game_window(timeout=remaining, should_stop=limits.stop_requested,
                                      launch_process=getattr(result, "process", None))
        if window is None and not limits.stop_requested():
            process = getattr(result, "process", None)
            code = process.poll() if process is not None else None
            if isinstance(code, int):
                reporter.update("failed", f"游戏本体已退出({code})，未进入游戏窗口",
                                error="game_process_exited")
                return 1
    if limits.stop_requested():
        reporter.update("stopped", "已按用户要求停止启动")
        return 0
    if window is None:
        logger.error("未找到游戏窗口")
        reporter.update("failed", "未找到游戏窗口", error="game_window_not_found")
        return 1

    hwnd, title, rect = window
    logger.info("找到游戏窗口: %s", title)
    logger.info("窗口坐标: %s", rect)
    bot = BattleBot(hwnd, ship_config)
    bot.blockage_diagnostic_root = BLOCKAGE_ROOT
    bot.run_limits = limits
    bot.runtime_reporter = reporter
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
    reporter.update("entering_game", "正在识别当前游戏场景" if limits.resuming else "游戏已启动，正在等待港口界面")
    screen_timeout = float(os.environ.get("WOWS_GAME_SCREEN_TIMEOUT", "300"))
    initial_frame, initial_state = None, ScreenState.UNKNOWN
    if limits.resuming:
        # A continued task may already be in combat. Observe the live page
        # before any startup recovery clicks or port-only setup.
        initial_state = recover_current_scene(
            bot, attempts=6, stable_frames=2, poll_interval=0.2,
        )
        logger.info("接续任务即时场景判定: %s", initial_state.value)
    if initial_state not in {ScreenState.PORT, ScreenState.BATTLE, ScreenState.RESULTS}:
        initial_frame, initial_state = wait_for_recognized_screen(
            bot,
            timeout=screen_timeout,
        )
    if limits.stop_requested():
        reporter.update("stopped", "已按用户要求安全停止")
        shutdown_bot(bot)
        return 0
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
        if not limits.resuming and return_to_port(bot, attempts=3):
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
    started_at = time.monotonic() - limits.initial_elapsed_seconds
    completed_rounds = limits.initial_completed_rounds
    current_round = completed_rounds
    if completed_rounds:
        logger.info("接续原任务：已完成 %s 局，下一局为第 %s 局", completed_rounds, completed_rounds + 1)
    port_configured = False
    # Ship selection is a task-level setup, not a per-round action.  Once the
    # first battle has been positively entered, keep the confirmed ship locked
    # and let later rounds use the same selected card from port without
    # scrolling the carousel again.  This also prevents a transient result/
    # loading classification from sending the workflow back into ship search.
    ship_locked = False
    preparation_failures = 0
    battle_recovery_failures = 0
    battle_timeout_count = 0
    quick_closure_failures = 0
    daily_reward_failures = 0
    # Once matchmaking/loading/battle has been committed, PORT is not allowed
    # to start selection again until settlement positively closes this round.
    round_in_progress = False
    round_entry_pending = False
    # A stable result page is the only boundary that permits a later PORT to
    # close the round. Keep this evidence across pauses because the player may
    # press Return to Port before reward OCR resumes.
    round_result_seen = False
    plan_completed = False
    if limits.resuming and initial_state == ScreenState.BATTLE:
        # This bot object is new, but the game match is not. Initialise local
        # diagnostics once without sending helm input, and then use the
        # existing-battle path (which never opens M to overwrite a live route).
        bot.reset(preserve_movement=True)
        setattr(bot, "_round_control_initialized", True)
        round_in_progress = True
        ship_locked = True
        logger.info("接续时仍处于战斗 HUD；保留局内驾驶，立即进入场景循环")

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
            if current_scene != ScreenState.BATTLE:
                observe_battle_phase(bot, None, current_scene)
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
                ship_locked = True
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
                ship_locked = True
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
                    ship_locked = True
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
                    daily_reward_failures = 0
                    time.sleep(1.0)
                else:
                    daily_reward_failures += 1
                    logger.warning(
                        "每日奖励本次未能确认领取 (%s/%s)；保留页面等待重试",
                        daily_reward_failures,
                        3,
                    )
                    if daily_reward_failures >= 3:
                        # A permanently unclickable reward page must not stall
                        # the whole plan.  Esc leaves it unclaimed but keeps
                        # the task alive; the failure stays visible in logs.
                        logger.warning(
                            "每日奖励连续 3 次无法领取，强制关闭页面并继续任务"
                        )
                        if not operation_paused(bot):
                            escape = getattr(bot.gamepad, "escape", None)
                            if escape is not None:
                                try:
                                    escape()
                                    acknowledge_automation_input(bot)
                                    time.sleep(0.8)
                                except Exception:
                                    logger.warning(
                                        "每日奖励强制关闭失败，保留页面由下一轮重试",
                                        exc_info=True,
                                    )
                        daily_reward_failures = 0
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
                    return_to_port(bot, attempts=3, post_battle_context=True)
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
                # Select/configure the ship only once per task.  After a
                # confirmed battle entry the ship lock is authoritative for
                # later rounds; the port path then only starts the next match.
                configured_this_attempt = not ship_locked or bool(
                    getattr(bot, "_port_recheck_required", False)
                )
                if not configured_this_attempt:
                    logger.info("本任务舰船已锁定：本轮港口只进入战斗，不重新选船或滚动舰船栏")
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
                    ship_locked = True
            else:
                if round_in_progress or round_entry_pending:
                    logger.warning(
                        "本局尚未确认结算且当前场景未知；检查角落关闭提示后再决定是否恢复"
                    )
                    dismissed_state = dismiss_unknown_overlay(bot, attempts=2)
                    if dismissed_state != ScreenState.UNKNOWN:
                        logger.info("未知遮挡恢复到 %s；下一循环按当前场景接续", dismissed_state.value)
                        continue
                else:
                    logger.warning("当前场景仍未知，仅在确认角落关闭提示后尝试返回港口")
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
                if recovered_state in {ScreenState.UNKNOWN, ScreenState.LOADING} and getattr(bot, "_recovery_needs_user", False):
                    raise RuntimeError("未知/广告页面已完成三次 Esc 尝试仍未恢复，已停止重试；请关闭遮挡页面后重启任务")
                if recovered_state == ScreenState.BATTLE:
                    round_entry_pending = False
                    round_in_progress = True
                    round_result_seen = False
                    ship_locked = True
                    logger.info("准备恢复确认已在战斗中，下一循环直接接管")
                    preparation_failures = 0
                    continue
                if recovered_state == ScreenState.LOADING:
                    # Loading is not round identity.  A queue can be cancelled
                    # or a reconnect can fall back to port; only stable battle
                    # HUD evidence is allowed to acquire the active-round lock.
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
                            round_entry_pending = False
                            round_in_progress = True
                            round_result_seen = False
                            ship_locked = True
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

            if battle_finished == "battle_timeout":
                # The watchdog ended a battle that never produced a confirmed
                # result page.  Force a port return without counting the round
                # and keep the plan running; after repeated timeouts stop the
                # task instead of looping the same stuck match forever.
                battle_timeout_count += 1
                reporter.update(
                    "returning",
                    "战斗超过最长时限，正在强制返回港口；本局不计数",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    rewards_status="skipped",
                    rewards_round=0,
                    last_rewards={},
                )
                closure_confirmed = return_to_port(
                    bot, attempts=6, post_battle_context=True
                )
                if closure_confirmed:
                    battle_timeout_count = 0
                    round_in_progress = False
                    round_entry_pending = False
                    round_result_seen = False
                    setattr(bot, "_round_control_initialized", False)
                    port_configured = False
                    logger.info("战斗超时保护：已返回港口，本局不计数，继续执行计划")
                    continue
                if battle_timeout_count >= 2:
                    logger.error(
                        "战斗连续 %s 次超时且无法返回港口，终止任务防止无限循环",
                        battle_timeout_count,
                    )
                    raise RuntimeError(
                        "战斗连续超时且无法返回港口，已安全终止任务"
                    )
                logger.warning(
                    "战斗超时后回港尚未确认（%s/%s），重新识别场景",
                    battle_timeout_count,
                    2,
                )
                continue

            if battle_finished in QUICK_BATTLE_COMPLETION_REASONS:
                # Quick mode skips reward OCR. A confirmed sunk ship counts
                # when the post-Esc Continue Battle action is dispatched;
                # only a naturally finished match needs a results page.
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
                        ship_sunk=battle_finished == "quick_death",
                        should_abort=lambda: operation_paused(bot),
                        diagnostic_dir=getattr(bot, "blockage_diagnostic_root", None),
                    )
                    # Death-screen Esc exposes an immediate Continue Battle
                    # action. This is not a results screen or a port return.
                    closure_confirmed = closure_scene in {
                        ScreenState.PORT, ScreenState.RESULTS,
                        ScreenState.QUICK_REQUEUE,
                    }
                    if closure_scene == ScreenState.PORT:
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
                    if operation_paused(bot):
                        logger.info("快速退出被用户暂停；保留本局及剩余退出重试次数")
                        continue
                    quick_closure_failures += 1
                    logger.warning(
                        "快速战斗尚未确认离开当前对局，本局不计数并重新判断场景 "
                        "(%s/%s)",
                        quick_closure_failures,
                        3,
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
                    if quick_closure_failures >= 3:
                        raise RuntimeError(
                            "快速战斗连续 3 次无法确认离开本局，已安全终止任务"
                        )
                    continue
                completed_rounds = count_quick_battle_for_plan(
                    completed_rounds,
                    battle_finished,
                    closure_confirmed=True,
                )
                quick_closure_failures = 0
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
                elif closure_scene == ScreenState.QUICK_REQUEUE:
                    queued = queue_next_battle(
                        bot.hwnd,
                        vision=bot.vision,
                        backend=backend,
                        should_abort=lambda: operation_paused(bot),
                        source=ScreenState.QUICK_REQUEUE,
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
                        ship_locked = True
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
                rewards_round=current_round,
                last_rewards={},
                last_outcome="unknown",
            )
            settlement = collect_rewards_before_navigation(
                bot,
                reward_reader,
                resume=lambda: not user_stop_requested() and bool(wait_for_web_resume(
                    limits, reporter, bot, resume_state="collecting_rewards",
                )),
                on_retry=lambda: reporter.update(
                    "collecting_rewards",
                    "收益尚未可靠识别，正在重试；仍未识别时将保存截图并继续计划",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                    rewards_status="reading",
                    rewards_round=current_round,
                    last_rewards={},
                    last_outcome="unknown",
                ),
            )
            if settlement is None:
                break
            result_confirmed, rewards, post_battle_state = settlement
            if not result_confirmed:
                if getattr(bot, "_settlement_recovery_pending", False):
                    round_result_seen = True
                    logger.warning("结算读取后页面已变化；保持当前局，下一循环重新识别")
                    reporter.update(
                        "recovering",
                        "结算读取后页面已变化，保留当前局并重新识别",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        rewards_status="pending",
                        last_rewards={},
                        last_outcome="unknown",
                    )
                    continue
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
                    # collect_battle_rewards closed the async OCR service while
                    # the settlement page was still not confirmed; this same
                    # battle re-enters without bot.reset(), so rebuild it or the
                    # next combat tick submits to a shutdown executor.
                    rebuild = getattr(bot, "rebuild_distance_ocr", None)
                    if rebuild is not None:
                        rebuild()
                else:
                    logger.warning("异常/未知页面优先尝试返回港口")
                    return_to_port(bot, post_battle_context=True)
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
                logger.warning("本局结算已确认，收益记为未识别；继续执行剩余计划")
                reporter.update(
                    "collecting_rewards",
                    "本局结算已确认，收益未识别，继续执行计划",
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
                returned_to_port = return_to_port(
                    bot, attempts=5, post_battle_context=True
                )
                if not returned_to_port:
                    time.sleep(2)
                    returned_to_port = return_to_port(
                        bot, attempts=5, post_battle_context=True
                    )
                if not returned_to_port:
                    reporter.update(
                        "failed",
                        "计划局数已完成，但连续两次未能确认返回港口",
                        current_round=current_round,
                        completed_rounds=completed_rounds,
                        error="plan_complete_port_unconfirmed",
                        movement_mode="idle",
                        movement_reason="已释放控制；等待人工确认当前页面",
                        paused_by_user=False,
                        manual_intervention_active=False,
                        manual_intervention_latched=False,
                    )
                    logger.error("计划完成后连续两次未确认返回港口，不再误报完成")
                    return 1
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
                    ship_locked = True
                    logger.info("下一局 HUD 已确认，下一循环重新确认场景后接管")
                elif recovered_state == ScreenState.RESULTS:
                    round_entry_pending = False
                    return_to_port(
                        bot, attempts=2, post_battle_context=True
                    )
                    port_configured = False
                elif recovered_state == ScreenState.SURVEY:
                    dismiss_current_battle_survey(bot)
                elif recovered_state == ScreenState.UNKNOWN:
                    logger.warning("下一局场景仍未知；保持入局锁，禁止选船并继续识别")
                continue
            logger.warning("无法直接继续战斗，开始执行已验证的回港兜底")
            returned_to_port = return_to_port(
                bot, attempts=6, post_battle_context=True
            )
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
        shutdown_after_plan(limits, plan_completed)


if __name__ == "__main__":
    raise SystemExit(run())
