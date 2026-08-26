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
from core.window import (
    activate_window,
    ensure_game_window_foreground,
    find_game_window,
    get_client_rect,
    maximize_game_window,
    physical_click,
    window_message_click,
)
from port_navigator import (
    confirm_no_commander,
    enter_battle,
    ensure_requested_mode,
    handle_post_battle,
    queue_next_battle,
    select_requested_ship,
    ShipSelectionError,
)
from runtime_control import RunLimits, RuntimeReporter

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("runner")


class GameWindowUnavailableWhilePaused(RuntimeError):
    """Raised after the paused workflow loses its game window for a grace period."""


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


def wait_for_recognized_screen(bot: BattleBot, timeout: float = 300.0):
    """Wait through login/splash/loading until an actionable screen is visible."""
    deadline = time.monotonic() + max(1.0, float(timeout))
    last_state = ScreenState.UNKNOWN
    previous_state = ScreenState.UNKNOWN
    consecutive = 0
    while time.monotonic() < deadline:
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        except CaptureFault as error:
            logger.info("游戏仍在启动，画面暂不可用: %s", error)
            time.sleep(2)
            continue
        last_state = bot.vision.classify_screen(image)
        if last_state == previous_state:
            consecutive += 1
        else:
            previous_state = last_state
            consecutive = 1
        if last_state in {
            ScreenState.PORT,
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
    if not ensure_game_window_foreground(bot.hwnd):
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
        logger.info("检测到结算界面，按状态导航返回港口")
        handle_post_battle(bot.hwnd, vision=bot.vision)
        time.sleep(2)
        # Result and port screens can legitimately remain pixel-identical for
        # several seconds; freshness is only a combat safety requirement.
        image = bot.vision.grab(bot.hwnd, allow_stale=True)
        state = bot.vision.classify_screen(image)

    if state != ScreenState.PORT:
        logger.warning("当前界面为 %s，优先尝试恢复到港口", state.value)
        return_to_port(bot, attempts=2)
        return False

    # Port actions are destructive to an active match (carousel scrolling and
    # clicks). Require a second fresh port frame and give a battle HUD absolute
    # priority. This catches transitions and any one-frame classifier error
    # before the ship-selection workflow is allowed to run.
    time.sleep(0.25)
    confirmation = bot.vision.grab(bot.hwnd, allow_stale=True)
    confirmed_state = bot.vision.classify_screen(confirmation)
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
        if not select_requested_ship(bot.hwnd, ship_key, vision=bot.vision):
            logger.warning("未能安全选择目标舰船")
            return False
        if not ensure_requested_mode(bot.hwnd, mode, vision=bot.vision):
            logger.warning("未能安全选择目标战斗模式")
            return False
    if not enter_battle(bot.hwnd, vision=bot.vision, configure_port=False):
        logger.warning("“加入战斗”请求未能派发或未通过港口复核")
        return False
    time.sleep(5)
    return True


def wait_for_battle(bot: BattleBot, timeout: float = 180.0, should_stop=None):
    logger.info("等待战斗 HUD")
    deadline = time.monotonic() + timeout
    commander_confirmed = False
    last_state = None
    result_frames = 0
    battle_frames = 0
    while time.monotonic() < deadline:
        if should_stop and should_stop():
            return False
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
        battle_frames = battle_frames + 1 if state == ScreenState.BATTLE else 0
        if battle_frames >= 3:
            logger.info("战斗 HUD 已连续确认，开始接管移动")
            return True
        result_frames = result_frames + 1 if state == ScreenState.RESULTS else 0
        if result_frames >= 3:
            logger.warning("等待新战斗时持续停留结算页，交回港口恢复流程")
            return False
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
    if not ensure_game_window_foreground(bot.hwnd):
        logger.warning("无法切换《战舰世界》到前台，跳过本次战斗控制")
        return False
    try:
        bot.reset(preserve_movement=resume_existing)
    except TypeError:
        # Compatibility for small test doubles and third-party adapters.
        bot.reset()
    autopilot_set = False
    if resume_existing:
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
                enable("恢复的游戏自动航线")
        else:
            # A recovered battle must use the same opening rule as a freshly
            # detected battle: establish native autopilot first, then let the
            # Q/E controller take over only after the game route ends.
            autopilot_set = configure_opening_autopilot(bot)
    else:
        autopilot_set = configure_opening_autopilot(bot)

    if not autopilot_set:
        enable_center_route = getattr(bot, "enable_generic_center_route", None)
        if enable_center_route is not None:
            enable_center_route(
                "战术地图自动航行设置失败，通用驾驶向地图中央接管"
            )
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
    if resume_existing and autopilot_set:
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
            if progress and now - last_progress >= 1.0:
                progress(bot)
                last_progress = now
            time.sleep(0.15)
            continue
        if (
            not paused
            and int(ctypes.windll.user32.GetForegroundWindow() or 0) != bot.hwnd
        ):
            # The next combat tick may issue a command, so focus immediately
            # beforehand and mark the focus event as automation-generated.
            if not ensure_game_window_foreground(bot.hwnd):
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
        if tick_result == "ended":
            if quick_battle:
                # Quick battles never wait on or OCR the result page.  Whether
                # the five-minute limit ended the battle or the ship was sunk,
                # Esc immediately returns to port and the next loop queues a
                # fresh battle without touching task totals.
                logger.info("快速战斗已离开战斗 HUD，立即回港重开；不统计收益")
                return "quick_ended"
            break
        if (
            quick_battle
            and getattr(getattr(bot, "last_analysis", None), "health", 1.0)
            <= 0.01
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
        # A/B/C/D circle detection is retained for the radar only.  Maps have
        # different layouts and a false circle must never redirect the opening
        # route.  Map centre is the invariant, safe initial objective.
        normalized_target = (0.5, 0.5)
        target_label = "地图中心"
        local_x, local_y = tactical_map_local_point(
            width,
            height,
            normalized_target,
        )
        rect = get_client_rect(bot.hwnd)
        verify_autopilot = getattr(bot.vision, "is_autopilot_enabled", None)
        accepted = False
        for attempt in range(2):
            intervention = getattr(bot, "intervention", None)
            if intervention is not None and intervention.poll(bot.gamepad):
                mark_pause = getattr(bot, "mark_manual_pause", None)
                if mark_pause is not None:
                    mark_pause()
                logger.info("用户键盘介入，取消本次自动航行设置")
                return False
            toggle_map()
            time.sleep(0.65)
            if intervention is not None and intervention.poll(bot.gamepad):
                mark_pause = getattr(bot, "mark_manual_pause", None)
                if mark_pause is not None:
                    mark_pause()
                logger.info("用户键盘介入，战术地图已停止继续操作")
                toggle_map()
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
                    continue
            time.sleep(0.35)
            toggle_map()
            time.sleep(0.55)
            if verify_autopilot is None:
                accepted = True
                break
            verification = bot.vision.grab(bot.hwnd, allow_stale=True)
            if verify_autopilot(verification):
                accepted = True
                break
            logger.warning(
                "战术地图落点未出现自动驾驶标识，重试 %s/2",
                attempt + 1,
            )
        if not accepted:
            logger.warning("战术地图两次落点均未生效，交由通用驾驶接管")
            return False
        try:
            enable(target_label, target_normalized=normalized_target)
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
            toggle_map()
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
    stable_rewards: dict[tuple[int, int, int], int] = {}
    for attempt in range(max(1, attempts)):
        if attempt == 0 and bot.last_analysis is not None:
            image = bot.last_analysis.image
        else:
            time.sleep(0.5)
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
        last_state = bot.vision.classify_screen(image)
        if last_state != ScreenState.RESULTS:
            result_frames = 0
            if attempt >= 1 and last_state in {ScreenState.BATTLE, ScreenState.PORT}:
                break
            continue
        result_frames += 1
        page_confirmed = page_confirmed or result_frames >= 2
        rewards = reader.read(image)
        if rewards.recognized or not fallback.recognized:
            fallback = rewards
        if page_confirmed and rewards.recognized:
            signature = (
                int(rewards.credits),
                int(rewards.ship_xp),
                int(rewards.free_xp),
            )
            stable_rewards[signature] = stable_rewards.get(signature, 0) + 1
            # Result digits have a stable colour/position across frames. Two
            # identical reads prove the complete same-colour number was
            # captured, preventing a single clipped leading/trailing token
            # from entering the per-task total.
            if stable_rewards[signature] >= 2:
                return True, rewards, last_state
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
        try:
            image = bot.vision.grab(bot.hwnd, allow_stale=True)
            state = bot.vision.classify_screen(image)
        except CaptureFault as error:
            logger.info(
                "恢复检查暂时无法取得画面 (%s/%s): %s",
                attempt + 1,
                sample_count,
                error,
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
        "网页已暂停，不再下发新系统指令",
        paused_by_user=True,
        manual_intervention_latched=True,
        movement_mode="manual_pause",
        movement_reason="保持现有船速与舵位，等待网页继续",
    )
    window_missing_since = None
    while True:
        if limits.stop_requested():
            return False
        web_paused = limits.pause_requested()
        key_paused = keyboard_paused()
        if not web_paused and not key_paused:
            break
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
    reporter.update("starting", "已找到游戏窗口，正在默认最大化")
    if maximize_game_window(hwnd):
        logger.info("已在启动时最大化游戏窗口；后续切换不再改动窗口位置")
    else:
        logger.warning("未能确认游戏窗口最大化；后续仍不会改变窗口位置")
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
        if ensure_game_window_foreground(bot.hwnd):
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
        if ensure_game_window_foreground(bot.hwnd):
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
    preparation_failures = 0
    battle_recovery_failures = 0

    def should_stop():
        return limits.reached(completed_rounds, started_at)

    def user_stop_requested():
        return limits.stop_requested()

    try:
        while not should_stop():
            if not wait_for_web_resume(limits, reporter, bot):
                break
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
            if current_scene == ScreenState.BATTLE:
                prepared = True
                resuming_this_battle = True
                logger.info("当前已在战斗中：跳过选船，直接配置自动航行并接管")
            elif current_scene == ScreenState.LOADING:
                logger.info("当前处于加载中：等待 HUD 后进入战斗控制")
                prepared = wait_for_battle(bot, should_stop=should_stop)
            elif current_scene == ScreenState.RESULTS:
                logger.info("当前处于结算页：先回港，再从港口流程继续")
                return_to_port(bot, attempts=3)
                port_configured = False
                continue
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
                    bot, should_stop=should_stop
                )
            else:
                logger.warning("当前场景仍未知，按全局规则尝试 Esc 返回港口")
                return_to_port(bot, attempts=3)
                port_configured = False
            if not prepared:
                if should_stop():
                    break
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
                        ):
                            logger.info("加载恢复已确认 HUD，下一循环直接进入战斗")
                            continue
                    except (SafetyFault, CaptureFault) as error:
                        logger.info("准备恢复等待 HUD 时画面仍不稳定: %s", error)
                elif recovered_state == ScreenState.RESULTS:
                    # The observer has already confirmed the result page on
                    # consecutive frames.  Navigation is now phase-authorized.
                    return_to_port(bot, attempts=2)
                    port_configured = False
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
                    if analysis is None
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
                    stop_after_current=bool(
                        limits.duration_seconds
                        and time.monotonic() - started_at
                        >= limits.duration_seconds
                    ),
                )

            battle_finished = False
            try:
                battle_finished = run_battle(
                    bot,
                    # A time limit is a soft boundary: finish the active battle.
                    # Only an explicit user stop interrupts combat immediately.
                    should_stop=user_stop_requested,
                    progress=report_battle_progress,
                    resume_existing=resuming_this_battle,
                    quick_battle=limits.quick_battle,
                )
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

            if battle_finished in {"quick_timeout", "quick_death", "quick_ended"}:
                reporter.update(
                    "returning",
                    "快速战斗结束，正在 Esc 返回港口并重新开局；本局不统计收益",
                    current_round=current_round,
                    completed_rounds=completed_rounds,
                )
                escape = getattr(bot.gamepad, "escape", None)
                if escape is not None:
                    escape()
                    time.sleep(0.8)
                return_to_port(bot, attempts=5)
                port_configured = False
                continue
            if not battle_finished:
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
                bot,
                resume_state="requeueing",
            ):
                break
            if queue_next_battle(bot.hwnd, vision=bot.vision):
                if wait_for_battle(bot, should_stop=should_stop):
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
