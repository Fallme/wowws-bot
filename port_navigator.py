"""State-aware port and menu navigation.

All click locations are relative to the captured game window.  A click is only
issued after the corresponding screen state or button colour has been seen.
"""

import logging
import math
import os
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from core.frame_guard import CaptureFault
from core.blockage_diagnostics import BlockageTrace
from core.ui import (
    BATTLE_TYPE_COOPERATIVE_CARD,
    BATTLE_TYPE_SEARCH_AREA,
    ESCAPE_RESUME_BUTTON,
    EXIT_CONTINUE_BUTTON,
    LOADING_START_BUTTON,
    PORT_BATTLE_BUTTON,
    PORT_DIALOG_CLOSE,
    PORT_MODE_SELECTOR,
    QUICK_EXIT_BATTLE_BUTTON,
    QUICK_EXIT_CONFIRM_YES_BUTTON,
    RESULTS_REQUEUE_BUTTON,
    RESULTS_RETURN_TO_PORT_BUTTON,
    SELECTED_SHIP_NAME_TEMPLATES,
    SHIP_NAME_TEMPLATES,
    SHIP_REFERENCE_SIZE,
    RelativeRegion,
    ScreenState,
)
from core.ocr import RapidOcrBackend
from core.vision import Vision
from core.window import (
    activate_window,
    ensure_game_window_foreground,
    get_client_rect,
    get_window_rect,
    physical_click,
    physical_scroll,
    park_port_cursor,
    window_message_click,
)
from dxgi_capture import ScreenCapture

logger = logging.getLogger("port")

BASE_DIR = Path(__file__).resolve().parent
SHIP_TEMPLATE_DIR = BASE_DIR / "assets" / "ui"
LAST_SELECTED_SHIP_PATH = BASE_DIR / "data" / "last_selected_ship.txt"
SUPPORTED_MODES = {"cooperative", "asymmetric"}
SUPPORTED_SHIPS = frozenset(SHIP_NAME_TEMPLATES)
CUSTOM_SHIP_KEY = "custom"
BUILTIN_SHIP_OCR_NAMES = {
    "pommern": ("波美拉尼亚", "Pommern"),
    "napoli": ("那不勒斯", "Napoli"),
}
SHIP_TIER_PREFIXES = frozenset(
    {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
    | {str(tier) for tier in range(1, 11)}
)

# The mode emblem is the most stable visual element across Chinese UI text
# updates.  OpenCV hue: purple sits around 130-165; co-op is a dark teal/gray
# anchor, so it is selected from its stable first-card slot and then verified
# in the port header.
ASYMMETRIC_PURPLE_LOWER = np.array([125, 65, 45])
ASYMMETRIC_PURPLE_UPPER = np.array([170, 255, 255])
_WINDOW_CAPTURE = ScreenCapture()
_LAST_SELECTED_CARD_POINT = None

# Text search areas are deliberately wider than the old click rectangles. OCR
# returns the rendered glyph box, so these regions only limit false positives;
# the eventual click is always the centre of the recognized text itself.
PORT_BATTLE_TEXT_AREA = RelativeRegion(0.38, 0.0, 0.62, 0.11)
PORT_MODE_TEXT_AREA = RelativeRegion(0.51, 0.0, 0.72, 0.11)
RESULTS_RETURN_TEXT_AREA = RelativeRegion(0.25, 0.78, 0.70, 1.0)
RESULTS_REQUEUE_TEXT_AREA = RelativeRegion(0.58, 0.78, 1.0, 1.0)
QUICK_EXIT_CONFIRM_ACTION_AREA = RelativeRegion(0.34, 0.52, 0.59, 0.70)
QUICK_DEATH_ACTION_AREA = RelativeRegion(0.18, 0.14, 0.96, 0.94)
QUICK_DEATH_LEAVE_AREA = RelativeRegion(0.36, 0.78, 0.68, 1.0)
BATTLE_SURVEY_AREA = RelativeRegion(0.14, 0.12, 0.86, 0.88)
PORT_EXIT_CONFIRM_ACTION_AREA = RelativeRegion(0.40, 0.38, 0.60, 0.58)

# Resolution and in-game UI scale are independent. The first term follows the
# framebuffer width; these factors cover the common compact/normal/large UI
# settings without maintaining one template set per monitor.
UI_TEMPLATE_SCALE_FACTORS = (0.56, 0.64, 0.72, 0.84, 0.94, 1.0, 1.08, 1.20, 1.35, 1.52, 1.70)


def _ui_template_scales(width, height):
    """UI zoom follows neither framebuffer width nor aspect ratio alone."""
    bases = (width / SHIP_REFERENCE_SIZE[0], height / SHIP_REFERENCE_SIZE[1])
    return sorted({round(base * factor, 4) for base in bases
                   for factor in UI_TEMPLATE_SCALE_FACTORS})


class ShipSelectionError(RuntimeError):
    """Raised when a requested custom ship cannot be found and verified."""


def _operation_paused(should_abort=None) -> bool:
    """Check the shared lifecycle gate before every port-side action."""
    if should_abort is None:
        return False
    try:
        paused = bool(should_abort())
    except Exception:
        logger.exception("港口操作暂停门禁检查失败；按暂停处理")
        paused = True
    if paused:
        logger.info("[USER] 港口操作已被暂停门禁拦截，不再截图、切窗或点击")
    return paused


def _remember_selected_ship(ship_key):
    LAST_SELECTED_SHIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_SELECTED_SHIP_PATH.write_text(str(ship_key), encoding="utf-8")


def _last_confirmed_ship():
    try:
        return LAST_SELECTED_SHIP_PATH.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""


def _capture(hwnd=None):
    """Capture the game window in BGR order."""
    if not hwnd:
        raise RuntimeError("拒绝桌面截图：港口识别与 OCR 必须绑定游戏窗口")
    image = _WINDOW_CAPTURE.capture_window(hwnd)
    if image is None:
        raise CaptureFault(
            f"游戏窗口截取失败: {_WINDOW_CAPTURE.last_error or 'unknown'}"
        )
    return image


def _screen_origin(hwnd):
    rect = get_client_rect(hwnd)
    return rect["left"], rect["top"]


def _click_local(hwnd, point):
    if not ensure_game_window_foreground(hwnd):
        return False
    time.sleep(0.12)
    origin_x, origin_y = _screen_origin(hwnd)
    screen_x, screen_y = origin_x + point[0], origin_y + point[1]
    if physical_click(screen_x, screen_y, hwnd=hwnd):
        return True
    # Global pointer movement can be denied on a scaled multi-monitor desktop.
    # Do not emit an unchecked global click; target the already verified game
    # window with client messages instead.
    if hwnd:
        logger.warning("物理点击未派发，改用窗口消息点击: local=%s", point)
        return window_message_click(hwnd, screen_x, screen_y)
    return False


def _right_click_local(hwnd, point):
    """Open a verified game context menu without leaving the game window."""
    if not ensure_game_window_foreground(hwnd):
        return False
    time.sleep(0.12)
    origin_x, origin_y = _screen_origin(hwnd)
    screen_x, screen_y = origin_x + point[0], origin_y + point[1]
    if physical_click(screen_x, screen_y, hwnd=hwnd, button="right"):
        return True
    logger.warning("物理右键未派发，改用窗口消息右键: local=%s", point)
    return window_message_click(
        hwnd,
        screen_x,
        screen_y,
        button="right",
    )


def _click_region(hwnd, image, region: RelativeRegion):
    height, width = image.shape[:2]
    return _click_local(hwnd, region.center(width, height))


def _largest_color_center(image, region, hsv_ranges, minimum_ratio=0.04):
    """Return a button center within a relative ROI, or ``None``."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = region.pixels(width, height)
    area = image[y1:y2, x1:x2]
    if area.size == 0:
        return None
    hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
    mask = np.zeros(area.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if np.count_nonzero(mask) / max(mask.size, 1) < minimum_ratio:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < area.shape[0] * area.shape[1] * minimum_ratio:
        return None
    return x1 + x + w // 2, y1 + y + h // 2


def find_battle_button(hwnd=None, image=None):
    """Locate the blue or orange ``加入战斗`` button."""
    image = image if image is not None else _capture(hwnd)
    return _largest_color_center(
        image,
        PORT_BATTLE_BUTTON,
        [
            ((90, 55, 65), (135, 255, 255)),
            ((5, 55, 65), (30, 255, 255)),
        ],
        minimum_ratio=0.08,
    )


def click_battle(hwnd=None, image=None, backend=None):
    image = image if image is not None else _capture(hwnd)
    position = (
        port_battle_action_point(image, backend)
        if backend is not None
        else find_battle_button(hwnd, image)
    )
    if position is None:
        # The header text can be hidden for one or two frames while the port
        # artwork/card animation settles. The caller has already positively
        # classified PORT; use the guarded coloured button geometry rather
        # than rejecting an already selected ship on a transient OCR miss.
        position = find_battle_button(hwnd, image)
        if position is not None:
            logger.info("加入战斗文字暂不可读，使用已确认颜色按钮坐标: local=%s", position)
    if position is None:
        logger.warning("未识别到“加入战斗”文字按钮")
        return False
    logger.info("定位“加入战斗”: local=%s", position)
    clicked = _click_local(hwnd, position)
    if clicked:
        logger.info("已向“加入战斗”派发一次物理点击: local=%s", position)
    else:
        logger.warning("“加入战斗”物理点击未派发: local=%s", position)
    return clicked


def _observe_battle_entry(
    hwnd, vision, *, samples=8, interval=0.35, should_abort=None
):
    """Verify one Join Battle click without blindly clicking it again.

    A successful transition can be the loading page, the battle HUD, a queue
    page that the generic classifier calls ``unknown``, or disappearance of
    the actionable Join Battle button.  A retry is authorized only after the
    unchanged port and the still-actionable coloured button have both been
    observed repeatedly.  Capture uncertainty is treated as pending instead
    of failure because issuing another click could cancel an accepted queue.
    """
    stable_actionable_port = 0
    transition_frames = 0
    no_commander_frames = 0
    sample_count = max(1, int(samples))

    for sample in range(sample_count):
        if _operation_paused(should_abort):
            return False
        time.sleep(max(0.0, float(interval)))
        if _operation_paused(should_abort):
            return False
        try:
            image = _capture(hwnd)
        except Exception as error:
            logger.warning(
                "“加入战斗”已点击，但后续画面暂不可用；保持等待且不重复点击: %s",
                error,
            )
            return True

        state = vision.classify_screen(image)
        if state in {ScreenState.LOADING, ScreenState.BATTLE}:
            logger.info("“加入战斗”请求已确认: current=%s", state.value)
            return True

        # Loading artwork can resemble the dark no-commander dialog. Scene
        # classification owns the first decision, and the warning must remain
        # visible on two UNKNOWN frames before it may cancel battle entry.
        no_commander_detector = getattr(
            vision,
            "in_no_commander_confirmation",
            lambda _image: False,
        )
        no_commander_frames = (
            no_commander_frames + 1
            if state == ScreenState.UNKNOWN and no_commander_detector(image)
            else 0
        )
        if no_commander_frames >= 2:
            logger.warning(
                "连续两帧确认无指挥官拦截页；拒绝绕过提示进入战斗，返回港口重新复核指定舰船"
            )
            return False

        if state == ScreenState.PORT:
            transition_frames = 0
            if find_battle_button(hwnd, image) is None:
                logger.info("港口加入按钮已消失/禁用，确认正在进入匹配")
                return True
            stable_actionable_port += 1
            if stable_actionable_port >= 6:
                logger.warning(
                    "点击后连续 %s 帧仍为港口且加入按钮可用，允许下一轮安全重试",
                    stable_actionable_port,
                )
                return False
            continue

        stable_actionable_port = 0
        transition_frames += 1
        if state == ScreenState.UNKNOWN and transition_frames >= 2:
            logger.info("连续画面已离开港口，确认正在进入匹配/加载")
            return True
        if state in {
            ScreenState.ESCAPE_MENU,
            ScreenState.EXIT_CONFIRMATION,
            ScreenState.RESULTS,
        }:
            logger.warning(
                "加入战斗后进入非预期页面 %s；交由场景恢复且不重复点击",
                state.value,
            )
            return True

    if stable_actionable_port:
        logger.warning(
            "加入战斗点击后的港口状态尚未稳定；保持等待且不重复点击"
        )
    return True


def confirm_no_commander(hwnd=None, image=None, vision=None):
    """Fail closed on the post-Join no-commander warning.

    Commander recovery belongs exclusively to
    :func:`ensure_selected_ship_commander`, where the requested ship name and
    the selected detail panel can both be verified.  The modal shown after
    ``加入战斗`` does not prove which carousel ship produced it, so clicking its
    continue button would violate the selected-ship interlock.
    """
    vision = vision or Vision()
    image = image if image is not None else _capture(hwnd)
    if not vision.in_no_commander_confirmation(image):
        return False
    logger.warning("无指挥官拦截页已确认；禁止自动点击继续")
    return False


def open_mode_selector(hwnd=None, vision=None, *, backend=None):
    """Open the battle-mode selector only while the port is positively seen."""
    image = _capture(hwnd)
    vision = vision or Vision()
    if vision.classify_screen(image) != ScreenState.PORT:
        logger.warning("当前不是港口，拒绝点击模式选择器")
        return False
    if backend is not None:
        point = port_mode_selector_action_point(image, backend)
        return point is not None and _click_local(hwnd, point)
    return _click_region(hwnd, image, PORT_MODE_SELECTOR)


def _mode_header_scores(image):
    """Return screenshot-based color scores for the current port mode."""
    height, width = image.shape[:2]
    # The emblem occupies the left portion of the selector at every captured
    # port resolution.  Excluding the label avoids dependence on font glyphs.
    x1 = int(width * 0.532)
    y1 = 0
    x2 = int(width * 0.575)
    y2 = max(1, int(height * 0.055))
    area = image[y1:y2, x1:x2]
    if area.size == 0:
        return {"asymmetric": 0.0, "cooperative": 0.0}
    hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
    asymmetric = cv2.inRange(
        hsv, ASYMMETRIC_PURPLE_LOWER, ASYMMETRIC_PURPLE_UPPER
    )
    # The co-op emblem is desaturated steel/teal.  It is deliberately only a
    # weak score; confirmation also checks that asymmetric purple is absent.
    cooperative = cv2.inRange(
        hsv, np.array([70, 20, 45]), np.array([125, 170, 230])
    )
    total = max(asymmetric.size, 1)
    return {
        "asymmetric": np.count_nonzero(asymmetric) / total,
        "cooperative": np.count_nonzero(cooperative) / total,
    }


def _detect_port_mode_ocr(image, backend):
    """Read the exact current-mode label from the port header.

    Random and co-op emblems share enough desaturated teal pixels that colour
    alone can confuse them. The visible Chinese label is unambiguous, so a
    configured OCR backend is authoritative and deliberately fails closed.
    """
    if image is None or image.size == 0 or backend is None:
        return None
    height, width = image.shape[:2]
    searches = [
        image[:max(1, int(height * .075)), int(width * .525):int(width * .680)],
        image[:max(1, int(height * .32)), :],
    ]
    for header in searches:
        try:
            tokens = backend.recognize(header) or ()
        except Exception:
            logger.exception("港口战斗模式 OCR 失败")
            continue
        text = "".join(
            unicodedata.normalize("NFKC", str(token.text or ""))
            for token in tokens
            if float(getattr(token, "confidence", 0.0)) >= 0.65
        ).replace(" ", "")
        matches = set()
        if "非对称战斗" in text or "非对称作战" in text:
            matches.add("asymmetric")
        if "联合作战" in text:
            matches.add("cooperative")
        if "随机战" in text:
            # Random is intentionally not an automation target.
            matches.add("random")
        if len(matches) == 1:
            return matches.pop()
        if len(matches) > 1:
            return None
    return None


def detect_port_mode(image, backend=None):
    """Classify the selected mode, preferring the exact visible label."""
    if backend is not None:
        return _detect_port_mode_ocr(image, backend)
    scores = _mode_header_scores(image)
    if scores["asymmetric"] >= 0.018:
        return "asymmetric"
    if scores["cooperative"] >= 0.022 and scores["asymmetric"] < 0.006:
        return "cooperative"
    return None


def _find_asymmetric_card(image):
    """Locate the largest purple mode emblem on the battle-type screen."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = BATTLE_TYPE_SEARCH_AREA.pixels(width, height)
    area = image[y1:y2, x1:x2]
    if area.size == 0:
        return None
    hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, ASYMMETRIC_PURPLE_LOWER, ASYMMETRIC_PURPLE_UPPER
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    frame_area = width * height
    for contour in contours:
        area_size = cv2.contourArea(contour)
        if not frame_area * 0.00012 <= area_size <= frame_area * 0.02:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) < min(width, height) * 0.025:
            continue
        candidates.append((area_size, x1 + x + w // 2, y1 + y + h // 2))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


def _selector_title_seen(image, backend) -> bool:
    if image is None or image.size == 0 or backend is None:
        return False
    height, width = image.shape[:2]
    searches = [image[:max(1, int(height * .13)), int(width * .20):int(width * .80)],
                image[:max(1, int(height * .40)), :]]
    for heading in searches:
        try:
            tokens = backend.recognize(heading) or ()
        except Exception:
            logger.exception("战斗类型页面标题 OCR 失败")
            continue
        text = "".join(
            unicodedata.normalize("NFKC", str(token.text or ""))
            for token in tokens
            if float(getattr(token, "confidence", 0.0)) >= 0.65
        ).replace(" ", "")
        if "选择一种战斗模式" in text:
            return True
    return False


def _battle_survey_ocr(image, backend):
    if backend is None or image is None or getattr(image, "size", 0) == 0:
        return (), (0, 0)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = BATTLE_SURVEY_AREA.pixels(width, height)
    dialog = image[y1:y2, x1:x2]
    if dialog.size == 0:
        return (), (x1, y1)
    try:
        tokens = tuple(backend.recognize(dialog) or ())
        question, _choices, action = _battle_survey_evidence(tokens)
        if question and action:
            return tokens, (x1, y1)
        wider_tokens = tuple(backend.recognize(image) or ())
        wider_question, wider_choices, wider_action = _battle_survey_evidence(wider_tokens)
        if wider_question and (wider_action or wider_choices >= 2):
            return wider_tokens, (0, 0)
        return tokens, (x1, y1)
    except (TypeError, ValueError):
        logger.debug("战斗评价页面 OCR 返回了非序列结果")
    except Exception:
        logger.debug("战斗评价页面 OCR 失败", exc_info=True)
    return (), (x1, y1)


def _battle_survey_evidence(tokens) -> tuple[bool, int, bool]:
    text = "".join(
        unicodedata.normalize("NFKC", str(token.text or ""))
        for token in tokens
        if float(getattr(token, "confidence", 0.0)) >= 0.55
    ).replace(" ", "")
    question_seen = any(
        phrase in text
        for phrase in (
            "满意度如何",
            "战斗评价",
            "评价本场战斗",
            "评价这场战斗",
            "您对这场战斗",
            "您觉得这场战斗",
        )
    ) or (
        "刚刚进行" in text and "这场战斗" in text and "满意" in text
    )
    choice_hits = sum(
        label in text
        for label in ("非常不满意", "不满意", "一般", "满意", "非常满意")
    )
    action_seen = any(
        label in text
        for label in ("跳过", "关闭", "暂不评价", "以后再说", "取消")
    )
    return question_seen, choice_hits, action_seen


def is_battle_survey_page(image, backend=None) -> bool:
    """Recognize the optional post-battle satisfaction survey by OCR.

    The dialog is deliberately not detected from its dark backdrop or button
    colours: both are common on loading and confirmation pages.  Production
    therefore requires the survey-specific question plus either its Close
    action or multiple satisfaction choices before Esc recovery is allowed.
    """
    tokens, _origin = _battle_survey_ocr(image, backend)
    question_seen, choice_hits, action_seen = _battle_survey_evidence(tokens)
    return bool(question_seen and (action_seen or choice_hits >= 2))


def battle_survey_dismiss_point(image, backend=None):
    """Return the OCR-derived Skip/Close action only on a verified survey."""
    tokens, (origin_x, origin_y) = _battle_survey_ocr(image, backend)
    question_seen, choice_hits, action_seen = _battle_survey_evidence(tokens)
    if not question_seen or not (action_seen or choice_hits >= 2):
        return None
    labels = ("暂不评价", "以后再说", "跳过", "关闭", "取消")
    candidates = []
    for token in tokens:
        if float(getattr(token, "confidence", 0.0)) < 0.55:
            continue
        text = unicodedata.normalize("NFKC", str(token.text or "")).replace(" ", "")
        if not any(label in text for label in labels):
            continue
        geometry = _token_geometry(token)
        if geometry is None:
            continue
        x1, y1, x2, y2 = geometry
        candidates.append(
            (
                float(getattr(token, "confidence", 0.0)),
                (int(round(origin_x + (x1 + x2) / 2)), int(round(origin_y + (y1 + y2) / 2))),
            )
        )
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def dismiss_battle_survey(
    hwnd,
    image,
    *,
    backend=None,
    should_abort=None,
    escape_action=None,
) -> bool:
    """Dismiss a positively identified survey without choosing a rating."""
    if _operation_paused(should_abort) or not is_battle_survey_page(image, backend):
        return False
    if escape_action is not None:
        if _operation_paused(should_abort):
            return False
        logger.info("已确认战斗评价页，按 Esc 跳过评价")
        try:
            if escape_action() is not False:
                return True
        except RuntimeError as error:
            logger.info("评价页 Esc 暂未派发: %s", error)
    point = battle_survey_dismiss_point(image, backend)
    if point is not None:
        if _operation_paused(should_abort):
            return False
        logger.info("识别到战斗评价页，点击跳过/关闭: local=%s", point)
        return bool(_click_local(hwnd, point))
    return False


def in_battle_type_selector(image, backend=None):
    """Confirm that the full battle-type card page is still open.

    The old port-header colour check also returned ``cooperative`` on this
    screen, causing the selector to be treated as an already verified port.
    The asymmetric purple card is a stable page-specific anchor and is absent
    from normal port screenshots.
    """
    if backend is not None:
        # Production has OCR available.  Fail closed on the exact page title
        # instead of allowing a purple carousel/commander tile in port to be
        # mistaken for the selector.  The colour anchor remains only for
        # fixture/legacy callers that explicitly have no OCR backend.
        return _selector_title_seen(image, backend)
    point = _find_asymmetric_card(image)
    if point is None:
        return False
    height, width = image.shape[:2]
    # On the full selector the asymmetric card is the lower-right card of the
    # six primary mode choices. Purple commander panels and carousel cards in
    # the normal port can otherwise look like the same emblem.
    return bool(
        width * 0.50 <= point[0] <= width * 0.70
        and height * 0.43 <= point[1] <= height * 0.66
    )


def _find_mode_card_from_ocr(image, requested_mode, backend):
    """Locate the requested card by its rendered label at any UI scale."""
    if image is None or image.size == 0 or backend is None:
        return None
    height, width = image.shape[:2]
    aliases = (
        ("联合作战",)
        if requested_mode == "cooperative"
        else ("非对称战斗", "非对称作战")
    )
    searches = [(int(width * .18), int(height * .20), int(width * .82), int(height * .68)),
                (0, int(height * .13), width, int(height * .93))]
    for left, top, right, bottom in searches:
        point = _action_point_in_crop(
            image, backend, tuple(map(_normalize_ship_name, aliases)), (),
            left, top, right, bottom, .70, (1.0, 1.45), False,
        )
        if point is not None:
            return point
    return None


def select_mode_from_screen(hwnd, requested_mode, image=None, backend=None):
    """Select one of the two supported PvE cards on the opened menu."""
    image = image if image is not None else _capture(hwnd)
    if not in_battle_type_selector(image, backend=backend):
        logger.warning("未确认战斗模式选择页，拒绝按固定卡片位置点击")
        return False
    height, width = image.shape[:2]
    point = _find_mode_card_from_ocr(image, requested_mode, backend)
    if backend is not None and point is None:
        logger.warning("模式选择页 OCR 未定位到目标卡片: %s", requested_mode)
        return False
    if point is None:
        if requested_mode == "cooperative":
            point = BATTLE_TYPE_COOPERATIVE_CARD.center(width, height)
        else:
            point = _find_asymmetric_card(image)
            if point is None:
                logger.warning("未在战斗类型页识别到非对称作战紫色徽标")
                return False
    logger.info("按截图定位战斗模式 %s: local=%s", requested_mode, point)
    return _click_local(hwnd, point)


def ensure_requested_mode(
    hwnd=None,
    requested_mode="asymmetric",
    vision=None,
    *,
    backend=None,
    should_abort=None,
):
    """Select the configured mode and verify it again after returning to port."""
    requested_mode = (requested_mode or "asymmetric").strip().lower()
    if requested_mode not in SUPPORTED_MODES:
        logger.error("不支持的战斗模式: %s", requested_mode)
        return False
    vision = vision or Vision()
    for attempt in range(1, 4):
        if _operation_paused(should_abort):
            return False
        image = _capture(hwnd)
        selector_open = in_battle_type_selector(image, backend=backend)
        current = (
            None
            if selector_open
            else detect_port_mode(image, backend=backend)
        )
        logger.info(
            "模式校验 (%s/3): 页面=%s 当前=%s 目标=%s",
            attempt,
            "模式选择" if selector_open else "港口",
            current or "不确定",
            requested_mode,
        )
        if (
            not selector_open
            and vision.classify_screen(image) == ScreenState.PORT
            and current == requested_mode
        ):
            logger.info("港口右侧当前模式已确认: %s", requested_mode)
            return True

        if not selector_open:
            if _operation_paused(should_abort):
                return False
            if vision.classify_screen(image) != ScreenState.PORT:
                logger.warning("当前不是港口或模式选择页，本轮不执行模式点击")
                return False
            mode_point = (
                port_mode_selector_action_point(image, backend)
                if backend is not None
                else None
            )
            if backend is not None and mode_point is None:
                logger.warning("港口页 OCR 未定位到当前战斗模式入口")
                return False
            clicked = (
                _click_local(hwnd, mode_point)
                if mode_point is not None
                else _click_region(hwnd, image, PORT_MODE_SELECTOR)
            )
            if not clicked:
                continue
            time.sleep(1.0)
            image = _capture(hwnd)
            selector_open = in_battle_type_selector(image, backend=backend)

        if not selector_open:
            logger.warning("点击港口右侧模式入口后，未确认模式选择页")
            time.sleep(0.5)
            continue
        if not select_mode_from_screen(
            hwnd,
            requested_mode,
            image=image,
            backend=backend,
        ):
            time.sleep(0.5)
            continue

        # Selection animation and port restoration are asynchronous. Require
        # both selector closure and a matching right-side port emblem.
        for verification in range(6):
            if _operation_paused(should_abort):
                return False
            time.sleep(0.5 if verification else 1.0)
            if _operation_paused(should_abort):
                return False
            confirmation = _capture(hwnd)
            if in_battle_type_selector(confirmation, backend=backend):
                continue
            if vision.classify_screen(confirmation) != ScreenState.PORT:
                continue
            selected = detect_port_mode(confirmation, backend=backend)
            if selected == requested_mode:
                logger.info("模式选择完成并通过港口复核: %s", requested_mode)
                return True
            logger.warning(
                "返回港口但模式不匹配: 识别=%s 目标=%s",
                selected,
                requested_mode,
            )
            break
    logger.error("连续3次未能选择并确认战斗模式: %s", requested_mode)
    return False


def dismiss_port_dialog(hwnd=None, image=None):
    """Dismiss a centered port notice only when an X-shaped close mark exists."""
    image = image if image is not None else _capture(hwnd)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = PORT_DIALOG_CLOSE.pixels(width, height)
    area = image[y1:y2, x1:x2]
    if area.size == 0:
        return False
    gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_ratio = np.count_nonzero(edges) / max(edges.size, 1)
    if edge_ratio < 0.025:
        return False
    logger.info("关闭港口中央提示框")
    return _click_region(hwnd, image, PORT_DIALOG_CLOSE)


def _load_ship_name_template(ship_key):
    return cv2.imread(str(SHIP_TEMPLATE_DIR / SHIP_NAME_TEMPLATES[ship_key]))


def _load_selected_ship_name_template(ship_key):
    return cv2.imread(
        str(SHIP_TEMPLATE_DIR / SELECTED_SHIP_NAME_TEMPLATES[ship_key]),
        cv2.IMREAD_GRAYSCALE,
    )


def _gold_name_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([12, 70, 105]), np.array([42, 255, 255]))


def _tighten_mask(mask, padding=2):
    """Remove template whitespace so similar card backgrounds cannot dominate."""
    rows, columns = np.where(mask > 0)
    if not len(rows) or not len(columns):
        return mask
    top = max(0, int(rows.min()) - padding)
    bottom = min(mask.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(mask.shape[1], int(columns.max()) + padding + 1)
    return mask[top:bottom, left:right]


def _normalize_ship_name(value):
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _custom_ship_ocr_names(full_name):
    """Return the configured name plus known in-game OCR/name variants.

    The Chinese client renders the ship Zieten as ``齐滕``.  Older presets
    (and user-entered values) commonly contain the homophonous ``齐藤``.  The
    two strings refer to the same visible card here; accepting the pair only
    in the custom-ship path keeps exact matching for every other ship intact.
    """
    wanted = _normalize_ship_name(full_name)
    names = {wanted} if wanted else set()
    if wanted in {"齐藤", "齐滕"}:
        names.update({"齐藤", "齐滕"})
    return names


def _token_geometry(token):
    if not token.box:
        return None
    xs = [point[0] for point in token.box]
    ys = [point[1] for point in token.box]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _ocr_line_candidates(tokens, minimum_confidence=0.60):
    """Yield individual and adjacent OCR tokens as semantic text boxes.

    OCR engines may return ``加入战斗`` as one token at 2K and as ``加入`` +
    ``战斗`` at 1080p or a different UI scale. Joining only nearby tokens on
    the same baseline handles both cases without joining separate buttons.
    """
    entries = []
    for token in tokens or ():
        geometry = _token_geometry(token)
        confidence = float(getattr(token, "confidence", 0.0))
        text = _normalize_ship_name(getattr(token, "text", ""))
        if confidence < minimum_confidence or geometry is None or not text:
            continue
        x1, y1, x2, y2 = (float(value) for value in geometry)
        if x2 <= x1 or y2 <= y1:
            continue
        entries.append(
            {
                "text": text,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "center_y": (y1 + y2) / 2.0,
                "height": y2 - y1,
            }
        )

    lines = []
    for entry in sorted(
        entries,
        key=lambda item: (item["center_y"], item["box"][0]),
    ):
        compatible = None
        compatible_distance = None
        for line in lines:
            distance = abs(entry["center_y"] - line["center_y"])
            tolerance = max(entry["height"], line["height"]) * 0.65
            if distance <= tolerance and (
                compatible_distance is None or distance < compatible_distance
            ):
                compatible = line
                compatible_distance = distance
        if compatible is None:
            lines.append(
                {
                    "tokens": [entry],
                    "center_y": entry["center_y"],
                    "height": entry["height"],
                }
            )
        else:
            compatible["tokens"].append(entry)
            count = len(compatible["tokens"])
            compatible["center_y"] = (
                compatible["center_y"] * (count - 1) + entry["center_y"]
            ) / count
            compatible["height"] = max(compatible["height"], entry["height"])

    candidates = []
    for line in lines:
        ordered = sorted(line["tokens"], key=lambda item: item["box"][0])
        for start in range(len(ordered)):
            group = []
            for end in range(start, min(len(ordered), start + 4)):
                current = ordered[end]
                if group:
                    previous = group[-1]
                    gap = current["box"][0] - previous["box"][2]
                    max_gap = max(current["height"], previous["height"]) * 2.0
                    if gap > max_gap:
                        break
                group.append(current)
                candidates.append(
                    {
                        "text": "".join(item["text"] for item in group),
                        "confidence": min(item["confidence"] for item in group),
                        "box": (
                            min(item["box"][0] for item in group),
                            min(item["box"][1] for item in group),
                            max(item["box"][2] for item in group),
                            max(item["box"][3] for item in group),
                        ),
                    }
                )
    return candidates


def _verified_action_point(
    image,
    backend,
    wanted,
    *,
    rejected=(),
    region=None,
    minimum_confidence=0.60,
    retry_scales=(1.0, 1.45),
    exact=False,
    expand_search=True,
    allow_short_expanded=False,
):
    """Locate semantic text and map its OCR box back to framebuffer pixels."""
    if (
        image is None
        or not hasattr(image, "size")
        or image.size == 0
        or backend is None
    ):
        return None
    wanted = tuple(_normalize_ship_name(value) for value in wanted)
    rejected = tuple(_normalize_ship_name(value) for value in rejected)
    height, width = image.shape[:2]
    if region is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        left, top, right, bottom = region.pixels(width, height)
    searches = [(left, top, right, bottom)]
    if region is not None and expand_search:
        # These are search envelopes, never click coordinates. The first ROI
        # remains the fast path; moving/resizing the UI may require a wider
        # header, footer or dialog envelope.
        if region.top < .15:
            wider = (0, 0, width, min(height, int(height * .32)))
        elif region.bottom >= .95:
            wider = (0, int(height * .45), width, height)
        else:
            wider = (max(0, left - int(width * .23)),
                     max(0, top - int(height * .23)),
                     min(width, right + int(width * .23)),
                     min(height, bottom + int(height * .23)))
        if wider != searches[0]:
            searches.append(wider)

    for search_index, (left, top, right, bottom) in enumerate(searches):
        point = _action_point_in_crop(image, backend, wanted, rejected,
                                      left, top, right, bottom,
                                      minimum_confidence, retry_scales, exact,
                                      allow_short=(search_index == 0 or allow_short_expanded))
        if point is not None:
            return point
    return None


def _action_point_in_crop(image, backend, wanted, rejected, left, top, right,
                          bottom, minimum_confidence, retry_scales, exact,
                          *, allow_short=True):
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None

    for scale in retry_scales:
        scale = max(0.1, float(scale))
        sample = crop
        if abs(scale - 1.0) > 1e-3:
            sample = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
            )
        try:
            candidates = _ocr_line_candidates(
                backend.recognize(sample),
                minimum_confidence=minimum_confidence,
            )
        except Exception:
            logger.exception("界面动作 OCR 失败: scale=%.2f", scale)
            continue
        matches = []
        for candidate in candidates:
            text = candidate["text"]
            if any(value and value in text for value in rejected):
                continue
            matched_values = [
                value
                for value in wanted
                if value and (text == value if exact else value in text)
            ]
            if not matched_values:
                continue
            if not allow_short and all(len(value) < 3 for value in matched_values):
                # A lone "港口" in a larger search can be a page title. Only
                # complete action labels authorize the expanded search path.
                continue
            x1, y1, x2, y2 = candidate["box"]
            exact_bonus = (
                0.20 if any(text == value for value in matched_values) else 0.0
            )
            length_bonus = min(len(text), 12) * 0.001
            matches.append(
                (
                    candidate["confidence"] + exact_bonus + length_bonus,
                    (
                        left + int(round((x1 + x2) / (2.0 * scale))),
                        top + int(round((y1 + y2) / (2.0 * scale))),
                    ),
                    max(12.0, (y2 - y1) / scale * 1.5),
                )
            )
        if matches:
            best = max(matches, key=lambda item: item[0])
            if any(
                best[0] - other[0] <= .05
                and math.dist(best[1], other[1]) > max(best[2], other[2])
                for other in matches
            ):
                logger.debug("同一动作出现多个分离且可信度相近的候选，等待新画面")
                return None
            return best[1]
    return None


def port_battle_action_point(image, backend=None):
    """Locate the rendered ``加入战斗`` label at any resolution/UI scale."""
    backend = backend or RapidOcrBackend()
    return _verified_action_point(
        image,
        backend,
        ("加入战斗",),
        rejected=("继续战斗", "离开战斗"),
        region=PORT_BATTLE_TEXT_AREA,
    )


def port_mode_selector_action_point(image, backend=None):
    """Locate the current battle-mode label in the port header."""
    backend = backend or RapidOcrBackend()
    return _verified_action_point(
        image,
        backend,
        (
            "联合作战",
            "非对称战斗",
            "非对称作战",
            "随机战",
            "随机战斗",
            "排位战",
            "排位战斗",
            "标准战斗",
            "行动",
            "对决",
        ),
        rejected=("加入战斗",),
        region=PORT_MODE_TEXT_AREA,
        minimum_confidence=0.55,
        allow_short_expanded=True,
    )


def results_requeue_action_point(image, backend=None):
    backend = backend or RapidOcrBackend()
    return _verified_action_point(
        image,
        backend,
        ("继续战斗", "下一场战斗", "续战斗"),
        rejected=("返回港口", "回到港口", "港口"),
        region=RESULTS_REQUEUE_TEXT_AREA,
    )


def results_return_to_port_action_point(image, backend=None):
    backend = backend or RapidOcrBackend()
    return _verified_action_point(
        image,
        backend,
        ("返回港口", "回到港口", "港口"),
        rejected=("继续战斗", "续战斗"),
        region=RESULTS_RETURN_TEXT_AREA,
    )


def quick_exit_menu_action_point(image, backend=None):
    """Find ``离开战斗`` but never the port menu's ``退出游戏`` action."""
    backend = backend or RapidOcrBackend()
    return _verified_action_point(
        image,
        backend,
        ("离开战斗", "退出战斗"),
        rejected=("退出游戏", "关闭游戏", "回到战斗"),
    )


def quick_death_action_point(image, backend=None):
    """Find Continue Battle on the page reached *after* death-screen Esc.

    Callers must first confirm the sunk-ship hint and dispatch Esc. The
    same words before Esc can mean spectating the current match instead.
    """
    backend = backend or RapidOcrBackend()
    return _verified_action_point(
        image, backend, ("继续战斗",),
        rejected=("返回港口", "回到港口", "退出游戏"),
        region=QUICK_DEATH_ACTION_AREA, exact=True,
    )


def quick_death_leave_hint_visible(image, backend=None):
    """Require death text and the game's own Esc leave-battle hint together."""
    backend = backend or RapidOcrBackend()
    if image is None or image.size == 0:
        return False
    height, width = image.shape[:2]
    left, top, right, bottom = QUICK_DEATH_LEAVE_AREA.pixels(width, height)
    try:
        candidates = _ocr_line_candidates(backend.recognize(image[top:bottom, left:right]))
    except Exception:
        logger.debug("死亡离战提示 OCR 不可用", exc_info=True)
        return False
    texts = [candidate["text"] for candidate in candidates]
    return (
        any("摧毁" in text or "战舰已" in text for text in texts)
        and any("离开战斗" in text for text in texts)
    )


def _port_exit_confirmation_tokens(image, backend):
    """Return OCR tokens only for the exact port ``退出游戏?`` dialog."""
    if image is None or not hasattr(image, "size") or image.size == 0:
        return []
    backend = backend or RapidOcrBackend()
    try:
        tokens = list(backend.recognize(image) or ())
    except Exception:
        logger.warning("港口退出确认框 OCR 失败；拒绝点击", exc_info=True)
        return []
    normalized = [
        (_normalize_ship_name(token.text), token)
        for token in tokens
        if float(getattr(token, "confidence", 0.0)) >= 0.55
    ]
    combined = "".join(text for text, _token in normalized)
    if not (
        _normalize_ship_name("确认") in combined
        and any(
            label in combined
            for label in (
                _normalize_ship_name("退出游戏"),
                _normalize_ship_name("关闭游戏"),
            )
        )
    ):
        return []
    return [token for _text, token in normalized]


def port_exit_confirmation_no_action_point(image, backend=None):
    """Locate the exact ``否`` action on a verified port-exit dialog."""
    tokens = _port_exit_confirmation_tokens(image, backend)
    if not tokens:
        return None
    height, width = image.shape[:2]
    headings = [_token_geometry(token) for token in tokens
                if any(word in _normalize_ship_name(token.text)
                       for word in ("退出游戏", "关闭游戏"))]
    headings = [box for box in headings if box is not None]
    candidates = []
    for token in tokens:
        if _normalize_ship_name(token.text) != _normalize_ship_name("否"):
            continue
        geometry = _token_geometry(token)
        if geometry is None:
            continue
        x1, y1, x2, y2 = geometry
        center = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        # The dialog title and cancel action must belong to the same local
        # group. Their glyph size supplies the bounds when the modal moves.
        nearby_heading = any(abs(center[0] - (box[0] + box[2]) / 2) <= max(box[2] - box[0], box[3] - box[1]) * 3
                   and -1 <= (center[1] - box[3]) / max(box[3] - box[1], 1) <= 8
                   for box in headings)
        # Older OCR adapters omit title boxes. Keep the established bounded
        # cancel action only after the exact quit-dialog text was verified.
        if not headings:
            left, top, right, bottom = PORT_EXIT_CONFIRM_ACTION_AREA.pixels(width, height)
            nearby_heading = left <= center[0] <= right and top <= center[1] <= bottom
        if not nearby_heading:
            continue
        candidates.append((float(getattr(token, "confidence", 0.0)), center))
    return None if not candidates else max(candidates, key=lambda item: item[0])[1]


def is_port_exit_confirmation_page(image, backend=None) -> bool:
    return port_exit_confirmation_no_action_point(image, backend) is not None


def dismiss_port_exit_confirmation(
    hwnd,
    image,
    *,
    backend=None,
    should_abort=None,
) -> bool:
    """Cancel a verified request to quit the client without closing the game."""
    if _operation_paused(should_abort):
        return False
    point = port_exit_confirmation_no_action_point(image, backend)
    if point is None:
        return False
    logger.info("识别到港口“退出游戏”确认框，点击“否”: local=%s", point)
    return bool(_click_local(hwnd, point))


def is_early_exit_confirmation_page(image, backend=None) -> bool:
    """Public OCR gate for the battle-only early-exit confirmation."""
    if backend is None:
        return False
    return _is_early_exit_confirmation(image, backend)


def _is_early_exit_confirmation(image, backend):
    try:
        tokens = list(backend.recognize(image) or ())
    except Exception:
        logger.warning("提前退出确认页 OCR 失败；拒绝点击确认按钮", exc_info=True)
        return False
    texts = [
        _normalize_ship_name(token.text)
        for token in tokens
        if float(getattr(token, "confidence", 0.0)) >= 0.55
    ]
    combined = "".join(texts)
    return (
        _normalize_ship_name("提前退出战斗") in combined
        and _normalize_ship_name("离开战斗") in combined
    )


def quick_exit_confirmation_yes_action_point(image, backend=None):
    """Locate the exact ``是`` only on a verified early-exit dialog."""
    backend = backend or RapidOcrBackend()
    if not _is_early_exit_confirmation(image, backend):
        return None
    return _verified_action_point(
        image,
        backend,
        ("是",),
        rejected=("否",),
        region=QUICK_EXIT_CONFIRM_ACTION_AREA,
        minimum_confidence=0.50,
        exact=True,
        allow_short_expanded=True,
    )


def daily_reward_claim_point(image, backend=None):
    """Return the verified ``领取`` action on the daily login reward page.

    The page is positively identified from its heading before any action is
    considered. This keeps the orange-button fallback safe when seasonal
    artwork or reward-card positions change between game updates.
    """
    if image is None or not hasattr(image, "size") or image.size == 0:
        return None
    backend = backend or RapidOcrBackend()
    try:
        tokens = list(backend.recognize(image) or ())
    except Exception:
        logger.warning("每日奖励 OCR 失败；本轮不点击领取", exc_info=True)
        return None
    normalized = [
        (_normalize_ship_name(token.text), token)
        for token in tokens
        if float(token.confidence) >= 0.58
    ]
    combined = "".join(text for text, _token in normalized)
    daily_evidence = any(
        keyword in combined
        for keyword in (
            _normalize_ship_name("每日奖励"),
            _normalize_ship_name("每日补给"),
            _normalize_ship_name("每日登录"),
            _normalize_ship_name("登录奖励"),
            "dailyrewards",
            "dailyreward",
        )
    )
    if not daily_evidence:
        return None

    height, width = image.shape[:2]
    actions = []
    for text, token in normalized:
        if text.startswith(_normalize_ship_name("已领取")):
            continue
        if not any(
            keyword in text
            for keyword in (
                _normalize_ship_name("领取"),
                _normalize_ship_name("收取"),
                _normalize_ship_name("收集您的奖励"),
                _normalize_ship_name("收集奖励"),
                _normalize_ship_name("收集"),
                "claim",
            )
        ):
            continue
        geometry = _token_geometry(token)
        if geometry is None:
            continue
        x1, y1, x2, y2 = geometry
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        # Explanatory text near the heading often says ``有3天时间来领取奖励``.
        # It is not a button.  Claim actions are confined to the lower content
        # area and away from the window edges.
        if center[1] < height * 0.45:
            continue
        if not width * 0.20 <= center[0] <= width * 0.80:
            continue
        # Prefer the large lower-centre action; headings and already-claimed
        # day cards are not valid click targets.
        score = float(token.confidence)
        score += 0.15
        score += 0.10 if width * 0.25 <= center[0] <= width * 0.75 else 0.0
        actions.append((score, center))
    if actions:
        return max(actions, key=lambda item: item[0])[1]

    # OCR can miss white text on the glowing orange button. The visual
    # fallback is allowed only after the page heading was positively read.
    return _largest_color_center(
        image,
        # Daily-reward layouts vary, but the primary claim action remains in
        # the lower centre.  Keeping this fallback narrow prevents a large
        # orange reward card or seasonal illustration from becoming a click
        # target when OCR reads the heading but misses the button text.
        RelativeRegion(0.30, 0.60, 0.70, 0.94),
        [((5, 70, 80), (35, 255, 255))],
        minimum_ratio=0.012,
    )


def is_daily_reward_page(image, backend=None):
    return daily_reward_claim_point(image, backend) is not None


def claim_daily_reward(
    hwnd,
    image=None,
    *,
    backend=None,
    should_abort=None,
    confirm_action=None,
    close_action=None,
):
    """Claim and close the first-login daily reward after OCR verification.

    Mouse input is preferred because OCR provides the exact button centre.
    ``confirm_action`` is a narrow Enter-key fallback for game clients that
    reject synthetic mouse messages.  ``close_action`` is invoked only after
    the claim button disappeared (or a fresh verification capture was not
    available), so a missed click cannot silently close an unclaimed reward.
    """
    if _operation_paused(should_abort):
        return False
    image = image if image is not None else _capture(hwnd)
    point = daily_reward_claim_point(image, backend)
    if point is None:
        return False
    logger.info("识别到每日奖励页面，点击领取: local=%s", point)
    dispatched = _click_local(hwnd, point)
    used_keyboard_fallback = False
    if not dispatched and confirm_action is not None:
        if _operation_paused(should_abort) or not ensure_game_window_foreground(hwnd):
            return False
        try:
            logger.warning("每日奖励鼠标点击未派发，改用 Enter 确认领取")
            confirm_action()
            dispatched = True
            used_keyboard_fallback = True
        except Exception:
            logger.warning("每日奖励 Enter 领取兜底失败", exc_info=True)
    if not dispatched:
        return False

    time.sleep(0.85)
    if _operation_paused(should_abort):
        logger.info("每日奖励领取后进入暂停；恢复时再识别并关闭页面")
        return True

    # Re-capture before Esc. When the button is still present, the game did
    # not accept the first mouse event even if Windows reported it dispatched.
    # Retry once with Enter, then keep the page open if it still cannot be
    # confirmed. This is safer than closing and forfeiting the reward.
    # Default to "still visible" so a failed re-capture (or a later Esc that
    # cannot be verified) keeps the reward page open instead of closing an
    # unconfirmed claim and forfeiting the reward.
    claim_still_visible = True
    try:
        refreshed = _capture(hwnd)
        claim_still_visible = daily_reward_claim_point(refreshed, backend) is not None
    except Exception:
        logger.info("领取后暂时无法复核每日奖励页面；保留页面等待下一轮重试")

    if claim_still_visible and confirm_action is not None and not used_keyboard_fallback:
        if _operation_paused(should_abort) or not ensure_game_window_foreground(hwnd):
            return True
        try:
            logger.warning("领取按钮仍可见，使用 Enter 再确认一次")
            confirm_action()
            time.sleep(0.85)
            refreshed = _capture(hwnd)
            claim_still_visible = (
                daily_reward_claim_point(refreshed, backend) is not None
            )
        except Exception:
            logger.warning("每日奖励二次确认失败", exc_info=True)

    if claim_still_visible:
        logger.warning("领取按钮仍然可见，保留页面等待下一轮重试，不发送 Esc")
        return False

    if close_action is not None:
        if _operation_paused(should_abort):
            return True
        if not ensure_game_window_foreground(hwnd):
            logger.warning("领取已派发，但游戏未在前台；暂不发送 Esc")
            return True
        try:
            close_action()
            logger.info("每日奖励已领取，已按 Esc 关闭奖励页面")
            time.sleep(0.45)
        except Exception:
            # The reward action itself succeeded. The caller's scene recovery
            # will see the remaining overlay and retry Esc without re-selecting
            # a ship or starting a new round.
            logger.warning("每日奖励已领取，但 Esc 关闭页面失败", exc_info=True)
    return True


def _ocr_name_matches(
    image,
    full_name,
    backend,
    minimum_confidence=0.60,
    *,
    allow_tier_prefix=False,
    wanted_names=None,
):
    """Return exact normalized OCR matches, joining adjacent split tokens."""
    wanted_names = {
        _normalize_ship_name(name)
        for name in (wanted_names or (full_name,))
        if _normalize_ship_name(name)
    }
    if not wanted_names:
        return []
    try:
        tokens = list(backend.recognize(image) or ())
    except Exception:
        logger.warning("舰船名称 OCR 失败；本轮拒绝确认选船", exc_info=True)
        return []
    items = []
    for token in tokens:
        geometry = _token_geometry(token)
        if token.confidence < minimum_confidence or geometry is None:
            continue
        x1, y1, x2, y2 = geometry
        items.append(
            {
                "text": token.text,
                "normalized": _normalize_ship_name(token.text),
                "confidence": float(token.confidence),
                "box": (x1, y1, x2, y2),
                "cy": (y1 + y2) / 2,
                "height": max(1.0, y2 - y1),
            }
        )
    matches = []
    for first in items:
        same_line = sorted(
            (
                item
                for item in items
                if abs(item["cy"] - first["cy"])
                <= max(first["height"], item["height"]) * 0.75
            ),
            key=lambda item: item["box"][0],
        )
        for start in range(len(same_line)):
            combined = ""
            group = []
            for item in same_line[start : start + 5]:
                if group:
                    previous = group[-1]
                    gap = item["box"][0] - previous["box"][2]
                    if gap > max(previous["height"], item["height"]) * 4.0:
                        break
                combined += item["normalized"]
                group.append(item)
                exact_match = combined in wanted_names
                tiered_match = allow_tier_prefix and any(
                    combined == prefix + wanted
                    for prefix in SHIP_TIER_PREFIXES
                    for wanted in wanted_names
                )
                if exact_match or tiered_match:
                    x1 = min(member["box"][0] for member in group)
                    y1 = min(member["box"][1] for member in group)
                    x2 = max(member["box"][2] for member in group)
                    y2 = max(member["box"][3] for member in group)
                    matches.append(
                        (
                            (x1, y1, x2, y2),
                            min(member["confidence"] for member in group),
                            " ".join(member["text"] for member in group),
                        )
                    )
                    break
                maximum_length = max(map(len, wanted_names)) + (
                    4 if allow_tier_prefix else 0
                )
                if len(combined) >= maximum_length:
                    break
    return matches


def find_custom_ship_card(image, full_name, backend=None, minimum_confidence=0.60):
    """Find an exact custom ship name in the bottom port carousel via OCR."""
    backend = backend or RapidOcrBackend()
    height = image.shape[0]
    matches = []
    for ratio in (.735, .50):
        search_top = int(height * ratio)
        matches = _ocr_name_matches(
            image[search_top:height, :], full_name, backend,
            minimum_confidence, wanted_names=_custom_ship_ocr_names(full_name),
        )
        if matches:
            break
    if not matches:
        return None
    box, confidence, raw_text = max(matches, key=lambda match: match[1])
    x1, y1, x2, y2 = box
    name_y = search_top + int((y1 + y2) / 2)
    point = (int((x1 + x2) / 2), max(search_top, name_y - int((y2 - y1) * .5)))
    logger.info(
        "自定义舰船 OCR 命中: requested=%s recognized=%s confidence=%.3f",
        full_name,
        raw_text,
        confidence,
    )
    return point, confidence


def find_builtin_ship_card(
    image,
    ship_key,
    backend=None,
    *,
    minimum_score=0.55,
    minimum_ocr_confidence=0.45,
):
    """Locate a built-in ship card with template matching plus exact OCR.

    Gold-mask templates are fast, but the game may dim/recolour a card while
    the carousel is animating (Napoli is especially prone to this).  In that
    case an exact OCR match in the lower carousel is safer than rejecting a
    visible target or clicking a loose template correlation.
    """
    match = find_ship_card(image, ship_key, minimum_score=minimum_score)
    if match is not None or backend is None:
        return match
    aliases = BUILTIN_SHIP_OCR_NAMES.get((ship_key or "").strip().lower(), ())
    if not aliases:
        return None
    height = image.shape[0]
    normalized_aliases = {_normalize_ship_name(alias) for alias in aliases}
    best = None
    for ratio in (.70, .50):
        search_top = int(height * ratio)
        try:
            candidates = _ocr_line_candidates(
                backend.recognize(image[search_top:height, :]),
                minimum_confidence=minimum_ocr_confidence,
            )
        except Exception:
            logger.warning("舰船卡片 OCR 兜底失败: %s", ship_key, exc_info=True)
            continue
        for candidate in candidates:
            text = candidate["text"]
            if text not in normalized_aliases and not any(
                text == prefix + alias
                for prefix in SHIP_TIER_PREFIXES
                for alias in normalized_aliases
            ):
                continue
            match = (float(candidate["confidence"]), candidate["box"], text)
            if best is None or match[0] > best[0]:
                best = match
        if best is not None:
            break
    if best is None:
        logger.warning(
            "舰船卡片 %s 模板不足且 OCR 未命中 (template_threshold=%.2f)",
            ship_key,
            minimum_score,
        )
        return None
    confidence, box, raw_text = best
    x1, y1, x2, y2 = box
    name_y = search_top + int((y1 + y2) / 2)
    point = (
        int((x1 + x2) / 2),
        max(search_top, name_y - int((y2 - y1) * .5)),
    )
    logger.info(
        "舰船卡片 OCR 兜底命中: requested=%s recognized=%s confidence=%.3f",
        ship_key,
        raw_text,
        confidence,
    )
    return point, confidence


def is_custom_ship_selected(image, full_name, backend=None, minimum_confidence=0.60):
    """Strictly verify the selected ship name in the upper-right detail panel."""
    backend = backend or RapidOcrBackend()
    height = image.shape[0]
    # Keep enough header context for small title glyphs, but exclude other
    # ships' names elsewhere in the port and the lower commander panel.
    width = image.shape[1]
    detail = image[
        0 : int(height * 0.32),
        int(width * 0.50) : width,
    ]
    matches = _ocr_name_matches(
        detail,
        full_name,
        backend,
        minimum_confidence,
        allow_tier_prefix=True,
        wanted_names=_custom_ship_ocr_names(full_name),
    )
    if matches:
        logger.info(
            "自定义舰船详情复核通过: %s confidence=%.3f",
            full_name,
            max(match[1] for match in matches),
        )
        return True
    logger.info("自定义舰船详情复核未通过: %s", full_name)
    return False


def is_selected_ship_without_commander(image, backend=None):
    """Read ``没有指挥官`` only from the selected ship detail panel."""
    if image is None or image.size == 0:
        return False
    backend = backend or RapidOcrBackend()
    height, width = image.shape[:2]
    panel = image[
        int(height * 0.04) : int(height * 0.30),
        int(width * 0.79) : width,
    ]
    wanted = ("没有指挥官", "无指挥官")
    try:
        tokens = list(backend.recognize(panel) or ())
    except Exception:
        logger.warning("指挥官状态 OCR 失败；无法证明舰船可安全入队", exc_info=True)
        return None
    for token in tokens:
        text = _normalize_ship_name(token.text)
        if token.confidence >= 0.72 and any(word in text for word in wanted):
            logger.info(
                "右上角舰船详情确认无指挥官: confidence=%.3f",
                token.confidence,
            )
            return True
    return False


def find_ship_card_without_commander(image, ship_key, custom_name, backend):
    """Locate a named carousel card and its own no-commander overlay.

    Associate the overlay with the closest ship-name row below it, using
    rendered text height so resolution and UI scaling remain independent.
    """
    height = image.shape[0]
    top = int(height * 0.70)
    if ship_key == CUSTOM_SHIP_KEY:
        wanted = _custom_ship_ocr_names(custom_name)
    else:
        names = BUILTIN_SHIP_OCR_NAMES.get(ship_key, ())
        wanted = {_normalize_ship_name(name) for name in names if name}
    try:
        candidates = _ocr_line_candidates(
            backend.recognize(image[top:, :]), minimum_confidence=0.72
        )
    except Exception:
        logger.warning("下方舰船卡片指挥官 OCR 失败；本轮不召回", exc_info=True)
        return None
    missing = [c for c in candidates if c["text"] in {"没有指挥官", "无指挥官"}]
    labels = [
        c for c in candidates
        if any(ch.isalpha() for ch in c["text"])
        and "指挥官" not in c["text"]
    ]
    for warning in missing:
        wx1, wy1, wx2, wy2 = warning["box"]
        wh = max(1.0, wy2 - wy1)
        owners = []
        for label in labels:
            x1, y1, x2, y2 = label["box"]
            scale = max(wh, y2 - y1, 1.0)
            dy = (y1 + y2 - wy1 - wy2) / 2
            dx = x2 - wx2
            if 0.2 * scale <= dy <= 2.2 * scale and -scale <= dx <= 12 * scale:
                owners.append((dy / scale + abs(dx) / scale, label))
        if not owners:
            continue
        label = min(owners, key=lambda item: item[0])[1]
        text = label["text"]
        if text not in wanted and not any(
            text == prefix + name for prefix in SHIP_TIER_PREFIXES for name in wanted
        ):
            continue
        x1, y1, x2, y2 = label["box"]
        return (int((x1 + x2) / 2), top + int((y1 + y2) / 2))
    return None


def _find_recall_commander_action(image, backend):
    """Return the OCR centre of the ``召回指挥官`` context-menu action."""
    height = image.shape[0]
    search_top = int(height * 0.58)
    wanted = _normalize_ship_name("召回指挥官")
    try:
        tokens = list(backend.recognize(image[search_top:, :]) or ())
    except Exception:
        logger.warning("召回指挥官菜单 OCR 失败；本轮不点击", exc_info=True)
        return None
    matches = []
    for token in tokens:
        geometry = _token_geometry(token)
        if token.confidence < 0.72 or geometry is None:
            continue
        if wanted not in _normalize_ship_name(token.text):
            continue
        x1, y1, x2, y2 = geometry
        matches.append(
            (
                float(token.confidence),
                (int((x1 + x2) / 2), search_top + int((y1 + y2) / 2)),
            )
        )
    return max(matches, default=(0.0, None), key=lambda item: item[0])[1]


def ensure_selected_ship_commander(
    hwnd,
    ship_key,
    *,
    custom_name=None,
    backend=None,
    should_abort=None,
    attempts=2,
):
    """Recall the selected ship's commander before joining matchmaking.

    Verify the selected title first, then require no-commander text in both
    that detail panel and the requested carousel card before opening a menu.
    """
    backend = backend or RapidOcrBackend()
    ship_key = (ship_key or "").strip().lower()

    def requested_ship_selected(frame):
        if ship_key == CUSTOM_SHIP_KEY:
            return bool(custom_name) and is_custom_ship_selected(frame, custom_name, backend)
        return is_requested_ship_selected(
            frame, ship_key, minimum_score=0.64, minimum_margin=0.08, backend=backend
        )

    def recall_confirmed():
        if _operation_paused(should_abort):
            return False
        frame = _capture(hwnd)
        return requested_ship_selected(frame) and is_selected_ship_without_commander(frame, backend) is False

    for attempt in range(1, max(1, int(attempts)) + 1):
        if _operation_paused(should_abort):
            return False
        image = _capture(hwnd)
        requested_selected = requested_ship_selected(image)
        if not requested_selected:
            logger.warning(
                "拒绝召回指挥官：右上角当前舰船不是指定舰船 (%s)",
                custom_name if ship_key == CUSTOM_SHIP_KEY else ship_key,
            )
            return False
        commander_missing = is_selected_ship_without_commander(image, backend)
        if commander_missing is None:
            return False
        if not commander_missing:
            return True
        point = find_ship_card_without_commander(
            image, ship_key, custom_name, backend
        )
        if point is None:
            logger.warning("右上角无指挥官，但下方指定舰船卡片未同步确认；等待重新识别，不召回")
            return False
        if _operation_paused(should_abort):
            return False
        action = _find_recall_commander_action(image, backend)
        if action is not None:
            logger.info("检测到已打开的召回菜单，直接点击已识别文字: local=%s", action)
            if _click_local(hwnd, action):
                time.sleep(1.2)
                if recall_confirmed():
                    logger.info("指挥官召回复核通过")
                    return True
            continue
        logger.info(
            "舰名复核通过且上下两处均无指挥官，右键舰船卡片召回 (%s/%s): local=%s",
            attempt,
            attempts,
            point,
        )
        if not _right_click_local(hwnd, point):
            continue
        time.sleep(0.45)
        if _operation_paused(should_abort):
            return False
        menu = _capture(hwnd)
        if not requested_ship_selected(menu):
            logger.warning("召回菜单打开后舰船名称变化或未识别；取消召回")
            return False
        action = _find_recall_commander_action(menu, backend)
        if action is None:
            logger.warning("舰船右键菜单中未识别到“召回指挥官”")
            continue
        if _operation_paused(should_abort):
            return False
        if not _click_local(hwnd, action):
            continue
        time.sleep(1.2)
        if _operation_paused(should_abort):
            return False
        if recall_confirmed():
            logger.info("指挥官召回复核通过")
            return True
        logger.warning("召回后舰名或指挥官状态未通过复核")
    return False


def find_ship_card(image, ship_key, minimum_score=0.55):
    """Find a supported ship by matching its gold name glyphs in the carousel."""
    ship_key = (ship_key or "").strip().lower()
    if ship_key not in SUPPORTED_SHIPS:
        raise ValueError(f"Unsupported ship: {ship_key}")
    template = _load_ship_name_template(ship_key)
    if template is None or template.size == 0:
        return None

    height, width = image.shape[:2]
    search_top = int(height * 0.50)
    search = image[search_top:height, :]
    search_mask = _gold_name_mask(search)
    best = None
    for scale in _ui_template_scales(width, height):
        scaled = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        template_mask = _tighten_mask(_gold_name_mask(scaled))
        if template_mask.size == 0 or np.count_nonzero(template_mask) < 12:
            continue
        if (
            search_mask.shape[0] < template_mask.shape[0]
            or search_mask.shape[1] < template_mask.shape[1]
        ):
            continue
        result = cv2.matchTemplate(
            search_mask,
            template_mask,
            cv2.TM_CCOEFF_NORMED,
        )
        _, score, _, location = cv2.minMaxLoc(result)
        candidate = (float(score), location, template_mask.shape)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    score, location, template_shape = best
    if score < minimum_score:
        logger.warning("舰船卡片 %s 匹配置信度不足: %.3f", ship_key, score)
        return None
    template_height, template_width = template_shape
    x = location[0] + template_width // 2
    name_y = search_top + location[1] + template_height // 2
    # Click the card body above its name, avoiding adjacent card borders.
    card_y = max(search_top, name_y - int(template_height * .6))
    return (x, card_y), float(score)


def selected_ship_scores(image):
    """Score supported ship names in the port's upper-right detail panel."""
    height, width = image.shape[:2]
    search = image[
        0 : int(height * 0.32),
        int(width * 0.50) : width,
    ]
    search_mask = _gold_name_mask(search)
    scores = {}
    for ship_key in SUPPORTED_SHIPS:
        template = _load_selected_ship_name_template(ship_key)
        if template is None or template.size == 0:
            scores[ship_key] = 0.0
            continue
        best_score = 0.0
        for scale in _ui_template_scales(width, height):
            scaled = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
            if (
                search_mask.shape[0] < scaled.shape[0]
                or search_mask.shape[1] < scaled.shape[1]
            ):
                continue
            result = cv2.matchTemplate(
                search_mask,
                scaled,
                cv2.TM_CCOEFF_NORMED,
            )
            best_score = max(best_score, float(cv2.minMaxLoc(result)[1]))
        scores[ship_key] = best_score
    return scores


def detect_selected_ship(
    image,
    backend=None,
    *,
    minimum_template_score=0.86,
    minimum_template_margin=0.08,
    minimum_ocr_confidence=0.68,
):
    """Read the currently selected built-in ship from the upper-right card.

    The bottom carousel may contain several requested-ship name matches and is
    deliberately excluded. OCR is confined to the upper-right detail card;
    the existing gold-title templates provide a resolution/UI-scale fallback.
    Returns ``(ship_key, confidence, source)`` or ``(None, confidence, source)``.
    """
    if image is None or not hasattr(image, "size") or image.size == 0:
        return None, 0.0, "invalid"

    if backend is not None:
        height, width = image.shape[:2]
        # The card can grow leftward at larger UI scales, while the ship title
        # itself remains in the upper-right quadrant.
        detail = image[
            0 : int(height * 0.32),
            int(width * 0.50) : width,
        ]
        raw_tokens = ()
        try:
            raw_tokens = tuple(backend.recognize(detail) or ())
            candidates = _ocr_line_candidates(
                raw_tokens,
                minimum_confidence=minimum_ocr_confidence,
            )
        except Exception:
            logger.warning("右上角当前舰船 OCR 失败，改用舰名模板", exc_info=True)
            candidates = []
        ocr_matches = []
        for candidate in candidates:
            text = candidate["text"]
            confidence = float(candidate["confidence"])
            for ship_key, aliases in BUILTIN_SHIP_OCR_NAMES.items():
                for alias in aliases:
                    wanted = _normalize_ship_name(alias)
                    if text == wanted or any(
                        text == prefix + wanted for prefix in SHIP_TIER_PREFIXES
                    ):
                        ocr_matches.append((confidence, ship_key, text))
        if ocr_matches:
            confidence, ship_key, raw_text = max(ocr_matches)
            logger.info(
                "右上角舰船卡片识别: ship=%s source=ocr confidence=%.3f text=%s",
                ship_key,
                confidence,
                raw_text,
            )
            return ship_key, confidence, "ocr"

        # During the port carousel animation the title can be dimmed enough
        # for RapidOCR to report ~0.45-0.67 confidence. Accept only an exact
        # built-in alias in this tightly scoped upper-right panel.
        if raw_tokens:
            low_confidence = max(0.42, min(float(minimum_ocr_confidence), 0.45))
            low_candidates = _ocr_line_candidates(
                raw_tokens,
                minimum_confidence=low_confidence,
            )
            for candidate in low_candidates:
                text = candidate["text"]
                confidence = float(candidate["confidence"])
                for low_ship_key, aliases in BUILTIN_SHIP_OCR_NAMES.items():
                    normalized_aliases = {
                        _normalize_ship_name(alias) for alias in aliases
                    }
                    if text in normalized_aliases or any(
                        text == prefix + alias
                        for prefix in SHIP_TIER_PREFIXES
                        for alias in normalized_aliases
                    ):
                        logger.info(
                            "右上角舰船卡片识别: ship=%s source=ocr_low_confidence confidence=%.3f text=%s",
                            low_ship_key,
                            confidence,
                            text,
                        )
                        return low_ship_key, confidence, "ocr_low_confidence"

    scores = selected_ship_scores(image)
    if not scores:
        return None, 0.0, "template"
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ship_key, score = ordered[0]
    competitor = ordered[1][1] if len(ordered) > 1 else 0.0
    logger.info(
        "右上角舰船卡片模板复核: best=%s %.3f competitor=%.3f",
        ship_key,
        score,
        competitor,
    )
    if (
        # A ~0.65 match to the gold no-commander text used to impersonate
        # Pommern and bypass the carousel click. Callers may tighten this
        # threshold, but cannot weaken the identity check below this floor.
        score >= max(0.86, minimum_template_score)
        and score - competitor >= minimum_template_margin
    ):
        return ship_key, score, "template"
    logger.info("舰名未确认，模板 %.3f 不足以跳过切船；应定位并点击目标舰船", score)
    return None, score, "template_ambiguous"


def is_requested_ship_selected(
    image,
    ship_key,
    minimum_score=0.72,
    minimum_margin=0.15,
    backend=None,
):
    """Verify the selected ship from its detail-panel title, not card styling."""
    ship_key = (ship_key or "").strip().lower()
    if ship_key not in SUPPORTED_SHIPS:
        raise ValueError(f"Unsupported ship: {ship_key}")
    selected_key, confidence, source = detect_selected_ship(
        image,
        backend,
        minimum_template_score=minimum_score,
        minimum_template_margin=minimum_margin,
    )
    logger.info(
        "舰船详情标题复核 %s: selected=%s confidence=%.3f source=%s",
        ship_key,
        selected_key or "unknown",
        confidence,
        source,
    )
    return selected_key == ship_key


def _verify_builtin_ship_after_click(
    hwnd,
    ship_key,
    backend,
    *,
    vision=None,
    should_abort=None,
    attempts=6,
    stable_samples=2,
):
    """Verify a built-in ship only after the carousel click was dispatched.

    The detail card animates independently of the carousel. Require two
    consecutive right-upper-card observations of the requested ship so a
    stale frame cannot be mistaken for a successful switch.
    """
    consecutive = 0
    last_confidence = 0.0
    last_source = "not_checked"
    for attempt in range(max(1, int(attempts))):
        if _operation_paused(should_abort):
            return False, last_confidence, "paused"
        try:
            image = _capture(hwnd)
        except CaptureFault:
            consecutive = 0
            continue
        if vision is not None:
            try:
                if vision.classify_screen(image) != ScreenState.PORT:
                    logger.warning(
                        "点击切船后画面已离开港口，拒绝确认舰船: %s",
                        ship_key,
                    )
                    consecutive = 0
                    continue
            except Exception:
                logger.debug("点击切船后港口状态复核失败", exc_info=True)
        selected, confidence, source = detect_selected_ship(
            image,
            backend,
        )
        last_confidence = confidence
        last_source = source
        logger.info(
            "点击切船后右上角复核 (%s/%s): selected=%s target=%s confidence=%.3f source=%s",
            attempt + 1,
            max(1, int(attempts)),
            selected or "unknown",
            ship_key,
            confidence,
            source,
        )
        if selected == ship_key:
            consecutive += 1
            if consecutive >= max(1, int(stable_samples)):
                return True, confidence, source
        else:
            consecutive = 0
        if attempt + 1 < max(1, int(attempts)):
            time.sleep(0.30)
    return False, last_confidence, last_source


def _confirm_custom_ship_after_click(
    hwnd,
    point,
    full_name,
    backend,
    *,
    click_attempts=2,
    confirmation_seconds=4.0,
    should_abort=None,
):
    """Try the chosen card twice, then stop; never scroll after a card hit."""
    global _LAST_SELECTED_CARD_POINT
    for click_attempt in range(1, max(1, int(click_attempts)) + 1):
        if _operation_paused(should_abort):
            return False
        logger.info(
            "点击自定义舰船 %s (%s/%s)",
            full_name,
            click_attempt,
            click_attempts,
        )
        if not _click_local(hwnd, point):
            continue
        _LAST_SELECTED_CARD_POINT = tuple(point)
        deadline = time.monotonic() + max(0.5, float(confirmation_seconds))
        while time.monotonic() < deadline:
            if _operation_paused(should_abort):
                return False
            time.sleep(0.5)
            if _operation_paused(should_abort):
                return False
            if is_custom_ship_selected(_capture_ship_cards(hwnd), full_name, backend):
                _remember_selected_ship(f"custom:{full_name}")
                return True
        logger.warning(
            "第 %s 次点击后，右上角尚未确认舰船: %s",
            click_attempt,
            full_name,
        )
    raise ShipSelectionError(
        f"已尝试选择“{full_name}”两次，但右上角舰名未确认；任务已退出，请重新选择"
    )


def _rewind_ship_carousel(hwnd, rect, *, steps=20, should_abort=None):
    """Put the port carousel at its first card before doing one full sweep.

    Mouse-wheel paging in the port is horizontal, but its visible direction
    changes with the selected UI layout.  Repeated positive wheel input reaches
    the left/top boundary reliably; from there, negative input visits every
    card once without oscillating back across already-scanned ships.
    """
    if not hwnd or not rect:
        return False
    carousel_x = rect["left"] + rect["width"] // 2
    carousel_y = rect["top"] + int(rect["height"] * 0.90)
    logger.info("舰船栏先回到顶部/起点，再单向向下遍历")
    moved = False
    for _ in range(max(1, int(steps))):
        if _operation_paused(should_abort):
            return False
        if not physical_scroll(carousel_x, carousel_y, 8, hwnd=hwnd):
            break
        moved = True
        time.sleep(0.18)
    return moved


def _scroll_ship_carousel_down(hwnd, rect, *, step=3):
    """Advance one page through the carousel after it has been rewound."""
    carousel_x = rect["left"] + rect["width"] // 2
    carousel_y = rect["top"] + int(rect["height"] * 0.90)
    return physical_scroll(carousel_x, carousel_y, -abs(int(step)), hwnd=hwnd)


def _capture_ship_cards(hwnd):
    if hwnd:
        park_port_cursor(hwnd)
    return _capture(hwnd)


def _select_custom_ship(
    hwnd, image, full_name, backend, max_scrolls=18, *, should_abort=None
):
    if _operation_paused(should_abort):
        return False
    if is_custom_ship_selected(image, full_name, backend):
        logger.info("目标自定义舰船已选中: %s", full_name)
        _remember_selected_ship(f"custom:{full_name}")
        return True

    rect = get_client_rect(hwnd) if hwnd else None
    if hwnd and rect:
        if not _rewind_ship_carousel(hwnd, rect, should_abort=should_abort):
            return False
        if _operation_paused(should_abort):
            return False
        image = _capture_ship_cards(hwnd)
        if is_custom_ship_selected(image, full_name, backend):
            logger.info("回到舰船栏起点后已确认目标自定义舰船: %s", full_name)
            _remember_selected_ship(f"custom:{full_name}")
            return True
    match = find_custom_ship_card(image, full_name, backend)
    for attempt in range(max_scrolls + 1):
        if _operation_paused(should_abort):
            return False
        if match is not None:
            point, confidence = match
            logger.info(
                "按 OCR 定位自定义舰船 %s: local=%s confidence=%.3f",
                full_name,
                point,
                confidence,
            )
            return _confirm_custom_ship_after_click(
                hwnd,
                point,
                full_name,
                backend,
                should_abort=should_abort,
            )
        if not hwnd or attempt >= max_scrolls:
            break
        logger.info(
            "未确认自定义舰船 %s，向下遍历港口舰船栏 (%s/%s)",
            full_name,
            attempt + 1,
            max_scrolls,
        )
        if not _scroll_ship_carousel_down(hwnd, rect):
            break
        time.sleep(0.35)
        if _operation_paused(should_abort):
            return False
        image = _capture_ship_cards(hwnd)
        if is_custom_ship_selected(image, full_name, backend):
            _remember_selected_ship(f"custom:{full_name}")
            return True
        match = find_custom_ship_card(image, full_name, backend)

    raise ShipSelectionError(
        f"港口舰船栏中未找到并确认“{full_name}”；任务已退出，请在网页重新选择"
    )


def select_requested_ship(
    hwnd=None,
    ship_key="pommern",
    vision=None,
    *,
    custom_name=None,
    ocr_backend=None,
    custom_max_scrolls=18,
    should_abort=None,
    require_port_action=False,
):
    """Select a built-in template ship or an exact custom ship OCR name."""
    ship_key = (ship_key or "").strip().lower()
    if _operation_paused(should_abort):
        return False
    is_custom = ship_key == CUSTOM_SHIP_KEY
    if ship_key not in SUPPORTED_SHIPS and not is_custom:
        logger.error("不支持的舰船: %s", ship_key)
        return False
    image = _capture_ship_cards(hwnd)
    vision = vision or Vision()
    if vision.classify_screen(image) != ScreenState.PORT:
        logger.warning("当前不是港口，拒绝选择舰船")
        return False
    selection_backend = ocr_backend or RapidOcrBackend()
    full_name = ""
    current_selected_key = None
    if is_custom:
        full_name = (
            custom_name or os.environ.get("WOWS_CUSTOM_SHIP_NAME", "")
        ).strip()
        if not full_name:
            raise ShipSelectionError("自定义舰船名称为空，请返回网页重新选择")
        if is_custom_ship_selected(image, full_name, selection_backend):
            logger.info("右上角舰船卡片已是目标自定义舰船: %s", full_name)
            _remember_selected_ship(f"custom:{full_name}")
            return True
        logger.info("右上角舰船卡片不是目标自定义舰船，开始查找: %s", full_name)
    else:
        current_selected_key, confidence, source = detect_selected_ship(
            image,
            selection_backend,
        )
        if current_selected_key is None and hwnd:
            # The right-side detail card fades in independently of the port
            # background. Give it two fresh game-window frames before
            # touching the carousel; otherwise a selected Pommern/Napoli is
            # mistaken for an unknown ship and the safe action interlock
            # rejects the whole run.
            for retry in range(2):
                if _operation_paused(should_abort):
                    return False
                time.sleep(0.25)
                try:
                    refreshed = _capture_ship_cards(hwnd)
                except CaptureFault:
                    continue
                if vision.classify_screen(refreshed) != ScreenState.PORT:
                    continue
                image = refreshed
                current_selected_key, confidence, source = detect_selected_ship(
                    image,
                    selection_backend,
                )
                if current_selected_key is not None:
                    logger.info(
                        "右上角舰船卡片第 %s 次刷新后恢复识别: %s confidence=%.3f source=%s",
                        retry + 1,
                        current_selected_key,
                        confidence,
                        source,
                    )
                    break
        if current_selected_key == ship_key:
            logger.info(
                "右上角舰船卡片已是目标舰船: %s confidence=%.3f source=%s；"
                "跳过舰船栏轮询",
                ship_key,
                confidence,
                source,
            )
            _remember_selected_ship(ship_key)
            return True
        if current_selected_key:
            logger.info(
                "右上角当前舰船为 %s，不符合目标 %s；开始查找舰船栏",
                current_selected_key,
                ship_key,
            )
        else:
            logger.info("右上角当前舰船暂未确认；开始查找目标舰船 %s", ship_key)

    if require_port_action:
        # Scene colour/texture is not enough to authorize carousel input. A
        # live battle at some UI scales can satisfy the broad port anchors.
        # Reject any positive battle HUD, then require the exact rendered
        # “加入战斗” text before the first ship scroll or click.
        battle_detector = getattr(vision, "_has_battle_hud", None)
        try:
            battle_hud_visible = bool(
                callable(battle_detector) and battle_detector(image)
            )
        except Exception:
            logger.debug("选船前战斗 HUD 互锁检查失败", exc_info=True)
            battle_hud_visible = False
        if battle_hud_visible:
            logger.warning("选船互锁：当前画面仍有战斗 HUD，禁止港口选船操作")
            return False
        action_point = port_battle_action_point(image, selection_backend)
        if action_point is None:
            action_point = find_battle_button(hwnd, image)
            if action_point is not None:
                logger.info(
                    "选船互锁：加入战斗文字暂不可读，颜色按钮已确认: local=%s",
                    action_point,
                )
        # A port frame can be captured during the header animation before the
        # action button is painted. Retry fresh game-window frames briefly;
        # never scroll the carousel until the positive PORT + action evidence
        # is available, but do not reject a correctly selected ship forever.
        if action_point is None and hwnd:
            for retry in range(3):
                if _operation_paused(should_abort):
                    return False
                time.sleep(0.25)
                try:
                    image = _capture_ship_cards(hwnd)
                except CaptureFault:
                    continue
                try:
                    if vision.classify_screen(image) != ScreenState.PORT:
                        continue
                except Exception:
                    continue
                action_point = port_battle_action_point(image, selection_backend)
                if action_point is None:
                    action_point = find_battle_button(hwnd, image)
                if action_point is not None:
                    logger.info(
                        "选船互锁：第 %s 次刷新后确认加入战斗按钮: local=%s",
                        retry + 1,
                        action_point,
                    )
                    break
        if action_point is None:
            logger.warning(
                "选船互锁：连续刷新仍未确认港口加入战斗按钮，禁止滚动或点击舰船"
            )
            return False
    if is_custom:
        return _select_custom_ship(
            hwnd,
            image,
            full_name,
            selection_backend,
            max_scrolls=max(0, int(custom_max_scrolls)),
            should_abort=should_abort,
        )
    match = find_builtin_ship_card(
        image,
        ship_key,
        selection_backend,
    )
    if match is None and hwnd:
        rect = get_client_rect(hwnd)
        if not _rewind_ship_carousel(hwnd, rect, should_abort=should_abort):
            return False
        if _operation_paused(should_abort):
            return False
        image = _capture_ship_cards(hwnd)
        selected_key, _confidence, _source = detect_selected_ship(
            image,
            selection_backend,
        )
        if selected_key == ship_key:
            _remember_selected_ship(ship_key)
            return True
        match = find_builtin_ship_card(
            image,
            ship_key,
            selection_backend,
        )
        # A single direction sweep avoids revisiting cards and never leaves the
        # requested ship just outside a reverse-search boundary.
        for attempt in range(18):
            if _operation_paused(should_abort):
                return False
            if match is not None:
                break
            logger.info(
                "未发现舰船 %s，向下遍历港口舰船栏 (%s/18)",
                ship_key,
                attempt + 1,
            )
            if not _scroll_ship_carousel_down(hwnd, rect):
                break
            time.sleep(0.25)
            if _operation_paused(should_abort):
                return False
            image = _capture_ship_cards(hwnd)
            selected_key, _confidence, _source = detect_selected_ship(
                image,
                selection_backend,
            )
            if selected_key == ship_key:
                _remember_selected_ship(ship_key)
                return True
            match = find_builtin_ship_card(
                image,
                ship_key,
                selection_backend,
            )
            if match is not None:
                break
    if match is None:
        # A persisted last-selected value is only a hint for diagnostics. It
        # must never authorize this run without a fresh click and right-upper
        # detail-card confirmation.
        if current_selected_key is None and _last_confirmed_ship() == ship_key:
            logger.warning(
                "滚动搜索后舰名区域仍暂不可读；忽略上次确认舰船 %s，未发出切船点击",
                ship_key,
            )
        return False
    point, score = match
    logger.info("按截图定位舰船 %s: local=%s score=%.3f", ship_key, point, score)
    if not _click_local(hwnd, point):
        return False
    global _LAST_SELECTED_CARD_POINT
    _LAST_SELECTED_CARD_POINT = tuple(point)
    logger.info("已发出切船点击，开始读取右上角当前船卡复核: %s", ship_key)
    time.sleep(1.8)
    if _operation_paused(should_abort):
        return False
    verified, confidence, source = _verify_builtin_ship_after_click(
        hwnd,
        ship_key,
        selection_backend,
        vision=vision,
        should_abort=should_abort,
    )
    if not verified:
        logger.warning(
            "舰船选择复核失败（点击后右上角未连续确认）: %s",
            ship_key,
        )
        return False
    logger.info(
        "舰船选择复核成功: %s confidence=%.3f source=%s",
        ship_key,
        confidence,
        source,
    )
    _remember_selected_ship(ship_key)
    return True


def click_center(hwnd=None):
    image = _capture(hwnd)
    height, width = image.shape[:2]
    return _click_local(hwnd, (width // 2, height // 2))


def click_loading_start_action(
    hwnd=None,
    image=None,
    *,
    vision=None,
    backend=None,
    should_abort=None,
):
    """Click the positively detected second-stage ``开始战斗`` action.

    After ``继续战斗`` the client can stop on a roster/tips page instead of
    entering the live HUD immediately.  That page is deliberately classified
    as loading so no movement keys are sent, but it still requires this one
    explicit mouse confirmation.  The visual detector is the same narrow
    green-bar detector used by the lifecycle guard. With an OCR backend, the
    rendered label supplies the click point after that visual verification.
    """
    if _operation_paused(should_abort):
        return False
    image = image if image is not None else _capture(hwnd)
    vision = vision or Vision()
    detector = getattr(vision, "_has_loading_start_action", None)
    if not callable(detector):
        logger.warning("加载页没有可用的“开始战斗”按钮检测器，拒绝点击")
        return False
    try:
        visible = bool(detector(image))
    except Exception:
        logger.warning("加载页“开始战斗”按钮复核失败，拒绝点击", exc_info=True)
        return False
    if not visible:
        logger.debug("当前画面未确认“开始战斗”按钮，拒绝点击")
        return False
    height, width = image.shape[:2]
    if backend is not None:
        point = _verified_action_point(
            image, backend, ("开始战斗",),
            rejected=("离开战斗", "继续战斗"), region=LOADING_START_BUTTON,
        )
        if point is None:
            locate_box = getattr(vision, "loading_start_action_box", None)
            bounds = locate_box(image) if callable(locate_box) else None
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
                logger.debug("加载按钮文字与实际矩形尚未识别，保留页面等待新帧")
                return False
            x1, y1, x2, y2 = bounds
            point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    else:
        # Legacy diagnostic callers use the narrow, confirmed solid bar.
        point = LOADING_START_BUTTON.center(width, height)
    if _operation_paused(should_abort):
        return False
    logger.info("加载页已确认“开始战斗”按钮，点击进入战斗: local=%s", point)
    clicked = _click_local(hwnd, point)
    if not clicked:
        logger.warning("“开始战斗”按钮点击未派发，保留页面并稍后有限重试")
    return bool(clicked)


def enter_battle(
    hwnd=None,
    *,
    vision=None,
    configure_port=True,
    backend=None,
    should_abort=None,
):
    """Click ``加入战斗`` once, then verify or conservatively await transition."""
    vision = vision or Vision()
    if _operation_paused(should_abort):
        return False
    image = _capture(hwnd)
    state = vision.classify_screen(image)
    if state != ScreenState.PORT:
        logger.warning("当前界面为 %s，不点击“加入战斗”", state.value)
        return False

    if configure_port:
        if not select_requested_ship(
            hwnd,
            os.environ.get("WOWS_SHIP", "pommern"),
            vision,
            ocr_backend=backend,
            should_abort=should_abort,
            require_port_action=True,
        ):
            logger.warning("未能安全选择目标舰船")
            return False

        if not ensure_requested_mode(
            hwnd,
            os.environ.get("WOWS_MODE", "asymmetric"),
            vision,
            backend=backend,
            should_abort=should_abort,
        ):
            logger.warning("未能安全选择目标战斗模式")
            return False

    if _operation_paused(should_abort):
        return False
    image = _capture(hwnd)
    if not click_battle(hwnd, image, backend=backend):
        return False
    return _observe_battle_entry(
        hwnd,
        vision,
        should_abort=should_abort,
    )


def queue_next_battle(hwnd=None, *, vision=None, backend=None, should_abort=None,
                      source=ScreenState.RESULTS):
    """Queue from verified results or the post-Esc sunk-ship continuation.

    True also means a dispatched click is still pending verification. Keep
    the entry lock during capture failures/pauses/transitions after that click.
    """
    vision = vision or Vision()
    if _operation_paused(should_abort):
        logger.info("[USER] 结算续局已暂停，不点击“继续战斗”")
        return False
    try:
        image = _capture(hwnd)
    except CaptureFault as error:
        logger.info("结算续局画面暂不可用，保留页面并稍后重试: %s", error)
        return False
    state = vision.classify_screen(image)
    if source == ScreenState.RESULTS and state != ScreenState.RESULTS:
        logger.warning("当前界面为 %s，不点击“继续战斗”", state.value)
        return False
    if source == ScreenState.QUICK_REQUEUE and state == ScreenState.PORT:
        logger.warning("死亡续局页面已变为港口，不按过期的继续战斗坐标")
        return False
    if source not in {ScreenState.RESULTS, ScreenState.QUICK_REQUEUE}:
        return False
    if _operation_paused(should_abort):
        logger.info("[USER] 点击续局前检测到介入，保留结算页")
        return False
    locator = (
        quick_death_action_point
        if source == ScreenState.QUICK_REQUEUE
        else results_requeue_action_point
    )
    action_point = locator(image, backend) if backend is not None else None
    if source == ScreenState.QUICK_REQUEUE and action_point is None:
        logger.warning("死亡后页面未识别到“继续战斗”，保留页面重试")
        return False
    if backend is not None and action_point is None:
        logger.warning("结算页 OCR 未定位到“继续战斗”，拒绝按旧坐标点击")
        return False
    logger.info("%s 已确认，点击“继续战斗”进入下一局: %s", source.value, action_point)
    clicked = (
        _click_local(hwnd, action_point)
        if action_point is not None
        else _click_region(hwnd, image, RESULTS_REQUEUE_BUTTON)
    )
    if not clicked:
        return False
    # Verify that the click actually left the result page.  Some result pages
    # are still animating when the button first appears, so retry once instead
    # of reporting success for a click the game ignored.
    retried = False
    no_commander_frames = 0
    for attempt in range(24):
        if _operation_paused(should_abort):
            logger.info("[USER] 等待下一局期间暂停，不再派发续局动作")
            return True
        time.sleep(0.25)
        try:
            confirmation = _capture(hwnd)
        except CaptureFault as error:
            logger.info("续局确认画面暂不可用，保留流程并稍后重识别: %s", error)
            return True
        state = vision.classify_screen(confirmation)
        if state in {ScreenState.LOADING, ScreenState.BATTLE}:
            return True
        if state == ScreenState.PORT and backend is not None and _verified_action_point(
            confirmation, backend, ("加入战斗",), region=PORT_BATTLE_TEXT_AREA
        ) is not None:
            logger.info("继续战斗返回了港口，改用港口常规入口")
            return False
        warning_visible = False
        if state == ScreenState.UNKNOWN and vision.in_no_commander_confirmation(confirmation):
            if backend is not None:
                h, w = confirmation.shape[:2]
                try:
                    tokens = backend.recognize(confirmation[int(h*.30):int(h*.68), int(w*.25):int(w*.75)])
                    warning_visible = any(
                        token.confidence >= .75 and any(
                            word in _normalize_ship_name(token.text)
                            for word in ("没有指挥官", "无指挥官", "未分配指挥官")
                        ) for token in tokens or ()
                    )
                except Exception:
                    logger.debug("续局指挥官提示 OCR 暂不可用", exc_info=True)
        no_commander_frames = no_commander_frames + 1 if warning_visible else 0
        if no_commander_frames >= 2:
            logger.warning("续局连续两帧文字确认无指挥官提示，交回场景恢复")
            return False
        retry_point = (
            locator(confirmation, backend)
            if backend is not None and (
                state == ScreenState.RESULTS
                or source == ScreenState.QUICK_REQUEUE
            )
            else None
        )
        if attempt >= 7 and not retried and (
            state == ScreenState.RESULTS
            or source == ScreenState.QUICK_REQUEUE and retry_point is not None
        ):
            logger.info("继续战斗按钮未生效，重新点击一次")
            if backend is not None and retry_point is None:
                return True
            retry_clicked = (
                _click_local(hwnd, retry_point)
                if retry_point is not None
                else _click_region(hwnd, confirmation, RESULTS_REQUEUE_BUTTON)
            )
            if not retry_clicked:
                return True
            retried = True
    logger.warning("续局点击已派发，场景尚未确认；保持入局锁等待识别")
    return True


def force_quick_battle_return_to_port(
    hwnd=None,
    *,
    vision=None,
    backend=None,
    open_menu=None,
    ship_sunk=False,
    should_abort=None,
    attempts=60,
    diagnostic_dir=None,
):
    """Exit a quick match, or use the immediate post-death requeue path."""
    vision = vision or Vision()
    backend = backend or RapidOcrBackend()
    trace = BlockageTrace(diagnostic_dir, "quick_exit_blocked")
    pending_action = None
    image = None
    state = ScreenState.UNKNOWN
    if _operation_paused(should_abort):
        return ScreenState.UNKNOWN
    menu_opened_at = None
    menu_retries = 0
    exit_clicks = 0
    yes_clicks = 0
    last_exit_click = float("-inf")
    last_yes_click = float("-inf")
    closed_state = None
    closed_frames = 0
    death_action_seen = None
    death_action_frames = 0
    death_leave_frames = 0
    death_leave_presses = 0
    last_death_leave_press = float("-inf")
    for attempt in range(max(1, int(attempts))):
        if _operation_paused(should_abort):
            return ScreenState.UNKNOWN
        if attempt:
            time.sleep(0.25)
            if _operation_paused(should_abort):
                return ScreenState.UNKNOWN
        try:
            image = _capture(hwnd)
        except CaptureFault as error:
            logger.info("快速回港画面暂不可用，稍后重识别: %s", error)
            return ScreenState.UNKNOWN
        state = vision.classify_screen(image)
        if pending_action is not None:
            action, action_attempt = pending_action
            trace.record(image, state=state, action=action, attempt=action_attempt, phase="after")
            pending_action = None
        if ship_sunk and death_leave_presses:
            point = quick_death_action_point(image, backend)
            if point is not None:
                same_action = (
                    death_action_seen is not None
                    and abs(point[0] - death_action_seen[0]) <= image.shape[1] * .03
                    and abs(point[1] - death_action_seen[1]) <= image.shape[0] * .03
                )
                death_action_frames = death_action_frames + 1 if same_action else 1
                death_action_seen = point
                trace.record(image, state=state, action="death_requeue_ready", attempt=attempt)
                if death_action_frames >= 2:
                    logger.info("死亡后 Esc 页面已确认“继续战斗”；先按死亡计数，再决定是否排下一局")
                    trace.record(image, state=ScreenState.QUICK_REQUEUE,
                                 action="death_requeue_ready", attempt=attempt, phase="result")
                    return ScreenState.QUICK_REQUEUE
                continue
            death_action_seen = None
            death_action_frames = 0
        if state in {ScreenState.PORT, ScreenState.RESULTS}:
            closed_frames = closed_frames + 1 if state == closed_state else 1
            closed_state = state
            if closed_frames >= 2:
                logger.info("快速战斗已连续确认离开: %s", state.value)
                trace.record(image, state=state, action="closure_confirmed", attempt=attempt, phase="result")
                return state
            trace.record(image, state=state, action="await_second_closure_frame", attempt=attempt)
            continue
        closed_state = None
        closed_frames = 0

        if ship_sunk and state in {ScreenState.UNKNOWN, ScreenState.LOADING, ScreenState.BATTLE}:
            death_leave = quick_death_leave_hint_visible(image, backend)
            death_leave_frames = death_leave_frames + 1 if death_leave else 0
            if death_leave:
                trace.record(image, state=state, action="death_leave_hint", attempt=attempt)
                if (
                    death_leave_frames >= 2
                    and open_menu is not None
                    and death_leave_presses < 2
                    and time.monotonic() - last_death_leave_press >= 3.0
                ):
                    if _operation_paused(should_abort):
                        return ScreenState.UNKNOWN
                    trace.record(image, state=state, action="death_leave_escape", attempt=attempt, phase="before")
                    open_menu()
                    pending_action = ("death_leave_escape", attempt)
                    death_leave_presses += 1
                    last_death_leave_press = time.monotonic()
                    menu_opened_at = last_death_leave_press
                    logger.info("死亡提示连续两帧确认“ESC 离开战斗”，已按 Esc (%s/2)", death_leave_presses)
                continue
        else:
            death_leave_frames = 0

        if ship_sunk and death_leave_presses:
            # The post-Esc page can be classified as loading/unknown. Never
            # open the normal exit menu here: it would bypass the immediate
            # Continue Battle action or return the player to port.
            trace.record(image, state=state, action="await_death_requeue", attempt=attempt)
            continue

        # A dead ship may leave a live battle under a transition overlay. The
        # broad classifier then says LOADING even though the next frame can
        # prove BATTLE. Observe it instead of aborting three outer attempts.
        if _is_early_exit_confirmation(image, backend):
            yes_point = _verified_action_point(
                image,
                backend,
                ("是",),
                rejected=("否",),
                region=QUICK_EXIT_CONFIRM_ACTION_AREA,
                minimum_confidence=0.50,
                exact=True,
            )
            trace.record(image, state=state, action="exit_confirmation", attempt=attempt)
            if time.monotonic() - last_yes_click < 1.5 or yes_clicks >= 3:
                continue
            if _operation_paused(should_abort):
                return ScreenState.UNKNOWN
            trace.record(image, state=state, action="confirm_exit", attempt=attempt, phase="before")
            clicked = (
                _click_local(hwnd, yes_point)
                if yes_point is not None
                else _click_region(hwnd, image, QUICK_EXIT_CONFIRM_YES_BUTTON)
            )
            if clicked:
                pending_action = ("confirm_exit", attempt)
                yes_clicks += 1
                last_yes_click = time.monotonic()
                logger.info("已点击经 OCR 复核的“提前退出战斗→是” (%s/3)", yes_clicks)
            else:
                logger.warning("快速退出确认按钮点击未派发，保留页面重试")
            continue

        exit_point = quick_exit_menu_action_point(image, backend)
        if exit_point is not None and (
            menu_opened_at is not None or state == ScreenState.ESCAPE_MENU
        ):
            trace.record(image, state=state, action="exit_menu", attempt=attempt)
            if time.monotonic() - last_exit_click < 1.5 or exit_clicks >= 3:
                continue
            if _operation_paused(should_abort):
                return ScreenState.UNKNOWN
            trace.record(image, state=state, action="click_leave_battle", attempt=attempt, phase="before")
            if _click_local(hwnd, exit_point):
                pending_action = ("click_leave_battle", attempt)
                exit_clicks += 1
                last_exit_click = time.monotonic()
                logger.info("已点击经 OCR 复核的“离开战斗” (%s/3): local=%s", exit_clicks, exit_point)
            else:
                logger.warning("快速战斗菜单点击未派发，保留页面重试")
            continue

        if state != ScreenState.BATTLE:
            trace.record(image, state=state, action="await_battle_or_result", attempt=attempt)
            continue
        if open_menu is None:
            return ScreenState.BATTLE
        if menu_opened_at is None:
            if _operation_paused(should_abort):
                return ScreenState.UNKNOWN
            trace.record(image, state=state, action="open_escape_menu", attempt=attempt, phase="before")
            open_menu()
            pending_action = ("open_escape_menu", attempt)
            menu_opened_at = time.monotonic()
        elif (
            exit_clicks > 0
            and yes_clicks == 0
            and time.monotonic() - last_exit_click >= 3.0
            and menu_retries < 2
        ):
            # The menu vanished without a readable confirmation. Recover only
            # when the live HUD is back and neither overlay colour is present.
            menu_visible = bool(getattr(vision, "in_escape_menu", lambda _image: False)(image))
            confirm_visible = bool(getattr(vision, "in_exit_confirmation", lambda _image: False)(image))
            if menu_visible or confirm_visible:
                trace.record(image, state=state, action="await_overlay_ocr", attempt=attempt)
                continue
            if _operation_paused(should_abort):
                return ScreenState.UNKNOWN
            menu_retries += 1
            logger.info("离开战斗点击后未出现确认页，重新打开战斗菜单 (%s/3)", menu_retries + 1)
            trace.record(image, state=state, action="reopen_escape_menu", attempt=attempt, phase="before")
            open_menu()
            pending_action = ("reopen_escape_menu", attempt)
            menu_opened_at = time.monotonic()
            last_exit_click = time.monotonic()
        elif time.monotonic() - menu_opened_at >= 2.0 and menu_retries < 2 and exit_clicks == 0:
            menu_retries += 1
            logger.info("战斗菜单尚未出现，重试 Esc (%s/3)", menu_retries + 1)
            trace.record(image, state=state, action="retry_escape_menu", attempt=attempt, phase="before")
            open_menu()
            pending_action = ("retry_escape_menu", attempt)
            menu_opened_at = time.monotonic()
        else:
            trace.record(image, state=state, action="await_exit_menu", attempt=attempt)
    logger.warning("快速战斗未确认死亡后续局入口或离战场景；保留当前局数并重新判断")
    if pending_action is not None and trace.root is not None and not _operation_paused(should_abort):
        try:
            time.sleep(0.25)
            if not _operation_paused(should_abort):
                image = _capture(hwnd)
                state = vision.classify_screen(image)
                action, action_attempt = pending_action
                trace.record(image, state=state, action=action, attempt=action_attempt, phase="after")
        except CaptureFault:
            image = None
    trace.record(image, state=state, action="closure_unconfirmed", attempt=attempt, phase="result")
    return ScreenState.UNKNOWN


def handle_post_battle(
    hwnd=None,
    *,
    vision=None,
    backend=None,
    max_steps=6,
    should_abort=None,
    diagnostic_dir=None,
):
    """Advance result screens without blind center clicks.

    Escape menus are treated as active-battle states: the bot resumes the
    battle once, then reports ``False`` instead of clicking an exit choice.
    """
    vision = vision or Vision()
    trace = BlockageTrace(diagnostic_dir, "post_battle_navigation")
    pending_action = None
    image = None
    state = ScreenState.UNKNOWN

    def capture_action_result(action, action_attempt):
        if trace.root is None or _operation_paused(should_abort):
            return
        time.sleep(0.25)
        if _operation_paused(should_abort):
            return
        try:
            after = _capture(hwnd)
            after_state = vision.classify_screen(after)
        except CaptureFault:
            return
        trace.record(after, state=after_state, action=action, attempt=action_attempt, phase="after")
        trace.record(after, state=after_state, action="navigation_result", attempt=action_attempt, phase="result")
    return_requested = False
    pending_overlay = ScreenState.UNKNOWN
    pending_overlay_frames = 0
    for attempt in range(max_steps):
        if _operation_paused(should_abort):
            logger.info("[USER] 结算导航暂停，不切窗口、不点击")
            return False
        try:
            image = _capture(hwnd)
        except CaptureFault as error:
            logger.info("结算导航画面暂不可用，保留当前页面: %s", error)
            return False
        state = vision.classify_screen(image)
        if pending_action is not None:
            action, action_attempt = pending_action
            trace.record(image, state=state, action=action, attempt=action_attempt, phase="after")
            pending_action = None
        trace.record(image, state=state, action="inspect", attempt=attempt)
        if is_port_exit_confirmation_page(image, backend):
            trace.record(image, state=state, action="cancel_port_exit", attempt=attempt, phase="before")
            pending_action = ("cancel_port_exit", attempt)
            if dismiss_port_exit_confirmation(
                hwnd,
                image,
                backend=backend,
                should_abort=should_abort,
            ):
                logger.info("结算导航已取消港口退出游戏确认框")
                time.sleep(0.8)
                continue
            logger.warning("结算导航未能取消港口退出确认框")
            return False
        logger.info("结算导航识别: %s", state.value)
        if state in {ScreenState.EXIT_CONFIRMATION, ScreenState.ESCAPE_MENU}:
            if state == pending_overlay:
                pending_overlay_frames += 1
            else:
                pending_overlay = state
                pending_overlay_frames = 1
            if pending_overlay_frames < 2:
                logger.info("覆盖层仅识别到一帧，等待复核后再点击")
                time.sleep(0.25)
                continue
        else:
            pending_overlay = ScreenState.UNKNOWN
            pending_overlay_frames = 0
        if state == ScreenState.PORT:
            trace.record(image, state=state, action="port_confirmed", attempt=attempt, phase="result")
            return True
        if state == ScreenState.LOADING:
            time.sleep(2)
            continue
        if state == ScreenState.RESULTS:
            if _operation_paused(should_abort):
                return False
            if return_requested:
                logger.info("已派发回港点击，结算页仍在过渡；等待而不重复点击")
                time.sleep(1)
                continue
            action_point = (
                results_return_to_port_action_point(image, backend)
                if backend is not None
                else None
            )
            if backend is not None and action_point is None:
                # RESULTS itself requires both the teal return button and the
                # orange requeue button in their independent relative areas.
                # That positive visual signature makes the resolution-scaled
                # return region a safe fallback when OCR misses the glyphs.
                logger.warning(
                    "结算页 OCR 未定位到“回到港口”；已由双按钮色彩确认，"
                    "使用相对按钮区域兜底"
                )
            if action_point is not None:
                trace.record(image, state=state, action="return_to_port", attempt=attempt, phase="before")
                clicked = _click_local(hwnd, action_point)
            else:
                trace.record(image, state=state, action="return_to_port", attempt=attempt, phase="before")
                clicked = _click_region(
                    hwnd,
                    image,
                    RESULTS_RETURN_TO_PORT_BUTTON,
                )
            if not clicked:
                if _operation_paused(should_abort):
                    return False
                logger.warning("结算页“回到港口”点击未派发，保留页面重试")
                time.sleep(0.5)
                continue
            return_requested = True
            pending_action = ("return_to_port", attempt)
            time.sleep(2)
            continue
        if state == ScreenState.EXIT_CONFIRMATION:
            if _operation_paused(should_abort):
                return False
            if not is_early_exit_confirmation_page(image, backend):
                logger.warning("离开战斗确认框缺少 OCR 复核，拒绝固定位置点击")
                return False
            trace.record(image, state=state, action="cancel_battle_exit", attempt=attempt, phase="before")
            _click_region(hwnd, image, EXIT_CONTINUE_BUTTON)
            logger.warning("检测到离开战斗确认框，已选择继续战斗")
            capture_action_result("cancel_battle_exit", attempt)
            return False
        if state == ScreenState.ESCAPE_MENU:
            if _operation_paused(should_abort):
                return False
            trace.record(image, state=state, action="resume_battle", attempt=attempt, phase="before")
            _click_region(hwnd, image, ESCAPE_RESUME_BUTTON)
            logger.warning("检测到战斗菜单，已返回游戏")
            capture_action_result("resume_battle", attempt)
            return False
        if return_requested:
            logger.info("回港点击后的过渡页面尚未识别，继续等待港口")
            time.sleep(1)
            continue
        logger.warning("未知界面，不执行猜测性点击")
        trace.record(image, state=state, action="navigation_unknown", attempt=attempt, phase="result")
        return False
    if pending_action is not None:
        capture_action_result(*pending_action)
    else:
        trace.record(image, state=state, action="navigation_unconfirmed", attempt=max_steps, phase="result")
    return False
