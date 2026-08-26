"""State-aware port and menu navigation.

All click locations are relative to the captured game window.  A click is only
issued after the corresponding screen state or button colour has been seen.
"""

import logging
import os
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from core.ui import (
    BATTLE_TYPE_COOPERATIVE_CARD,
    BATTLE_TYPE_SEARCH_AREA,
    ESCAPE_RESUME_BUTTON,
    EXIT_CONTINUE_BUTTON,
    NO_COMMANDER_CONFIRM_BUTTON,
    PORT_BATTLE_BUTTON,
    PORT_DIALOG_CLOSE,
    PORT_MODE_SELECTOR,
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


class ShipSelectionError(RuntimeError):
    """Raised when a requested custom ship cannot be found and verified."""


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
        raise RuntimeError(
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


def click_battle(hwnd=None, image=None):
    image = image if image is not None else _capture(hwnd)
    position = find_battle_button(hwnd, image)
    if position is None:
        logger.warning("未找到蓝色或橙色“加入战斗”按钮")
        return False
    logger.info("定位“加入战斗”: local=%s", position)
    clicked = _click_local(hwnd, position)
    if clicked:
        logger.info("已向“加入战斗”派发一次物理点击: local=%s", position)
    else:
        logger.warning("“加入战斗”物理点击未派发: local=%s", position)
    return clicked


def _observe_battle_entry(hwnd, vision, *, samples=8, interval=0.35):
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
    sample_count = max(1, int(samples))

    for sample in range(sample_count):
        time.sleep(max(0.0, float(interval)))
        try:
            image = _capture(hwnd)
        except Exception as error:
            logger.warning(
                "“加入战斗”已点击，但后续画面暂不可用；保持等待且不重复点击: %s",
                error,
            )
            return True

        no_commander_detector = getattr(
            vision,
            "in_no_commander_confirmation",
            lambda _image: False,
        )
        if no_commander_detector(image):
            if confirm_no_commander(hwnd, image, vision):
                logger.info("无指挥官确认已处理，继续观察入场状态")
                stable_actionable_port = 0
                transition_frames = 0
                continue
            logger.warning("无指挥官确认框已识别，但确认点击未能安全派发")
            return True

        state = vision.classify_screen(image)
        if state in {ScreenState.LOADING, ScreenState.BATTLE}:
            logger.info("“加入战斗”请求已确认: current=%s", state.value)
            return True

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
    """Accept the known no-commander warning only when positively detected."""
    vision = vision or Vision()
    image = image if image is not None else _capture(hwnd)
    if not vision.in_no_commander_confirmation(image):
        return False
    logger.info("检测到无指挥官确认框，选择继续进入战斗")
    return _click_region(hwnd, image, NO_COMMANDER_CONFIRM_BUTTON)


def open_mode_selector(hwnd=None, vision=None):
    """Open the battle-mode selector only while the port is positively seen."""
    image = _capture(hwnd)
    vision = vision or Vision()
    if vision.classify_screen(image) != ScreenState.PORT:
        logger.warning("当前不是港口，拒绝点击模式选择器")
        return False
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


def detect_port_mode(image):
    """Classify the selected PvE mode from its port-header emblem."""
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


def in_battle_type_selector(image):
    """Confirm that the full battle-type card page is still open.

    The old port-header colour check also returned ``cooperative`` on this
    screen, causing the selector to be treated as an already verified port.
    The asymmetric purple card is a stable page-specific anchor and is absent
    from normal port screenshots.
    """
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


def select_mode_from_screen(hwnd, requested_mode, image=None):
    """Select one of the two supported PvE cards on the opened menu."""
    image = image if image is not None else _capture(hwnd)
    if not in_battle_type_selector(image):
        logger.warning("未确认战斗模式选择页，拒绝按固定卡片位置点击")
        return False
    height, width = image.shape[:2]
    if requested_mode == "cooperative":
        point = BATTLE_TYPE_COOPERATIVE_CARD.center(width, height)
    else:
        point = _find_asymmetric_card(image)
        if point is None:
            logger.warning("未在战斗类型页识别到非对称作战紫色徽标")
            return False
    logger.info("按截图定位战斗模式 %s: local=%s", requested_mode, point)
    return _click_local(hwnd, point)


def ensure_requested_mode(hwnd=None, requested_mode="asymmetric", vision=None):
    """Select the configured mode and verify it again after returning to port."""
    requested_mode = (requested_mode or "asymmetric").strip().lower()
    if requested_mode not in SUPPORTED_MODES:
        logger.error("不支持的战斗模式: %s", requested_mode)
        return False
    vision = vision or Vision()
    for attempt in range(1, 4):
        image = _capture(hwnd)
        selector_open = in_battle_type_selector(image)
        current = None if selector_open else detect_port_mode(image)
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
            if vision.classify_screen(image) != ScreenState.PORT:
                logger.warning("当前不是港口或模式选择页，本轮不执行模式点击")
                return False
            if not _click_region(hwnd, image, PORT_MODE_SELECTOR):
                continue
            time.sleep(1.0)
            image = _capture(hwnd)
            selector_open = in_battle_type_selector(image)

        if not selector_open:
            logger.warning("点击港口右侧模式入口后，未确认模式选择页")
            time.sleep(0.5)
            continue
        if not select_mode_from_screen(hwnd, requested_mode, image=image):
            time.sleep(0.5)
            continue

        # Selection animation and port restoration are asynchronous. Require
        # both selector closure and a matching right-side port emblem.
        for verification in range(6):
            time.sleep(0.5 if verification else 1.0)
            confirmation = _capture(hwnd)
            if in_battle_type_selector(confirmation):
                continue
            if vision.classify_screen(confirmation) != ScreenState.PORT:
                continue
            selected = detect_port_mode(confirmation)
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


def _token_geometry(token):
    if not token.box:
        return None
    xs = [point[0] for point in token.box]
    ys = [point[1] for point in token.box]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _ocr_name_matches(
    image,
    full_name,
    backend,
    minimum_confidence=0.60,
    *,
    allow_tier_prefix=False,
):
    """Return exact normalized OCR matches, joining adjacent split tokens."""
    wanted = _normalize_ship_name(full_name)
    if not wanted:
        return []
    items = []
    for token in backend.recognize(image):
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
                exact_match = combined == wanted
                tiered_match = allow_tier_prefix and any(
                    combined == prefix + wanted for prefix in SHIP_TIER_PREFIXES
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
                maximum_length = len(wanted) + (4 if allow_tier_prefix else 0)
                if len(combined) >= maximum_length:
                    break
    return matches


def find_custom_ship_card(image, full_name, backend=None, minimum_confidence=0.60):
    """Find an exact custom ship name in the bottom port carousel via OCR."""
    backend = backend or RapidOcrBackend()
    height = image.shape[0]
    search_top = int(height * 0.735)
    matches = _ocr_name_matches(
        image[search_top:height, :],
        full_name,
        backend,
        minimum_confidence,
    )
    if not matches:
        return None
    box, confidence, raw_text = max(matches, key=lambda match: match[1])
    x1, y1, x2, y2 = box
    name_y = search_top + int((y1 + y2) / 2)
    point = (int((x1 + x2) / 2), max(search_top, name_y - int(height * 0.012)))
    logger.info(
        "自定义舰船 OCR 命中: requested=%s recognized=%s confidence=%.3f",
        full_name,
        raw_text,
        confidence,
    )
    return point, confidence


def is_custom_ship_selected(image, full_name, backend=None, minimum_confidence=0.60):
    """Strictly verify the selected ship name in the upper-right detail panel."""
    backend = backend or RapidOcrBackend()
    height = image.shape[0]
    # RapidOCR needs the surrounding header context to detect the relatively
    # small title glyphs reliably. A tight upper-right crop can drop a clearly
    # visible title; the upper quarter contains no carousel cards and remains
    # unambiguous for an exact ship-name check.
    detail = image[: int(height * 0.25), :]
    matches = _ocr_name_matches(
        detail,
        full_name,
        backend,
        minimum_confidence,
        allow_tier_prefix=True,
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


def find_ship_card(image, ship_key, minimum_score=0.55):
    """Find a supported ship by matching its gold name glyphs in the carousel."""
    ship_key = (ship_key or "").strip().lower()
    if ship_key not in SUPPORTED_SHIPS:
        raise ValueError(f"Unsupported ship: {ship_key}")
    template = _load_ship_name_template(ship_key)
    if template is None or template.size == 0:
        return None

    height, width = image.shape[:2]
    search_top = int(height * 0.735)
    search = image[search_top:height, :]
    target_scale = width / SHIP_REFERENCE_SIZE[0]
    template = cv2.resize(
        template,
        None,
        fx=target_scale,
        fy=target_scale,
        interpolation=cv2.INTER_AREA if target_scale < 1 else cv2.INTER_CUBIC,
    )
    search_mask = _gold_name_mask(search)
    template_mask = _tighten_mask(_gold_name_mask(template))
    if template_mask.size == 0 or np.count_nonzero(template_mask) < 20:
        return None
    result = cv2.matchTemplate(search_mask, template_mask, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    if score < minimum_score:
        logger.warning("舰船卡片 %s 匹配置信度不足: %.3f", ship_key, score)
        return None
    template_height, template_width = template_mask.shape
    x = location[0] + template_width // 2
    name_y = search_top + location[1] + template_height // 2
    # Click the card body above its name, avoiding adjacent card borders.
    card_y = max(search_top, name_y - int(height * 0.012))
    return (x, card_y), float(score)


def selected_ship_scores(image):
    """Score supported ship names in the port's upper-right detail panel."""
    height, width = image.shape[:2]
    search = image[
        int(height * 0.055) : int(height * 0.16),
        int(width * 0.82) : width,
    ]
    search_mask = _gold_name_mask(search)
    scale = width / SHIP_REFERENCE_SIZE[0]
    scores = {}
    for ship_key in SUPPORTED_SHIPS:
        template = _load_selected_ship_name_template(ship_key)
        if template is None or template.size == 0:
            scores[ship_key] = 0.0
            continue
        if scale != 1.0:
            template = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
        if (
            search_mask.shape[0] < template.shape[0]
            or search_mask.shape[1] < template.shape[1]
        ):
            scores[ship_key] = 0.0
            continue
        result = cv2.matchTemplate(
            search_mask,
            template,
            cv2.TM_CCOEFF_NORMED,
        )
        scores[ship_key] = float(cv2.minMaxLoc(result)[1])
    return scores


def is_requested_ship_selected(
    image,
    ship_key,
    minimum_score=0.72,
    minimum_margin=0.15,
):
    """Verify the selected ship from its detail-panel title, not card styling."""
    ship_key = (ship_key or "").strip().lower()
    if ship_key not in SUPPORTED_SHIPS:
        raise ValueError(f"Unsupported ship: {ship_key}")
    scores = selected_ship_scores(image)
    target_score = scores[ship_key]
    competitor_score = max(
        score for key, score in scores.items() if key != ship_key
    )
    logger.info(
        "舰船详情标题复核 %s: target=%.3f competitor=%.3f",
        ship_key,
        target_score,
        competitor_score,
    )
    return (
        target_score >= minimum_score
        and target_score - competitor_score >= minimum_margin
    )


def _confirm_custom_ship_after_click(
    hwnd,
    point,
    full_name,
    backend,
    *,
    click_attempts=2,
    confirmation_seconds=4.0,
):
    """Try the chosen card twice, then stop; never scroll after a card hit."""
    for click_attempt in range(1, max(1, int(click_attempts)) + 1):
        logger.info(
            "点击自定义舰船 %s (%s/%s)",
            full_name,
            click_attempt,
            click_attempts,
        )
        if not _click_local(hwnd, point):
            continue
        deadline = time.monotonic() + max(0.5, float(confirmation_seconds))
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if is_custom_ship_selected(_capture(hwnd), full_name, backend):
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


def _rewind_ship_carousel(hwnd, rect, *, steps=20):
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
        if not physical_scroll(carousel_x, carousel_y, 8, hwnd=hwnd):
            break
        moved = True
        time.sleep(0.18)
    return moved


def _scroll_ship_carousel_down(hwnd, rect, *, step=6):
    """Advance one page through the carousel after it has been rewound."""
    carousel_x = rect["left"] + rect["width"] // 2
    carousel_y = rect["top"] + int(rect["height"] * 0.90)
    return physical_scroll(carousel_x, carousel_y, -abs(int(step)), hwnd=hwnd)


def _select_custom_ship(hwnd, image, full_name, backend, max_scrolls=18):
    if is_custom_ship_selected(image, full_name, backend):
        logger.info("目标自定义舰船已选中: %s", full_name)
        _remember_selected_ship(f"custom:{full_name}")
        return True

    rect = get_client_rect(hwnd) if hwnd else None
    if hwnd and rect:
        _rewind_ship_carousel(hwnd, rect)
        image = _capture(hwnd)
        if is_custom_ship_selected(image, full_name, backend):
            logger.info("回到舰船栏起点后已确认目标自定义舰船: %s", full_name)
            _remember_selected_ship(f"custom:{full_name}")
            return True
    match = find_custom_ship_card(image, full_name, backend)
    for attempt in range(max_scrolls + 1):
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
        image = _capture(hwnd)
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
):
    """Select a built-in template ship or an exact custom ship OCR name."""
    ship_key = (ship_key or "").strip().lower()
    is_custom = ship_key == CUSTOM_SHIP_KEY
    if ship_key not in SUPPORTED_SHIPS and not is_custom:
        logger.error("不支持的舰船: %s", ship_key)
        return False
    image = _capture(hwnd)
    vision = vision or Vision()
    if vision.classify_screen(image) != ScreenState.PORT:
        logger.warning("当前不是港口，拒绝选择舰船")
        return False
    if is_custom:
        full_name = (custom_name or os.environ.get("WOWS_CUSTOM_SHIP_NAME", "")).strip()
        if not full_name:
            raise ShipSelectionError("自定义舰船名称为空，请返回网页重新选择")
        return _select_custom_ship(
            hwnd,
            image,
            full_name,
            ocr_backend or RapidOcrBackend(),
            max_scrolls=max(0, int(custom_max_scrolls)),
        )
    if is_requested_ship_selected(image, ship_key):
        logger.info("目标舰船已选中: %s", ship_key)
        _remember_selected_ship(ship_key)
        return True
    match = find_ship_card(image, ship_key)
    if match is None and hwnd:
        rect = get_client_rect(hwnd)
        _rewind_ship_carousel(hwnd, rect)
        image = _capture(hwnd)
        if is_requested_ship_selected(image, ship_key):
            _remember_selected_ship(ship_key)
            return True
        match = find_ship_card(image, ship_key)
        # A single direction sweep avoids revisiting cards and never leaves the
        # requested ship just outside a reverse-search boundary.
        for attempt in range(18):
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
            image = _capture(hwnd)
            if is_requested_ship_selected(image, ship_key):
                _remember_selected_ship(ship_key)
                return True
            match = find_ship_card(image, ship_key)
            if match is not None:
                break
    if match is None:
        if _last_confirmed_ship() == ship_key:
            logger.warning(
                "滚动搜索后舰名区域仍暂不可读，沿用上次确认舰船: %s",
                ship_key,
            )
            return True
        return False
    point, score = match
    logger.info("按截图定位舰船 %s: local=%s score=%.3f", ship_key, point, score)
    if not _click_local(hwnd, point):
        return False
    time.sleep(1.8)
    if not is_requested_ship_selected(_capture(hwnd), ship_key):
        logger.warning("舰船选择复核失败: %s", ship_key)
        return False
    _remember_selected_ship(ship_key)
    return True


def click_center(hwnd=None):
    image = _capture(hwnd)
    height, width = image.shape[:2]
    return _click_local(hwnd, (width // 2, height // 2))


def enter_battle(hwnd=None, *, vision=None, configure_port=True):
    """Click ``加入战斗`` once, then verify or conservatively await transition."""
    vision = vision or Vision()
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
        ):
            logger.warning("未能安全选择目标舰船")
            return False

        if not ensure_requested_mode(
            hwnd,
            os.environ.get("WOWS_MODE", "asymmetric"),
            vision,
        ):
            logger.warning("未能安全选择目标战斗模式")
            return False

    image = _capture(hwnd)
    if not click_battle(hwnd, image):
        return False
    return _observe_battle_entry(hwnd, vision)


def queue_next_battle(hwnd=None, *, vision=None):
    """Queue the next battle from a positively identified result screen.

    The orange button is used only after the complete result-screen colour
    signature has been verified.  Callers can safely fall back to the existing
    return-to-port path when this returns ``False``.
    """
    vision = vision or Vision()
    image = _capture(hwnd)
    state = vision.classify_screen(image)
    if state != ScreenState.RESULTS:
        logger.warning("当前界面为 %s，不点击“继续战斗”", state.value)
        return False
    logger.info("结算界面已确认，点击“继续战斗”自动进入下一局")
    if not _click_region(hwnd, image, RESULTS_REQUEUE_BUTTON):
        return False
    # Verify that the click actually left the result page.  Some result pages
    # are still animating when the button first appears, so retry once instead
    # of reporting success for a click the game ignored.
    retried = False
    for attempt in range(24):
        time.sleep(0.25)
        confirmation = _capture(hwnd)
        state = vision.classify_screen(confirmation)
        if state in {ScreenState.LOADING, ScreenState.BATTLE}:
            return True
        if state == ScreenState.PORT:
            logger.info("继续战斗返回了港口，改用港口常规入口")
            return False
        if (
            state == ScreenState.UNKNOWN
            and confirm_no_commander(hwnd, confirmation, vision)
        ):
            return True
        if state == ScreenState.RESULTS and attempt >= 7 and not retried:
            logger.info("继续战斗按钮未生效，重新点击一次")
            if not _click_region(hwnd, confirmation, RESULTS_REQUEUE_BUTTON):
                return False
            retried = True
    logger.warning("点击继续战斗后界面未发生变化")
    return False


def handle_post_battle(hwnd=None, *, vision=None, max_steps=4):
    """Advance result screens without blind center clicks.

    Escape menus are treated as active-battle states: the bot resumes the
    battle once, then reports ``False`` instead of clicking an exit choice.
    """
    vision = vision or Vision()
    for _ in range(max_steps):
        image = _capture(hwnd)
        state = vision.classify_screen(image)
        logger.info("结算导航识别: %s", state.value)
        if state == ScreenState.PORT:
            return True
        if state == ScreenState.LOADING:
            time.sleep(2)
            continue
        if state == ScreenState.RESULTS:
            _click_region(hwnd, image, RESULTS_RETURN_TO_PORT_BUTTON)
            time.sleep(2)
            continue
        if state == ScreenState.EXIT_CONFIRMATION:
            _click_region(hwnd, image, EXIT_CONTINUE_BUTTON)
            logger.warning("检测到离开战斗确认框，已选择继续战斗")
            return False
        if state == ScreenState.ESCAPE_MENU:
            _click_region(hwnd, image, ESCAPE_RESUME_BUTTON)
            logger.warning("检测到战斗菜单，已返回游戏")
            return False
        logger.warning("未知界面，不执行猜测性点击")
        return False
    return False
