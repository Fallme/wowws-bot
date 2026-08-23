"""State-aware port and menu navigation.

All click locations are relative to the captured game window.  A click is only
issued after the corresponding screen state or button colour has been seen.
"""

import ctypes
import ctypes.wintypes
import logging
import os
import time
from pathlib import Path

import cv2
import mss
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
from core.vision import Vision
from core.window import (
    activate_window,
    get_window_rect,
    physical_click,
    physical_scroll,
)
from dxgi_capture import ScreenCapture

logger = logging.getLogger("port")

BASE_DIR = Path(__file__).resolve().parent
SHIP_TEMPLATE_DIR = BASE_DIR / "assets" / "ui"
LAST_SELECTED_SHIP_PATH = BASE_DIR / "data" / "last_selected_ship.txt"
SUPPORTED_MODES = {"cooperative", "asymmetric"}
SUPPORTED_SHIPS = frozenset(SHIP_NAME_TEMPLATES)

# The mode emblem is the most stable visual element across Chinese UI text
# updates.  OpenCV hue: purple sits around 130-165; co-op is a dark teal/gray
# anchor, so it is selected from its stable first-card slot and then verified
# in the port header.
ASYMMETRIC_PURPLE_LOWER = np.array([125, 65, 45])
ASYMMETRIC_PURPLE_UPPER = np.array([170, 255, 255])
_WINDOW_CAPTURE = ScreenCapture()


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
    if hwnd:
        image = _WINDOW_CAPTURE.capture_window(hwnd)
        if image is None:
            raise RuntimeError(
                f"游戏窗口截取失败: {_WINDOW_CAPTURE.last_error or 'unknown'}"
            )
        return image
    with mss.MSS() as capture:
        monitor = {
            "left": 0,
            "top": 0,
            "width": ctypes.windll.user32.GetSystemMetrics(0),
            "height": ctypes.windll.user32.GetSystemMetrics(1),
        }
        return np.array(capture.grab(monitor))[:, :, :3]


def _screen_origin(hwnd):
    if not hwnd:
        return 0, 0
    rect = get_window_rect(hwnd)
    return rect["left"], rect["top"]


def _click_local(hwnd, point):
    if hwnd:
        # The web panel or Codex preview can become foreground between screen
        # recognition and the click. Re-activate immediately before every UI
        # action so a physical click cannot land on the panel above the game.
        if not activate_window(hwnd):
            return False
        time.sleep(0.12)
    origin_x, origin_y = _screen_origin(hwnd)
    return physical_click(origin_x + point[0], origin_y + point[1])


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
    logger.info("点击“加入战斗”: local=%s", position)
    return _click_local(hwnd, position)


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


def select_mode_from_screen(hwnd, requested_mode):
    """Select one of the two supported PvE cards on the opened menu."""
    image = _capture(hwnd)
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
    """Select and verify cooperative/asymmetric mode from screenshots."""
    requested_mode = (requested_mode or "asymmetric").strip().lower()
    if requested_mode not in SUPPORTED_MODES:
        logger.error("不支持的战斗模式: %s", requested_mode)
        return False
    image = _capture(hwnd)
    current = detect_port_mode(image)
    logger.info("港口模式识别: %s (目标=%s)", current or "不确定", requested_mode)
    if current == requested_mode:
        return True
    if not open_mode_selector(hwnd, vision=vision):
        return False
    time.sleep(1.2)
    if not select_mode_from_screen(hwnd, requested_mode):
        return False
    time.sleep(1.8)
    selected = detect_port_mode(_capture(hwnd))
    if selected != requested_mode:
        logger.warning("模式选择复核失败: 识别=%s 目标=%s", selected, requested_mode)
        return False
    return True


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


def select_requested_ship(hwnd=None, ship_key="pommern", vision=None):
    """Select Napoli/Pommern, scrolling the carousel until it is visible."""
    ship_key = (ship_key or "").strip().lower()
    if ship_key not in SUPPORTED_SHIPS:
        logger.error("不支持的舰船: %s", ship_key)
        return False
    image = _capture(hwnd)
    vision = vision or Vision()
    if vision.classify_screen(image) != ScreenState.PORT:
        logger.warning("当前不是港口，拒绝选择舰船")
        return False
    if is_requested_ship_selected(image, ship_key):
        logger.info("目标舰船已选中: %s", ship_key)
        _remember_selected_ship(ship_key)
        return True
    match = find_ship_card(image, ship_key)
    if match is None and hwnd:
        rect = get_window_rect(hwnd)
        carousel_x = rect["left"] + rect["width"] // 2
        carousel_y = rect["top"] + int(rect["height"] * 0.90)
        # The port carousel converts wheel input into horizontal card paging.
        # Search one direction first, then reverse to cover ships on both sides
        # of the initially visible cards.
        for attempt in range(12):
            direction = -4 if attempt < 7 else 6
            logger.info(
                "未发现舰船 %s，滚动港口舰船栏 (%s/12)",
                ship_key,
                attempt + 1,
            )
            activate_window(hwnd)
            if not physical_scroll(carousel_x, carousel_y, direction):
                break
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
    """Click ``加入战斗`` only after confirming the port and colored button."""
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

    if hwnd:
        activate_window(hwnd)
        time.sleep(0.3)
    image = _capture(hwnd)
    clicked = click_battle(hwnd, image)
    if clicked:
        time.sleep(1.2)
        confirmation = _capture(hwnd)
        if confirm_no_commander(hwnd, confirmation, vision):
            time.sleep(1)
    return clicked


def queue_next_battle(hwnd=None, *, vision=None):
    """Queue the next battle from a positively identified result screen.

    The orange button is used only after the complete result-screen colour
    signature has been verified.  Callers can safely fall back to the existing
    return-to-port path when this returns ``False``.
    """
    vision = vision or Vision()
    if hwnd:
        activate_window(hwnd)
        time.sleep(0.3)
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
    if hwnd:
        activate_window(hwnd)
        time.sleep(0.3)

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
