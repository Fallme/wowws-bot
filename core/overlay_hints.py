"""Recognize explicit Esc-to-close hints in the two upper game-window corners."""

import logging
import math
import re
import unicodedata

import cv2
import numpy as np


logger = logging.getLogger("overlay_hints")
_CLOSE_WORDS = frozenset({"x", "×", "关闭", "關閉", "返回", "取消", "back", "close"})
_FORBIDDEN_WORDS = (
    "离开", "離開", "退出", "菜单", "菜單", "设置", "設置",
    "leave", "exit", "menu", "setting", "options",
)


def _normalized(text):
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"[^a-z0-9\u3400-\u9fff×]+", "", value)


def _tokens_in_crop(tokens, width, height):
    result = []
    for token in tokens or ():
        try:
            confidence = float(token.confidence)
            box = np.asarray(token.box, dtype=float)
            if not np.isfinite(confidence) or confidence < .75 or box.shape != (4, 2) or not np.all(np.isfinite(box)):
                continue
            x1, y1 = box.min(axis=0)
            x2, y2 = box.max(axis=0)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                continue
            result.append((_normalized(token.text), float(x1), float(y1), float(x2), float(y2)))
        except (AttributeError, TypeError, ValueError):
            continue
    return result


def _adjacent(key, other):
    _, x1, y1, x2, y2 = key
    _, ox1, oy1, ox2, oy2 = other
    gap = max(0, ox1 - x2, x1 - ox2)
    vertical_overlap = min(y2, oy2) - max(y1, oy1)
    same_line = (
        vertical_overlap >= min(y2 - y1, oy2 - oy1) * .25
        and abs((y1 + y2 - oy1 - oy2) / 2) <= max(y2 - y1, oy2 - oy1) * .6
    )
    return same_line and gap <= (x2 - x1) * 3


def port_subpage_title(image, backend) -> str | None:
    """Identify known port subpages before their artwork acquires battle control.

    A painted keycap is only a cheap OCR prefilter. A back link and a known
    page title must both be read; Esc authorization remains with the separate
    ``escape_close_hint`` and live numeric-HUD veto.
    """
    if (backend is None or not callable(getattr(backend, "recognize", None))
            or not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.size == 0):
        return None
    height, width = image.shape[:2]
    badge = image[:round(height * .07), round(width * .82):]
    if badge.size == 0:
        return None
    hsv = cv2.cvtColor(badge, cv2.COLOR_BGR2HSV)
    pale = np.uint8((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 165)) * 255
    _, _, components, _ = cv2.connectedComponentsWithStats(pale)
    if not any(
        8 <= w <= width * .06 and 6 <= h <= height * .05
        and 1.25 <= w / h <= 3.8 and area / (w * h) >= .52
        and y + h / 2 <= height * .055
        for _, y, w, h, area in components[1:]
    ):
        return None
    try:
        left = image[:round(height * .10), :round(width * .26)]
        left_text = "".join(token[0] for token in _tokens_in_crop(
            backend.recognize(left), left.shape[1], left.shape[0],
        ))
        port_back = any(word in left_text for word in (
            "返回港口", "回到港口", "backtoport", "returntoport",
        ))
        if port_back:
            right = image[:round(height * .18), round(width * .72):]
            right_text = "".join(token[0] for token in _tokens_in_crop(
                backend.recognize(right), right.shape[1], right.shape[0],
            ))
            if any(word in right_text for word in ("补给箱", "補給箱", "containers")):
                return "补给箱"
        if not port_back and not any(word in left_text for word in ("返回", "back")):
            return None
        heading = image[:round(height * .12), round(width * .35):round(width * .65)]
        heading_text = "".join(token[0] for token in _tokens_in_crop(
            backend.recognize(heading), heading.shape[1], heading.shape[0],
        ))
        if any(word in heading_text for word in ("经济加成", "經濟加成", "economicbonuses")):
            return "经济加成"
        # Shop artwork changes often. Read persistent shell controls rather
        # than product names or prices, which must never become actions.
        shop = image[:round(height * .12), round(width * .38):round(width * .86)]
        shop_text = "".join(token[0] for token in _tokens_in_crop(
            backend.recognize(shop), shop.shape[1], shop.shape[0],
        ))
        search = any(word in shop_text for word in (
            "按名称搜索", "按名稱搜尋", "按名稱搜索", "searchbyname",
        ))
        commerce = any(word in shop_text for word in (
            "优惠券", "優惠券", "coupons", "购买达布隆", "購買達布隆",
            "buydoubloons", "purchasedoubloons", "钱包", "錢包", "wallet",
        ))
        if search and commerce:
            return "商店"
        return None
    except Exception:
        logger.debug("港口子页面返回链接/标题 OCR 暂不可用", exc_info=True)
        return None


def is_port_container_page(image, backend) -> bool:
    """Compatibility predicate for the positively identified container page."""
    return port_subpage_title(image, backend) == "补给箱"


def escape_close_hint(image, backend) -> tuple[int, int] | None:
    """Return a confirmed close-hint center, not authorization to click it.

    Reuse the caller's OCR backend. Only Esc paired with explicit close/back
    semantics in a top corner qualifies; battle-exit/menu prompts never do.
    Missing or ambiguous evidence and OCR failures all return None.
    """
    if backend is None or not callable(getattr(backend, "recognize", None)):
        return None
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return None
    height, width = image.shape[:2]
    top_height = int(height * .14)
    if top_height < 1:
        return None
    candidates = set()
    corner_candidates = set()
    for start, end in ((0, int(width * .26)), (int(width * .72), width)):
        crop = image[:top_height, start:end]
        if crop.size == 0:
            return None
        try:
            tokens = _tokens_in_crop(
                backend.recognize(crop), end - start, top_height,
            )
        except Exception:
            logger.debug("顶部 Esc 关闭提示 OCR 暂不可用", exc_info=True)
            return None
        for key in tokens:
            text = key[0]
            if text.count("esc") != 1:
                continue
            remainder = text.replace("esc", "", 1)
            if remainder and remainder not in _CLOSE_WORDS:
                continue
            nearby = [token for token in tokens if _adjacent(key, token)]
            if any(word in token[0] for token in nearby for word in _FORBIDDEN_WORDS):
                continue
            if remainder in _CLOSE_WORDS:
                closes = [key]
            else:
                closes = [token for token in nearby if token[0] in _CLOSE_WORDS]
                # The stock close badge is a painted Esc keycap plus a ×
                # glyph that OCR often misses. The keycap sits at the far
                # right of the top bar (0.84–0.97 of the width on different
                # pages), so accept a lone ESC there and at the far left;
                # anything closer to the middle stays unverified.
                _, x1, y1, x2, y2 = key
                in_outer_corner = (
                    start + x2 >= width * .84 or start + x1 <= width * .08
                ) and (y1 + y2) / 2 <= top_height * .55
                if in_outer_corner:
                    corner_candidates.add((
                        round(start + (x1 + x2) / 2), round((y1 + y2) / 2),
                    ))
            for close in closes:
                candidates.add((round(start + (close[1] + close[3]) / 2), round((close[2] + close[4]) / 2)))
    points = list(candidates or corner_candidates)
    if not points:
        return None
    # Key and × can both answer for the same badge; nearby centers count as
    # one close action, while separated centers remain ambiguous evidence.
    threshold = max(24, width * .02)
    clusters = []
    for point in points:
        for cluster in clusters:
            if any(math.dist(point, other) <= threshold for other in cluster):
                cluster.append(point)
                break
        else:
            clusters.append([point])
    if len(clusters) != 1:
        return None
    cluster = clusters[0]
    return (
        round(sum(point[0] for point in cluster) / len(cluster)),
        round(sum(point[1] for point in cluster) / len(cluster)),
    )
