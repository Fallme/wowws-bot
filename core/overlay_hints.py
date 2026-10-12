"""Recognize explicit Esc-to-close hints in the two upper game-window corners."""

import logging
import re
import unicodedata

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
    for start, end in ((0, int(width * .26)), (int(width * .72), width)):
        crop = image[:top_height, start:end]
        if crop.size == 0:
            return None
        try:
            tokens = _tokens_in_crop(backend.recognize(crop), end - start, top_height)
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
            closes = [key] if remainder in _CLOSE_WORDS else [token for token in nearby if token[0] in _CLOSE_WORDS]
            for close in closes:
                candidates.add((round(start + (close[1] + close[3]) / 2), round((close[2] + close[4]) / 2)))
    return next(iter(candidates)) if len(candidates) == 1 else None
