"""菜单自动化 - 匹配/下一局 (支持副屏)"""

import time
import logging
import cv2
import numpy as np
from core.capture import ScreenCapture
from input_control import InputController

logger = logging.getLogger("menu")


class MenuSystem:
    def __init__(self, capture: ScreenCapture, controller: InputController, screen_config: dict):
        self.capture = capture
        self.controller = controller
        self.screen_config = screen_config
        self.buttons = screen_config.get("buttons", {})

        # 副屏偏移量 (mss monitor的left值)
        monitor = capture.monitor
        self.offset_x = monitor.get("left", 0)
        self.offset_y = monitor.get("top", 0)
        logger.info(f"屏幕偏移: ({self.offset_x}, {self.offset_y})")

    def _abs_coords(self, x, y):
        """将相对坐标转为绝对屏幕坐标 (用于pydirectinput)"""
        return x + self.offset_x, y + self.offset_y

    def click_battle(self):
        """点击'加入战斗'按钮"""
        if "battle" in self.buttons:
            btn = self.buttons["battle"]
            abs_x, abs_y = self._abs_coords(btn["x"], btn["y"])
            logger.info(f"点击'战斗' -> 绝对坐标 ({abs_x}, {abs_y})")
            self.controller.click_at(abs_x, abs_y)
            return True
        return False

    def click_continue(self):
        """点击'继续'按钮"""
        if "continue" in self.buttons:
            btn = self.buttons["continue"]
            abs_x, abs_y = self._abs_coords(btn["x"], btn["y"])
            logger.info(f"点击'继续' -> 绝对坐标 ({abs_x}, {abs_y})")
            self.controller.click_at(abs_x, abs_y)
            return True
        return False

    def wait_for_battle(self, timeout=120):
        """等待战斗开始"""
        logger.info("等待战斗开始...")
        start = time.time()

        while time.time() - start < timeout:
            minimap = self.capture.grab_minimap()
            if minimap is not None and minimap.size > 0:
                logger.info("战斗开始!")
                return True
            time.sleep(1)

        logger.warning("等待超时")
        return False

    def wait_for_battle_end(self, timeout=600):
        """等待战斗结束"""
        logger.info("等待战斗结束...")
        start = time.time()

        while time.time() - start < timeout:
            full_screen = self.capture.grab_full()
            gray = cv2.cvtColor(full_screen, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            h, w = binary.shape
            center = binary[h // 3:2 * h // 3, w // 3:2 * w // 3]
            white_ratio = np.count_nonzero(center) / max(center.size, 1)

            if white_ratio > 0.15:
                logger.info("战斗结束!")
                return True
            time.sleep(1)

        logger.warning("等待超时")
        return False
