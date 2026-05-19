"""图像识别系统 - 检测游戏元素"""

import cv2
import numpy as np
from core.state import GameState, Enemy


class Recognizer:
    def __init__(self):
        # 红色HSV范围 (敌舰在小地图上的颜色)
        self.red_lower1 = np.array([0, 120, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 120, 100])
        self.red_upper2 = np.array([180, 255, 255])

        # 绿色HSV范围 (装填完成/血条)
        self.green_lower = np.array([35, 100, 100])
        self.green_upper = np.array([85, 255, 255])

        # 黄色HSV范围 (鱼雷轨迹)
        self.yellow_lower = np.array([15, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])

        # 上一帧的小地图敌人位置 (用于计算航向)
        self.prev_minimap_enemies = []

    def detect_enemies_on_minimap(self, minimap_img):
        """在小地图上检测敌方红点"""
        hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)

        # 红色掩码 (红色在HSV中有两个范围)
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = mask1 | mask2

        # 形态学处理去噪
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # 找轮廓 -> 每个轮廓是一个敌舰
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        enemies = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:  # 过滤噪点
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            enemies.append(Enemy(x=cx, y=cy))

        return enemies

    def detect_torpedoes_on_minimap(self, minimap_img):
        """在小地图上检测鱼雷轨迹 (黄色线条)"""
        hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        torpedo_found = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20:  # 鱼雷轨迹有一定面积
                torpedo_found = True
                break

        return torpedo_found

    def detect_reload_status(self, reload_bar_img):
        """检测装填条是否已满 (绿色占比)"""
        hsv = cv2.cvtColor(reload_bar_img, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        total_pixels = green_mask.shape[0] * green_mask.shape[1]
        green_pixels = np.count_nonzero(green_mask)

        if total_pixels == 0:
            return False

        green_ratio = green_pixels / total_pixels
        return green_ratio > 0.7  # 绿色占比>70%认为装填完成

    def detect_health(self, health_bar_img):
        """检测血量百分比"""
        # 血条通常是红色/橙色条
        hsv = cv2.cvtColor(health_bar_img, cv2.COLOR_BGR2HSV)

        # 红色+橙色血量
        red_lower = np.array([0, 80, 80])
        red_upper = np.array([20, 255, 255])
        orange_lower = np.array([10, 80, 80])
        orange_upper = np.array([25, 255, 255])

        mask = cv2.inRange(hsv, red_lower, red_upper) | cv2.inRange(hsv, orange_lower, orange_upper)

        # 也检测白色(满血时血条可能偏白)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
        mask = mask | white_mask

        total_width = mask.shape[1]
        if total_width == 0:
            return 1.0

        # 找最右侧的有色像素
        col_sums = np.sum(mask > 0, axis=0)
        filled_cols = np.count_nonzero(col_sums)

        return min(filled_cols / total_width, 1.0)

    def detect_distance_ocr(self, distance_img):
        """OCR读取准心下方的距离数字 (简单模板匹配版本)"""
        # 简化版：通过像素亮度判断是否有距离显示
        gray = cv2.cvtColor(distance_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        white_pixels = np.count_nonzero(binary)
        total_pixels = binary.shape[0] * binary.shape[1]

        if white_pixels / total_pixels > 0.05:
            # 有文字显示，返回一个估算距离
            # 实际需要PaddleOCR来精确读取
            return 10.0  # 默认10km

        return 0.0

    def detect_battle_ended(self, full_screen_img):
        """检测战斗是否结束 (检测结算界面)"""
        # 简化版：检测屏幕中央是否有大面积文字
        h, w = full_screen_img.shape[:2]
        center_region = full_screen_img[h // 3:2 * h // 3, w // 3:2 * w // 3]

        gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        white_ratio = np.count_nonzero(binary) / binary.size

        # 结算界面通常有大量白色文字
        return white_ratio > 0.15

    def find_nearest_enemy(self, minimap_img, my_position=None):
        """找到小地图上最近的敌人"""
        enemies = self.detect_enemies_on_minimap(minimap_img)

        if not enemies:
            return None

        if my_position is None:
            # 默认己方位置在小地图中央
            h, w = minimap_img.shape[:2]
            my_position = (w // 2, h // 2)

        # 按距离排序
        def dist(enemy):
            return ((enemy.x - my_position[0]) ** 2 +
                    (enemy.y - my_position[1]) ** 2) ** 0.5

        enemies.sort(key=dist)
        return enemies[0]
