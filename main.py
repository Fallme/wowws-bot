"""战舰世界副炮流Bot v3 - 完全自动后台挂机
完全不抢焦点、不移动鼠标、不干扰正常工作。
使用 PostMessage 后台点击 + 虚拟手柄战斗控制。
"""

import sys
import time
import yaml
import logging
import ctypes

# === DPI 感知 (必须最早设置) ===
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import mss
import cv2
import numpy as np
import vgamepad as vg
import win32gui
import win32con
from core.window import find_game_window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wowws_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("bot")


def load_config(path, key):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)[key]


# ==================== 后台点击 (PostMessage) ====================

def bg_click(hwnd, rel_x, rel_y):
    """后台发送点击消息 - 不移动鼠标、不抢焦点"""
    lparam = (int(rel_y) << 16) | (int(rel_x) & 0xFFFF)
    try:
        ctypes.windll.user32.PostMessageW(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
        time.sleep(0.05)
        ctypes.windll.user32.PostMessageW(hwnd, win32con.WM_LBUTTONUP, 0, lparam)
        return True
    except Exception:
        return False


def bg_click_center(hwnd):
    """点击窗口中心 (用于'继续'按钮)"""
    ct = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(ct))
    cx = ct.right // 2
    cy = ct.bottom // 2
    return bg_click(hwnd, cx, cy)


def bg_click_top_center(hwnd):
    """点击窗口顶部中间 (用于'加入战斗'按钮)"""
    ct = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(ct))
    cx = ct.right // 2
    return bg_click(hwnd, cx, 37)


# ==================== 虚拟手柄 ====================

class Gamepad:
    def __init__(self):
        self.gp = vg.VX360Gamepad()
        self._lx = 0.0
        self._ly = 0.0

    def _update(self):
        self.gp.left_joystick(x_value=int(self._lx * 32767),
                               y_value=int(self._ly * 32767))
        self.gp.update()

    def steer_left(self, a=0.7):
        self._lx = -a; self._update()

    def steer_right(self, a=0.7):
        self._lx = a; self._update()

    def straight(self):
        self._lx = 0.0; self._update()

    def full_speed(self):
        self._ly = 1.0; self._update()

    def stop(self):
        self._lx = 0.0; self._ly = 0.0; self._update()

    def fire(self):
        self.gp.right_trigger(value=255); self.gp.update()
        time.sleep(0.1)
        self.gp.right_trigger(value=0); self.gp.update()

    def lock(self):
        self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        time.sleep(0.1)
        self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gp.update()

    def torpedo(self):
        self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        time.sleep(0.1)
        self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        self.gp.update()

    def smoke(self):
        self.gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        time.sleep(0.1)
        self.gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gp.update()


# ==================== 截图分析 ====================

class Vision:
    def __init__(self):
        self.sct = mss.MSS()
        self.red_lo1 = np.array([0, 120, 100])
        self.red_hi1 = np.array([10, 255, 255])
        self.red_lo2 = np.array([160, 120, 100])
        self.red_hi2 = np.array([180, 255, 255])
        self.yellow_lo = np.array([15, 100, 100])
        self.yellow_hi = np.array([35, 255, 255])
        self.green_lo = np.array([35, 100, 100])
        self.green_hi = np.array([85, 255, 255])

    def grab(self, hwnd):
        """后台截图 - 不移动窗口、不抢焦点"""
        ct = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(ct))
        mon = {"left": ct.left, "top": ct.top,
               "width": ct.right - ct.left, "height": ct.bottom - ct.top}
        return np.array(self.sct.grab(mon))[:, :, :3]

    def find_enemies(self, minimap):
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.red_lo1, self.red_hi1) | \
               cv2.inRange(hsv, self.red_lo2, self.red_hi2)
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for c in cnts:
            if cv2.contourArea(c) < 15:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            result.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        return result

    def has_torpedoes(self, minimap):
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) > 50 for c in cnts)

    def reload_ready(self, area):
        if area is None or area.size == 0:
            return False
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lo, self.green_hi)
        total = mask.shape[0] * mask.shape[1]
        return (np.count_nonzero(mask) / max(total, 1)) > 0.3

    def health_pct(self, area):
        if area is None or area.size == 0:
            return 0.5
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255])) |
                cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255])) |
                cv2.inRange(hsv, np.array([0, 50, 100]), np.array([20, 255, 255])))
        filled = np.count_nonzero(np.sum(mask > 0, axis=0))
        return min(filled / max(mask.shape[1], 1), 1.0)

    def battle_ended(self, img):
        h, w = img.shape[:2]
        c = img[h // 3:2 * h // 3, w // 3:2 * w // 3]
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return (np.count_nonzero(b) / max(b.size, 1)) > 0.08

    def in_battle(self, img):
        """检测是否在战斗中 (小地图有内容 + 整体不是菜单)"""
        h, w = img.shape[:2]
        mm = img[max(0, h - 300):h - 20, max(0, w - 300):w - 20]
        if mm.size == 0:
            return False
        # 小地图标准差 > 25 说明有内容
        return mm.std() > 25

    def in_port(self, img):
        """检测是否在港口 (看顶部是否有橙色战斗按钮)"""
        h, w = img.shape[:2]
        top = img[0:80, w // 2 - 200:w // 2 + 200]
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([5, 100, 150]), np.array([25, 255, 255]))
        return np.count_nonzero(mask) > 500


# ==================== 主 Bot ====================

class Bot:
    def __init__(self, hwnd, ship_cfg):
        self.hwnd = hwnd
        self.ship = ship_cfg
        self.vision = Vision()
        self.gp = Gamepad()
        self.strategy = ship_cfg.get("strategy", {})

        # 状态
        self.smoke_used = False
        self.last_fire = 0
        self.last_lock = 0
        self.torpedo_evade_until = 0
        self.tick = 0

    def reset(self):
        self.smoke_used = False
        self.last_fire = 0
        self.last_lock = 0
        self.torpedo_evade_until = 0
        self.tick = 0
        self.gp.stop()

    def analyze(self):
        img = self.vision.grab(self.hwnd)
        h, w = img.shape[:2]
        s = {"img": img, "h": h, "w": w, "enemies": [], "torps": False,
             "reload": False, "hp": 0.5, "ended": False, "in_battle": False}

        # 小地图
        mm = img[max(0, h - 300):h - 20, max(0, w - 300):w - 20]
        if mm.size > 0:
            s["enemies"] = self.vision.find_enemies(mm)
            s["torps"] = self.vision.has_torpedoes(mm)

        # 装填条
        rl = img[h - 130:h - 100, w // 2 - 200:w // 2 + 200]
        s["reload"] = self.vision.reload_ready(rl)

        # 血条
        hp = img[h - 170:h - 140, w // 2 - 250:w // 2 + 250]
        s["hp"] = self.vision.health_pct(hp)

        # 战斗结束
        s["ended"] = self.vision.battle_ended(img) and self.tick > 5
        s["in_battle"] = self.vision.in_battle(img)

        return s

    def combat_tick(self):
        s = self.analyze()
        h, w, now = s["h"], s["w"], time.time()

        if s["ended"]:
            return "ended"

        # 鱼雷规避
        if s["torps"] and now > self.torpedo_evade_until:
            logger.info("规避鱼雷!")
            self.gp.steer_left(0.9)
            time.sleep(1.0)
            self.gp.steer_right(0.9)
            time.sleep(0.8)
            self.gp.straight()
            self.torpedo_evade_until = now + 3

        # 全速
        self.gp.full_speed()

        # 找最近敌人并转向
        enemies = s["enemies"]
        if enemies:
            cx, cy = w // 2, h // 2
            nearest = min(enemies, key=lambda e: ((e[0] - cx) ** 2 + (e[1] - cy) ** 2) ** 0.5)
            dx = nearest[0] - cx
            if abs(dx) > 40:
                amt = min(abs(dx) / 250, 1.0)
                (self.gp.steer_right if dx > 0 else self.gp.steer_left)(amt)
            else:
                self.gp.straight()

            # 副炮锁定 (3秒一次)
            if now - self.last_lock > 3:
                self.gp.lock()
                self.last_lock = now
                logger.info(f"[{self.tick:03d}] 副炮锁定")

            # 主炮射击
            if s["reload"] and now - self.last_fire > 2:
                self.gp.fire()
                self.last_fire = now
                logger.info(f"[{self.tick:03d}] 主炮开火!")

        # 烟雾 (只放一次)
        smoke_thr = self.strategy.get("smoke_threshold", 0.5)
        if self.ship.get("has_smoke") and not self.smoke_used and 0 < s["hp"] < smoke_thr:
            self.gp.smoke()
            self.smoke_used = True
            logger.info(f"释放烟雾 (HP: {s['hp']:.0%})")

        # 鱼雷 (Pommern)
        if self.ship.get("has_torpedoes") and s["reload"] and now - self.last_fire > 3:
            self.gp.torpedo()
            logger.info("鱼雷发射!")

        logger.info(f"[{self.tick:03d}] HP:{s['hp']:.0%} 装填:{'OK' if s['reload'] else '中'} "
                     f"敌人:{len(enemies)} 鱼雷:{'有' if s['torps'] else '无'}")
        self.tick += 1
        return "combat"


def main():
    ship_key = "napoli"
    ship_cfg = load_config("config/ship.yaml", ship_key)
    logger.info(f"船: {ship_cfg['name']} | 副炮射程: {ship_cfg['secondary']['range']}km")
    logger.info(f"有烟雾: {ship_cfg.get('has_smoke', False)}")
    logger.info("Bot v3 - 完全自动后台挂机模式")

    # 等待游戏窗口
    logger.info("搜索战舰世界窗口...")
    windows = None
    for _ in range(60):
        windows = find_game_window()
        if windows:
            break
        time.sleep(1)

    if not windows:
        logger.error("未找到游戏窗口!")
        return

    hwnd, title = windows[0]
    logger.info(f"找到游戏窗口: {title}")
    bot = Bot(hwnd, ship_cfg)

    logger.info("=" * 50)
    logger.info("Bot 就绪! 开始自动循环...")
    logger.info("=" * 50)

    try:
        while True:
            # === 阶段1: 在港口 -> 点击加入战斗 ===
            logger.info("检查当前状态...")
            img = bot.vision.grab(hwnd)
            time.sleep(1)

            # 检查是否在港口
            if bot.vision.in_port(img):
                logger.info("在港口，点击'加入战斗'...")
                bg_click_top_center(hwnd)
                time.sleep(3)
                # 确认是否进入匹配 (按钮文字变化)
                logger.info("等待匹配...")
                time.sleep(5)

            # === 阶段2: 等待战斗开始 ===
            logger.info("等待战斗开始...")
            start = time.time()
            while time.time() - start < 180:  # 最多等3分钟
                time.sleep(2)
                img = bot.vision.grab(hwnd)
                if bot.vision.in_battle(img):
                    logger.info("战斗开始!")
                    time.sleep(2)  # 等UI稳定
                    break
            else:
                logger.warning("等待超时，重试...")
                continue

            # === 阶段3: 战斗循环 ===
            bot.reset()
            logger.info("进入战斗!")

            while True:
                result = bot.combat_tick()
                if result == "ended":
                    logger.info("战斗结束!")
                    break
                time.sleep(0.5)

            # === 阶段4: 结算 -> 点击继续 ===
            logger.info("等待结算画面...")
            time.sleep(8)

            logger.info("点击'继续'...")
            bg_click_center(hwnd)
            time.sleep(3)

            # 再点一次确保
            bg_click_center(hwnd)
            time.sleep(5)

            logger.info("准备下一局...")
            time.sleep(3)

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        bot.gp.stop()
        logger.info("Bot 已停止")


if __name__ == "__main__":
    main()
