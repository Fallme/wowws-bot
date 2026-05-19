"""输入控制模块 - 虚拟手柄 + 真实键鼠双模式"""

import time
import random
import vgamepad as vg
import pydirectinput


class InputController:
    def __init__(self, use_gamepad=True):
        """
        use_gamepad=True  -> 虚拟手柄模式 (不影响真实鼠标键盘)
        use_gamepad=False -> 真实键鼠模式 (旧方式)
        """
        self.use_gamepad = use_gamepad

        if use_gamepad:
            # 创建虚拟 Xbox 360 手柄
            self.gamepad = vg.VX360Gamepad()
            print("[输入] 虚拟手柄模式 - 不影响真实鼠标键盘")
        else:
            pydirectinput.FAILSAFE = True
            pydirectinput.PAUSE = 0.01
            print("[输入] 真实键鼠模式 - 会控制鼠标键盘!")

        self.screen_center_x = 960
        self.screen_center_y = 540

        # 手柄摇杆状态
        self._left_x = 0.0    # 左摇杆X: 左右舵
        self._left_y = 0.0    # 左摇杆Y: 前进后退
        self._right_x = 0.0   # 右摇杆X: 瞄准左右
        self._right_y = 0.0   # 右摇杆Y: 瞄准上下
        self._triggers = 0.0  # 扳机: 开火

    def _random_delay(self, base=0.05, variance=0.03):
        time.sleep(base + random.uniform(-variance, variance))

    def _update_gamepad(self):
        """推送手柄状态到虚拟设备"""
        if self.use_gamepad:
            self.gamepad.left_joystick(x_value=int(self._left_x * 32767),
                                        y_value=int(self._left_y * 32767))
            self.gamepad.right_joystick(x_value=int(self._right_x * 32767),
                                         y_value=int(self._right_y * 32767))
            self.gamepad.update()

    # ==================== 手柄模式 (推荐) ====================

    def steer_left(self, amount=0.7):
        """左转舵"""
        self._left_x = -amount
        self._update_gamepad()

    def steer_right(self, amount=0.7):
        """右转舵"""
        self._left_x = amount
        self._update_gamepad()

    def steer_straight(self):
        """回正舵"""
        self._left_x = 0.0
        self._update_gamepad()

    def throttle_full(self):
        """全速前进"""
        self._left_y = 1.0
        self._update_gamepad()

    def throttle_stop(self):
        """停船"""
        self._left_y = 0.0
        self._update_gamepad()

    def aim_relative(self, dx, dy):
        """相对瞄准 (右摇杆)"""
        # 限制在 -1 ~ 1 范围
        self._right_x = max(-1.0, min(1.0, dx))
        self._right_y = max(-1.0, min(1.0, dy))
        self._update_gamepad()

    def aim_center(self):
        """瞄准归中"""
        self._right_x = 0.0
        self._right_y = 0.0
        self._update_gamepad()

    def fire_main_gun(self):
        """主炮开火 (RT扳机)"""
        if self.use_gamepad:
            self.gamepad.right_trigger(value=255)
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.right_trigger(value=0)
            self.gamepad.update()
        else:
            pydirectinput.click()
        self._random_delay(0.05, 0.02)

    def lock_secondary(self):
        """副炮锁定 (A键/右键)"""
        if self.use_gamepad:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            time.sleep(0.1)
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            self.gamepad.update()
        else:
            pydirectinput.rightClick()
        self._random_delay(0.05, 0.02)

    def fire_torpedo(self):
        """发射鱼雷 (X键)"""
        if self.use_gamepad:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            time.sleep(0.1)
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            self.gamepad.update()
        else:
            pydirectinput.press('t')
        self._random_delay(0.03, 0.01)

    def release_smoke(self):
        """释放烟雾 (B键)"""
        if self.use_gamepad:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            time.sleep(0.1)
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            self.gamepad.update()
        else:
            pydirectinput.press('x')
        self._random_delay(0.03, 0.01)

    def click_at(self, x, y):
        """点击屏幕指定位置 (用于菜单操作)"""
        if self.use_gamepad:
            # 手柄模式下菜单操作仍需用鼠标
            pydirectinput.click(x, y)
        else:
            pydirectinput.click(x, y)
        self._random_delay(0.05, 0.02)

    def type_text(self, text):
        """输入文字 (菜单搜索等)"""
        if self.use_gamepad:
            pydirectinput.typewrite(text, interval=0.05)
        else:
            pydirectinput.typewrite(text, interval=0.05)

    # ==================== 航行辅助 ====================

    def navigate_toward(self, target_dx, target_dy):
        """根据相对偏移调整航向"""
        # target_dx: 正=目标在右边, 负=目标在左边
        # target_dy: 正=目标在下方, 负=目标在上方

        # 左右转舵 (左摇杆X)
        if abs(target_dx) > 20:  # 偏移阈值
            if target_dx > 0:
                self.steer_right(amount=min(abs(target_dx) / 200, 1.0))
            else:
                self.steer_left(amount=min(abs(target_dx) / 200, 1.0))
        else:
            self.steer_straight()

        # 全速前进
        self.throttle_full()

    def stop_all(self):
        """停止所有操作"""
        self._left_x = 0.0
        self._left_y = 0.0
        self._right_x = 0.0
        self._right_y = 0.0
        self._update_gamepad()
