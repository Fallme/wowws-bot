"""屏幕捕获模块 - 使用mss高速截屏 (支持后台+副屏)"""

import numpy as np
import mss
import cv2


class ScreenCapture:
    def __init__(self, screen_config, monitor_index=2):
        """
        monitor_index: 显示器编号
          0 = 所有屏幕合并
          1 = 主屏
          2 = 副屏
        """
        self.sct = mss.mss()
        self.config = screen_config
        self.monitor_index = monitor_index

        # 使用指定显示器 (mss后台截图，不需要窗口在前台)
        self.monitor = self.sct.monitors[monitor_index]

    def grab_full(self):
        """截取全屏，返回numpy数组 (BGR格式)"""
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def grab_region(self, region_name):
        """截取指定区域，返回numpy数组"""
        region = self.config["regions"][region_name]
        monitor = {
            "top": region["y"],
            "left": region["x"],
            "width": region["w"],
            "height": region["h"],
        }
        img = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def grab_minimap(self):
        """截取小地图区域"""
        return self.grab_region("minimap")

    def grab_reload_bar(self):
        """截取主炮装填条"""
        return self.grab_region("main_gun_reload")

    def grab_health_bar(self):
        """截取血条"""
        return self.grab_region("health_bar")

    def grab_torpedo_reload(self):
        """截取鱼雷装填条"""
        return self.grab_region("torpedo_reload")

    def grab_distance_text(self):
        """截取距离文字区域"""
        return self.grab_region("distance_text")

    def close(self):
        self.sct.close()
