"""通用战斗逻辑 - 两船共用"""

import time
import logging
from core.state import GameState, Enemy
from core.capture import ScreenCapture
from core.recognizer import Recognizer
from input_control import InputController

logger = logging.getLogger("combat")


class CombatSystem:
    def __init__(self, capture: ScreenCapture, recognizer: Recognizer,
                 controller: InputController, ship_config: dict):
        self.capture = capture
        self.recognizer = recognizer
        self.controller = controller
        self.ship = ship_config
        self.strategy = ship_config["strategy"]

        # 状态
        self.state = GameState()
        self.current_target = None
        self.last_fire_time = 0
        self.last_torpedo_time = 0
        self.last_smoke_time = 0
        self.locked_target_id = None  # 用于判断目标是否切换

    def update_state(self):
        """更新游戏状态"""
        minimap = self.capture.grab_minimap()
        reload_bar = self.capture.grab_reload_bar()
        health_bar = self.capture.grab_health_bar()
        full_screen = self.capture.grab_full()

        # 战斗结束检测
        if self.recognizer.detect_battle_ended(full_screen):
            self.state.battle_ended = True
            return

        self.state.health = self.recognizer.detect_health(health_bar)
        self.state.main_gun_ready = self.recognizer.detect_reload_status(reload_bar)

        if self.ship.get("has_torpedoes", False):
            torpedo_bar = self.capture.grab_torpedo_reload()
            self.state.torpedo_ready = self.recognizer.detect_reload_status(torpedo_bar)

        self.state.minimap_enemies = self.recognizer.detect_enemies_on_minimap(minimap)
        self.state.torpedo_incoming = self.recognizer.detect_torpedoes_on_minimap(minimap)

    def find_target(self):
        """找最近的敌人"""
        minimap = self.capture.grab_minimap()
        h, w = minimap.shape[:2]
        enemy = self.recognizer.find_nearest_enemy(minimap, my_position=(w // 2, h // 2))

        if enemy:
            self.current_target = enemy
            return enemy
        return None

    def navigate_toward_target(self, target):
        """朝目标航行 (手柄摇杆控制)"""
        minimap = self.capture.grab_minimap()
        h, w = minimap.shape[:2]
        my_x, my_y = w // 2, h // 2

        dx = target.x - my_x
        dy = target.y - my_y

        self.controller.navigate_toward(dx, dy)

    def lock_target(self, target):
        """副炮锁定 (手柄A键)"""
        target_id = f"{target.x}_{target.y}"

        # 目标没变且已锁定，不重复锁
        if self.locked_target_id == target_id and self.state.secondary_locked:
            return

        self.controller.lock_secondary()
        self.state.secondary_locked = True
        self.locked_target_id = target_id
        logger.info("副炮锁定目标")

    def fire_main_gun(self):
        """主炮射击 (手柄RT扳机)"""
        if not self.state.main_gun_ready:
            return

        now = time.time()
        if now - self.last_fire_time < 2:
            return

        self.controller.fire_main_gun()
        self.last_fire_time = now
        logger.info("主炮开火")

    def fire_torpedo(self, target):
        """发射鱼雷 (手柄X键)"""
        if not self.ship.get("has_torpedoes", False):
            return
        if not self.state.torpedo_ready:
            return

        now = time.time()
        if now - self.last_torpedo_time < 3:
            return

        self.controller.fire_torpedo()
        self.last_torpedo_time = now
        logger.info("鱼雷发射")

    def activate_smoke(self):
        """释放烟雾 (手柄B键)"""
        if not self.ship.get("has_smoke", False):
            return

        smoke_threshold = self.strategy.get("smoke_threshold", 0.5)
        if self.state.health >= smoke_threshold:
            return

        now = time.time()
        if now - self.last_smoke_time < 5:
            return

        self.controller.release_smoke()
        self.last_smoke_time = now
        logger.info(f"释放烟雾 (血量: {self.state.health:.0%})")

    def evade_torpedo(self):
        """规避鱼雷 (手柄摇杆转舵)"""
        logger.info("检测到鱼雷，规避中...")
        self.controller.steer_left(amount=0.9)
        time.sleep(1.5)
        self.controller.steer_right(amount=0.9)
        time.sleep(1.0)
        self.controller.steer_straight()

    def combat_tick(self):
        """一次战斗循环"""
        self.update_state()

        if self.state.battle_ended:
            return "ended"

        # 鱼雷规避优先级最高
        if self.state.torpedo_incoming:
            self.evade_torpedo()
            return "evading"

        # 烟雾 (Napoli)
        self.activate_smoke()

        # 找目标
        target = self.find_target()

        if target is None:
            self.controller.throttle_full()
            return "sailing"

        # 航行接近
        self.navigate_toward_target(target)

        # 副炮锁定
        self.lock_target(target)

        # 主炮射击
        self.fire_main_gun()

        # 鱼雷 (Pommern)
        torpedo_range = self.strategy.get("torpedo_fire_distance", 6000)
        if target.distance > 0 and target.distance < torpedo_range:
            self.fire_torpedo(target)

        return "combat"

    def cleanup(self):
        """清理状态"""
        self.controller.stop_all()
        self.current_target = None
        self.locked_target_id = None
        self.state = GameState()
