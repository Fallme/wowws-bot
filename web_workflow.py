"""Click-only web workflows for game startup and supervised calibration."""

from __future__ import annotations

import threading
import time

import cv2

from core.calibration import CalibrationStore, InputCalibration
from core.input import configured_input_backend, create_input_controller
from core.launcher import launch_game
from core.ui import ScreenState
from core.vision import Vision
from core.window import activate_window, find_game_window
from port_navigator import (
    enter_battle,
    ensure_requested_mode,
    handle_post_battle,
    select_requested_ship,
)


CALIBRATION_ACTIONS = (
    ("throttle_forward", "前进", "舰船应切换前进档并开始移动"),
    ("rudder_left", "左转", "舰船应保持前进并向左打舵"),
    ("rudder_right", "右转", "舰船应保持前进并向右打舵"),
    ("main_fire", "主炮开火", "主炮应触发一次开火"),
    ("secondary_lock", "目标锁定", "目标锁定状态应发生变化"),
)


def _frame_delta(before, after) -> float:
    left = cv2.resize(before, (160, 90), interpolation=cv2.INTER_AREA)
    right = cv2.resize(after, (160, 90), interpolation=cv2.INTER_AREA)
    return float(cv2.absdiff(left, right).mean())


def game_status(window_finder=find_game_window) -> dict:
    windows = window_finder()
    if not windows:
        return {"running": False, "title": "", "resolution": [0, 0]}
    _, title, rect = max(
        windows,
        key=lambda item: max(0, item[2][2] - item[2][0])
        * max(0, item[2][3] - item[2][1]),
    )
    return {
        "running": True,
        "title": title,
        "resolution": [rect[2] - rect[0], rect[3] - rect[1]],
    }


class WebCalibrationWorkflow:
    """Prepare a Co-op battle and collect explicit browser confirmations."""

    def __init__(
        self,
        *,
        store=None,
        vision_factory=Vision,
        controller_factory=create_input_controller,
        launcher=launch_game,
        window_finder=find_game_window,
        activator=activate_window,
        sleep=time.sleep,
    ):
        self.store = store or CalibrationStore()
        self.vision_factory = vision_factory
        self.controller_factory = controller_factory
        self.launcher = launcher
        self.window_finder = window_finder
        self.activator = activator
        self.sleep = sleep
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self._reset()

    def _reset(self):
        self.state = "idle"
        self.message = "等待网页操作"
        self.error = ""
        self.ship = "pommern"
        self.step_index = 0
        self.awaiting_confirmation = False
        self.last_delta = 0.0
        self.confirmed_actions = []
        self.observations = {}
        self.hwnd = None
        self.game_title = ""
        self.resolution = [0, 0]
        self.vision = None
        self.controller = None
        self.worker = None

    @property
    def busy(self):
        return self.state in {"launching", "preparing", "matching", "testing"}

    def status(self) -> dict:
        with self.lock:
            action = None
            if (
                self.state in {"ready", "testing", "confirm"}
                and self.step_index < len(CALIBRATION_ACTIONS)
            ):
                key, label, instruction = CALIBRATION_ACTIONS[self.step_index]
                action = {
                    "key": key,
                    "label": label,
                    "instruction": instruction,
                }
            return {
                "state": self.state,
                "message": self.message,
                "error": self.error,
                "ship": self.ship,
                "step": self.step_index + 1 if action else self.step_index,
                "total_steps": len(CALIBRATION_ACTIONS),
                "action": action,
                "awaiting_confirmation": self.awaiting_confirmation,
                "frame_delta": self.last_delta,
                "confirmed_actions": list(self.confirmed_actions),
                "game": game_status(self.window_finder),
                "calibration": self.store.status().to_dict(),
            }

    def launch(self) -> dict:
        if game_status(self.window_finder)["running"]:
            return {"started": False, "method": "existing", "detail": "游戏已运行"}
        result = self.launcher()
        if not result.started:
            raise RuntimeError(result.detail)
        return {
            "started": result.started,
            "method": result.method,
            "detail": result.detail,
        }

    def prepare(self, ship: str = "pommern") -> dict:
        ship = str(ship or "pommern").strip().lower()
        if ship not in {"pommern", "napoli"}:
            raise ValueError("不支持的舰船")
        with self.lock:
            if self.busy or self.state == "confirm":
                raise RuntimeError("校准流程正在进行")
            self._safe_stop()
            self.cancel_event.set()
            self._reset()
            self.cancel_event = threading.Event()
            self.ship = ship
            self.state = "launching"
            self.message = "正在启动并进入游戏"
            cancel_event = self.cancel_event
            self.worker = threading.Thread(
                target=self._prepare_worker,
                args=(cancel_event,),
                daemon=True,
            )
            self.worker.start()
            return self.status()

    def _wait_for_window(self, cancel_event, timeout=300):
        deadline = time.monotonic() + timeout
        windows = self.window_finder()
        if not windows:
            result = self.launcher()
            if not result.started:
                raise RuntimeError(f"无法启动游戏: {result.detail}")
        while time.monotonic() < deadline:
            if cancel_event.is_set():
                return None
            windows = self.window_finder()
            if windows:
                return max(
                    windows,
                    key=lambda item: max(0, item[2][2] - item[2][0])
                    * max(0, item[2][3] - item[2][1]),
                )
            self.sleep(1)
        raise RuntimeError("等待游戏窗口超时")

    def _prepare_worker(self, cancel_event):
        try:
            window = self._wait_for_window(cancel_event)
            if window is None or cancel_event.is_set():
                return
            hwnd, title, rect = window
            vision = self.vision_factory()
            with self.lock:
                if cancel_event.is_set():
                    return
                self.hwnd = hwnd
                self.game_title = title
                self.resolution = [rect[2] - rect[0], rect[3] - rect[1]]
                self.vision = vision
                self.state = "preparing"
                self.message = "正在等待港口并配置联合作战"

            deadline = time.monotonic() + 300
            port_configured = False
            while time.monotonic() < deadline:
                if cancel_event.is_set():
                    return
                image = vision.grab(hwnd, allow_stale=True)
                screen = vision.classify_screen(image)
                if screen == ScreenState.BATTLE:
                    with self.lock:
                        if cancel_event.is_set():
                            return
                        self.controller = self.controller_factory()
                        self.state = "ready"
                        self.message = "战斗已就绪，可以发送第一个校准动作"
                    return
                if screen == ScreenState.RESULTS:
                    handle_post_battle(hwnd, vision=vision)
                    self.sleep(2)
                    continue
                if screen == ScreenState.PORT:
                    if not port_configured:
                        self.activator(hwnd)
                        if not select_requested_ship(hwnd, self.ship, vision=vision):
                            raise RuntimeError("网页流程未能确认目标舰船")
                        if not ensure_requested_mode(hwnd, "cooperative", vision=vision):
                            raise RuntimeError("网页流程未能确认联合作战模式")
                        port_configured = True
                    if not enter_battle(hwnd, vision=vision, configure_port=False):
                        raise RuntimeError("网页流程未能点击加入战斗")
                    with self.lock:
                        if cancel_event.is_set():
                            return
                        self.state = "matching"
                        self.message = "已加入联合作战，正在等待战斗 HUD"
                    self.sleep(3)
                    continue
                self.sleep(2)
            raise RuntimeError("等待联合作战开始超时")
        except Exception as error:
            if not cancel_event.is_set():
                self._fail(str(error))

    def run_action(self) -> dict:
        with self.lock:
            if self.state != "ready":
                raise RuntimeError("当前不能发送校准动作")
            if self.controller is None or self.vision is None or self.hwnd is None:
                raise RuntimeError("校准控制器尚未就绪")
            key, label, _ = CALIBRATION_ACTIONS[self.step_index]
            self.state = "testing"
            self.message = f"正在测试：{label}"
        try:
            before = self.vision.grab(self.hwnd)
            if self.vision.classify_screen(before) != ScreenState.BATTLE:
                raise RuntimeError("当前已不在战斗画面，校准停止")
            self.activator(self.hwnd)
            self._dispatch_action(key)
            self.sleep(1.8)
            after = self.vision.grab(self.hwnd)
            delta = round(_frame_delta(before, after), 3)
        except Exception as error:
            self._safe_stop()
            self._fail(str(error))
            raise RuntimeError(str(error)) from error
        self._safe_stop()
        with self.lock:
            self.last_delta = delta
            self.awaiting_confirmation = True
            self.state = "confirm"
            self.message = "请观察游戏后点击“动作正确”或“动作失败”"
            return self.status()

    def _dispatch_action(self, key: str):
        actions = {
            "throttle_forward": lambda: self.controller.set_movement(0.85, 0.0),
            "rudder_left": lambda: self.controller.set_movement(0.65, -0.75),
            "rudder_right": lambda: self.controller.set_movement(0.65, 0.75),
            "main_fire": self.controller.fire,
            "secondary_lock": self.controller.lock,
        }
        actions[key]()

    def confirm(self, confirmed: bool) -> dict:
        with self.lock:
            if self.state != "confirm" or not self.awaiting_confirmation:
                raise RuntimeError("当前没有等待确认的校准动作")
            key, label, _ = CALIBRATION_ACTIONS[self.step_index]
            self.observations[key] = {
                "frame_delta": self.last_delta,
                "operator_confirmed": bool(confirmed),
                "source": "web_control_panel",
            }
            self.awaiting_confirmation = False
            if not confirmed:
                self.state = "failed"
                self.error = f"{label} 未通过实机确认"
                self.message = "校准失败，未生成凭证；可重新开始"
                self._safe_stop()
                return self.status()
            self.confirmed_actions.append(key)
            self.step_index += 1
            if self.step_index < len(CALIBRATION_ACTIONS):
                self.state = "ready"
                self.message = "当前动作已确认，可以发送下一个动作"
                return self.status()

            record = InputCalibration(
                backend=configured_input_backend(),
                game_title=self.game_title,
                resolution=list(self.resolution),
                confirmed_actions=list(self.confirmed_actions),
                observations=dict(self.observations),
            )
            status = self.store.save(record)
            if not status.valid:
                self.state = "failed"
                self.error = status.reason
                self.message = "校准凭证写入后校验失败"
            else:
                self.state = "completed"
                self.message = "网页实机校准已通过，可以启动持续任务"
            self._safe_stop()
            return self.status()

    def cancel(self) -> dict:
        with self.lock:
            self.cancel_event.set()
            self._safe_stop()
            self._reset()
            self.cancel_event = threading.Event()
            self.message = "校准已取消，所有控制已释放"
            return self.status()

    def _safe_stop(self):
        controller = getattr(self, "controller", None)
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass

    def _fail(self, message: str):
        with self.lock:
            self._safe_stop()
            self.state = "failed"
            self.error = message
            self.message = "网页校准流程失败，可重新开始"
