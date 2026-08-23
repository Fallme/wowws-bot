"""Guided, operator-confirmed game-input calibration.

Run this only while the game is open in a safe training or Co-op battle.  A
calibration record is written only when every required action is visibly
confirmed by the operator.
"""

from __future__ import annotations

import time

import cv2

from core.calibration import CalibrationStore, InputCalibration
from core.input import configured_input_backend, create_input_controller
from core.vision import Vision
from core.window import find_game_window
from core.ui import ScreenState


def frame_delta(before, after) -> float:
    left = cv2.resize(before, (160, 90), interpolation=cv2.INTER_AREA)
    right = cv2.resize(after, (160, 90), interpolation=cv2.INTER_AREA)
    return float(cv2.absdiff(left, right).mean())


def ask_confirm(action_name: str, instruction: str, action, vision, hwnd, delay=1.8):
    print(f"\n[{action_name}] {instruction}")
    input("确认游戏处于安全场景后按 Enter 发送测试指令；输入 Ctrl+C 可退出。")
    before = vision.grab(hwnd)
    action()
    time.sleep(delay)
    after = vision.grab(hwnd)
    delta = frame_delta(before, after)
    answer = input(f"画面变化量 {delta:.2f}。游戏是否明确执行了该动作？[y/N] ").strip().lower()
    return answer in {"y", "yes"}, {"frame_delta": round(delta, 3)}


def main():
    print("战舰世界输入校准")
    print("请在训练房或人机战斗中运行；程序不会把画面变化自动当成控制成功。")
    windows = find_game_window()
    if not windows:
        print("未找到战舰世界窗口。")
        return 2
    hwnd, title, rect = max(
        windows,
        key=lambda item: max(0, item[2][2] - item[2][0]) * max(0, item[2][3] - item[2][1]),
    )
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    vision = Vision()
    image = vision.grab(hwnd)
    state = vision.classify_screen(image)
    if state != ScreenState.BATTLE:
        print(f"当前画面识别为 {state.value}。必须进入安全的战斗场景后校准。")
        return 3

    gamepad = create_input_controller()
    confirmations = []
    observations = {}
    actions = (
        (
            "throttle_forward",
            "舰船应开始前进或航速挡位发生变化",
            lambda: gamepad.set_movement(0.85, 0.0),
        ),
        (
            "rudder_left",
            "舰船应向左打舵",
            lambda: gamepad.set_movement(0.65, -0.75),
        ),
        (
            "rudder_right",
            "舰船应向右打舵",
            lambda: gamepad.set_movement(0.65, 0.75),
        ),
        ("main_fire", "主炮应触发开火", gamepad.fire),
        ("secondary_lock", "目标锁定状态应发生变化", gamepad.lock),
    )
    try:
        for key, instruction, action in actions:
            confirmed, observation = ask_confirm(
                key,
                instruction,
                action,
                vision,
                hwnd,
            )
            gamepad.stop()
            observations[key] = observation
            observations[key]["operator_confirmed"] = confirmed
            if not confirmed:
                print(f"校准失败：{key} 未得到实机确认。未写入校准凭证。")
                return 4
            confirmations.append(key)
    finally:
        gamepad.stop()

    record = InputCalibration(
        backend=configured_input_backend(),
        game_title=title,
        resolution=[width, height],
        confirmed_actions=confirmations,
        observations=observations,
    )
    status = CalibrationStore().save(record)
    if not status.valid:
        print(f"校准记录无效：{status.reason}")
        return 5
    print("\n校准通过。控制面板现可启动受保护的自动运行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
