import numpy as np

from core.calibration import CalibrationStore
from core.ui import ScreenState
from web_workflow import CALIBRATION_ACTIONS, WebCalibrationWorkflow


class FakeVision:
    def __init__(self):
        self.frames = 0

    def grab(self, _hwnd):
        self.frames += 1
        return np.full((90, 160, 3), self.frames, dtype=np.uint8)

    def classify_screen(self, _image):
        return ScreenState.BATTLE


class FakeController:
    def __init__(self):
        self.calls = []

    def set_movement(self, throttle, rudder):
        self.calls.append(("movement", throttle, rudder))

    def fire(self):
        self.calls.append(("fire",))

    def lock(self):
        self.calls.append(("lock",))

    def stop(self):
        self.calls.append(("stop",))


def ready_workflow(tmp_path):
    controller = FakeController()
    vision = FakeVision()
    workflow = WebCalibrationWorkflow(
        store=CalibrationStore(tmp_path / "input_calibration.json"),
        vision_factory=lambda: vision,
        controller_factory=lambda: controller,
        window_finder=lambda: [(1, "World of Warships", (0, 0, 2560, 1440))],
        activator=lambda _hwnd: True,
        sleep=lambda _seconds: None,
    )
    workflow.hwnd = 1
    workflow.game_title = "World of Warships"
    workflow.resolution = [2560, 1440]
    workflow.vision = vision
    workflow.controller = controller
    workflow.state = "ready"
    return workflow, controller


def test_web_calibration_saves_credential_only_after_all_confirmations(tmp_path):
    workflow, controller = ready_workflow(tmp_path)

    for _ in CALIBRATION_ACTIONS:
        action_status = workflow.run_action()
        assert action_status["state"] == "confirm"
        status = workflow.confirm(True)

    assert status["state"] == "completed"
    assert status["calibration"]["valid"]
    assert len(status["confirmed_actions"]) == len(CALIBRATION_ACTIONS)
    assert any(call[0] == "movement" for call in controller.calls)
    assert ("fire",) in controller.calls
    assert ("lock",) in controller.calls


def test_web_calibration_rejection_fails_closed(tmp_path):
    workflow, _ = ready_workflow(tmp_path)

    workflow.run_action()
    status = workflow.confirm(False)

    assert status["state"] == "failed"
    assert not status["calibration"]["valid"]
    assert "未生成凭证" in status["message"]
