import time

from core.calibration import (
    AUTOMATIC_PREFLIGHT_KEY,
    CalibrationStore,
    InputCalibration,
    REQUIRED_ACTIONS,
)


def test_complete_current_calibration_is_valid(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json", max_age_days=30)
    record = InputCalibration(
        game_title="World of Warships",
        resolution=[2560, 1440],
        confirmed_actions=list(REQUIRED_ACTIONS),
    )
    status = store.save(record)
    assert status.valid
    assert status.resolution == (2560, 1440)


def test_missing_action_blocks_runtime(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    record = InputCalibration(
        game_title="World of Warships",
        resolution=[2560, 1440],
        confirmed_actions=list(REQUIRED_ACTIONS[:-1]),
    )
    status = store.save(record)
    assert not status.valid
    assert "缺少动作" in status.reason


def test_expired_calibration_blocks_runtime(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json", max_age_days=30)
    created = time.time() - 31 * 86400
    record = InputCalibration(
        created_at=created,
        game_title="World of Warships",
        resolution=[2560, 1440],
        confirmed_actions=list(REQUIRED_ACTIONS),
    )
    store.save(record)
    status = store.status(now=time.time())
    assert not status.valid
    assert "有效期" in status.reason


def test_missing_game_window_identity_blocks_runtime(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    record = InputCalibration(
        game_title="",
        resolution=[2560, 1440],
        confirmed_actions=list(REQUIRED_ACTIONS),
    )
    status = store.save(record)
    assert not status.valid
    assert "游戏窗口" in status.reason


def test_automatic_preflight_is_valid_without_manual_combat_actions(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    record = InputCalibration(
        game_title="World of Warships",
        resolution=[2560, 1600],
        observations={AUTOMATIC_PREFLIGHT_KEY: {"passed": True}},
    )

    status = store.save(record)

    assert status.valid
    assert status.reason == "自动港口自检已通过"
