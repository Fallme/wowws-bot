from tools.simulate_screenshot_scenarios import replay_image


class RecordingMovement:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def test_each_offline_screenshot_starts_with_fresh_movement_state(tmp_path):
    movement = RecordingMovement()

    result = replay_image(
        tmp_path / "missing.png",
        vision=object(),
        backend=object(),
        reward_reader=object(),
        movement=movement,
    )

    assert result["error"] == "image_read_failed"
    assert movement.reset_calls == 1
