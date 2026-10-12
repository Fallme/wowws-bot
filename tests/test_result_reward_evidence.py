import json
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from core.frame_guard import CaptureFault
from core.results import BattleRewards
from core.ui import ScreenState
from main import collect_battle_rewards


def result_frame(marker):
    return np.full((40, 64, 3), marker, dtype=np.uint8)


def evidence_bot(tmp_path, frames, *, state=ScreenState.RESULTS):
    stale_battle = result_frame(7)
    vision = SimpleNamespace(
        grab=Mock(side_effect=frames),
        classify_screen=Mock(side_effect=lambda image: (
            ScreenState.BATTLE if image is stale_battle else state
        )),
        _has_battle_hud=Mock(return_value=False),
    )
    return SimpleNamespace(
        hwnd=123,
        vision=vision,
        last_analysis=SimpleNamespace(image=stale_battle),
        distance_ocr_service=None,
        blockage_diagnostic_root=tmp_path,
    )


def evidence_reader(*, complete=True):
    rewards = BattleRewards(
        credits=102_692,
        ship_xp=1_143 if complete else 0,
        free_xp=136 if complete else 0,
        recognized=complete,
        raw_text={"credits": "102 692", "ship_xp": "1 143" if complete else "?"},
        outcome="victory",
    )
    return SimpleNamespace(
        backend=SimpleNamespace(execution_provider="test"),
        read=Mock(return_value=rewards),
    )


def retained_rows(folder):
    return [json.loads(line) for line in
            (folder / "observations.jsonl").read_text(encoding="utf-8").splitlines()]


def saved_marker(folder, row):
    image = cv2.imread(str(folder / row["frame"]))
    assert image is not None
    return int(image[0, 0, 0])


def test_result_ocr_and_success_evidence_use_fresh_frames_after_cached_battle(tmp_path):
    frames = [result_frame(marker) for marker in (20, 40, 60)]
    bot = evidence_bot(tmp_path, frames)
    reader = evidence_reader()

    with patch("main.time.sleep"):
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert confirmed and rewards.recognized
    assert state == ScreenState.RESULTS
    assert bot.vision.grab.call_args_list == [call(123, allow_stale=True)] * 3
    assert all(args.args[0] is frame for args, frame in zip(reader.read.call_args_list, frames))
    assert reader.read.call_count == 3
    assert "result_rewards" not in bot._blockage_traces
    folders = list(tmp_path.glob("result_rewards_*"))
    assert len(folders) == 1
    rows = retained_rows(folders[0])
    observation = next(row for row in rows if row["phase"] == "observation")
    assert observation["details"] == {
        "source": "fresh_capture", "reward_surface": True, "battle_hud": False,
    }
    assert saved_marker(folders[0], observation) == 20
    success = rows[-1]
    assert success["phase"] == "result" and success["action"] == "reward_consensus"
    assert success["details"]["values"] == rewards.resource_values()
    assert success["details"]["outcome"] == "recognized"
    assert success["details"]["battle_outcome"] == "victory"
    assert saved_marker(folders[0], success) == 60


def test_retries_keep_same_trace_and_prior_failure_frames_when_consensus_succeeds(tmp_path):
    frames = [result_frame(marker) for marker in (20, 40, 50, 60, 70, 80, 90)]
    bot = evidence_bot(tmp_path, frames)
    reader = evidence_reader(complete=False)

    with patch("main.time.sleep"):
        first = collect_battle_rewards(bot, reader, attempts=2)
        trace = bot._blockage_traces["result_rewards"]
        folder = trace.folder
        first_rows = retained_rows(folder)
        second = collect_battle_rewards(bot, reader, attempts=2)
        assert bot._blockage_traces["result_rewards"] is trace
        reader.read.return_value = evidence_reader().read.return_value
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert first[0] and second[0]
    assert not first[1].recognized and not second[1].recognized
    assert confirmed and rewards.recognized and state == ScreenState.RESULTS
    assert "result_rewards" not in bot._blockage_traces
    assert list(tmp_path.glob("result_rewards_*")) == [folder]
    rows = retained_rows(folder)
    assert rows[:len(first_rows)] == first_rows
    terminal = [row for row in rows if row["phase"] == "result"]
    assert [row["action"] for row in terminal] == ["reward_retry", "reward_retry", "reward_consensus"]
    assert [saved_marker(folder, row) for row in terminal] == [40, 60, 90]
    assert terminal[0]["details"]["page_confirmed"] is True
    assert terminal[0]["details"]["field_votes"] == {
        "credits": {"102692": 1}, "ship_xp": {}, "free_xp": {},
    }
    assert terminal[0]["details"]["raw_text"] == {"credits": "102 692", "ship_xp": "?"}


def test_paused_retry_preserves_existing_evidence_without_more_capture_or_ocr(tmp_path):
    bot = evidence_bot(tmp_path, [result_frame(20), result_frame(40)])
    reader = evidence_reader(complete=False)
    with patch("main.time.sleep"):
        collect_battle_rewards(bot, reader, attempts=2)
    trace = bot._blockage_traces["result_rewards"]
    before = retained_rows(trace.folder)
    captured = bot.vision.grab.call_count
    read_count = reader.read.call_count

    with patch("main.operation_paused", return_value=True):
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert not confirmed and not rewards.recognized and state == ScreenState.UNKNOWN
    assert bot.vision.grab.call_count == captured
    assert reader.read.call_count == read_count
    assert bot._blockage_traces["result_rewards"] is trace
    assert retained_rows(trace.folder) == before
    assert len(list(trace.folder.glob("frame_*.png"))) == len(before)
    assert all((trace.folder / row["frame"]).is_file() for row in before)


def test_capture_faults_never_save_cached_battle_as_reward_evidence(tmp_path):
    bot = evidence_bot(tmp_path, [CaptureFault("capture unavailable")] * 3)
    reader = evidence_reader()

    with patch("main.time.sleep"):
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert not confirmed and not rewards.recognized and state == ScreenState.UNKNOWN
    assert bot.vision.grab.call_count == 3
    bot.vision.classify_screen.assert_not_called()
    reader.read.assert_not_called()
    assert bot._blockage_traces["result_rewards"].folder is None
    assert not list(tmp_path.iterdir())


def test_fresh_live_battle_closes_reward_trace_without_reward_ocr(tmp_path):
    live = result_frame(120)
    bot = evidence_bot(tmp_path, [live], state=ScreenState.RESULTS)
    bot.vision._has_battle_hud.return_value = True
    reader = evidence_reader()

    confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert not confirmed and not rewards.recognized and state == ScreenState.BATTLE
    assert bot.vision.grab.call_count == 1
    reader.read.assert_not_called()
    assert "result_rewards" not in bot._blockage_traces
    folder, = tmp_path.glob("result_rewards_*")
    row, = retained_rows(folder)
    assert row["phase"] == "result" and row["action"] == "live_battle"
    assert row["details"] == {
        "source": "fresh_capture", "battle_hud": True, "outcome": "battle_in_progress",
    }
    assert saved_marker(folder, row) == 120
