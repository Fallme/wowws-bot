from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import pytest

from bot import BattleBot
from core.ocr import OcrToken, RapidOcrBackend
from core.ui import ScreenState
from main import classify_runtime_screen, loading_start_confirmed
from tools import run_with_ocr


@pytest.mark.parametrize("device,expected", [("cpu", False), ("nvidia", True)])
def test_selected_device_applies_to_default_backends(monkeypatch, device, expected):
    monkeypatch.setenv("WOWS_OCR_DEVICE", device)
    assert RapidOcrBackend().prefer_gpu is expected
    assert RapidOcrBackend(prefer_gpu=True).prefer_gpu is True


def test_secondary_lock_never_dispatches_input():
    pad = Mock()
    bot = SimpleNamespace(gamepad=pad, last_lock=0, tick=100)
    BattleBot._engage_visible_enemy(bot, None, 999)
    pad.lock.assert_not_called()


@pytest.mark.parametrize("words,expected", [("开始战斗", ScreenState.LOADING), ("R T Y", ScreenState.BATTLE), ("", ScreenState.LOADING)])
def test_loading_colour_conflict_requires_button_text(words, expected):
    image = cv2.imread("tests/fixtures/battle_long_run.png")
    vision = SimpleNamespace(
        classify_screen=lambda image: ScreenState.LOADING,
        _has_battle_hud=lambda image: True,
        _has_loading_start_action=lambda image: True,
    )
    backend = SimpleNamespace(recognize=lambda image: [OcrToken(words, .99, ())])
    bot = SimpleNamespace(vision=vision, distance_reader=SimpleNamespace(backend=backend))
    assert classify_runtime_screen(bot, image) == expected
    assert loading_start_confirmed(bot, image) == (expected == ScreenState.LOADING)


def test_gpu_ready_does_not_install(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(run_with_ocr, "run_component", run)
    assert run_with_ocr.prepare_gpu()
    assert run.call_count == 1


def test_component_stop_before_launch_does_not_spawn(monkeypatch):
    monkeypatch.setenv("WOWS_STOP_FILE", "stop.request")
    monkeypatch.setattr(run_with_ocr.Path, "exists", lambda self: True)
    spawn = Mock()
    monkeypatch.setattr(run_with_ocr.subprocess, "Popen", spawn)
    with pytest.raises(InterruptedError):
        run_with_ocr.run_component(["unused"], timeout=10)
    spawn.assert_not_called()


def test_component_stop_during_install_terminates_child(monkeypatch):
    monkeypatch.setenv("WOWS_STOP_FILE", "stop.request")
    exists = iter([False, True])
    monkeypatch.setattr(run_with_ocr.Path, "exists", lambda self: next(exists))
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(run_with_ocr.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(InterruptedError):
        run_with_ocr.run_component(["unused"], timeout=10)
    process.terminate.assert_called_once()


def test_gpu_missing_installs_then_verifies_in_fresh_process(monkeypatch, tmp_path):
    monkeypatch.setattr(run_with_ocr, "ROOT", tmp_path)
    monkeypatch.setattr(run_with_ocr.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(run_with_ocr, "gpu_runtime_available", lambda: False)
    run = Mock(side_effect=[SimpleNamespace(returncode=x) for x in (2, 0, 0)])
    monkeypatch.setattr(run_with_ocr, "run_component", run)
    assert run_with_ocr.prepare_gpu()
    assert run.call_count == 3
    assert "pip" in run.call_args_list[1].args[0]
    assert run.call_args_list[1].kwargs["env"]["PIP_CACHE_DIR"].startswith(str(tmp_path))


def test_transient_probe_failure_retries_without_installing(monkeypatch, tmp_path):
    monkeypatch.setattr(run_with_ocr, "ROOT", tmp_path)
    monkeypatch.setattr(run_with_ocr.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(run_with_ocr, "gpu_runtime_available", lambda: True)
    monkeypatch.setattr(run_with_ocr.time, "sleep", lambda _s: None)
    run = Mock(side_effect=[SimpleNamespace(returncode=x) for x in (2, 0)])
    monkeypatch.setattr(run_with_ocr, "run_component", run)
    assert run_with_ocr.prepare_gpu()
    assert run.call_count == 2
    assert all("pip" not in call.args[0] for call in run.call_args_list)


def test_transient_probe_failure_twice_falls_back_to_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(run_with_ocr, "ROOT", tmp_path)
    monkeypatch.setattr(run_with_ocr.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(run_with_ocr, "gpu_runtime_available", lambda: True)
    monkeypatch.setattr(run_with_ocr.time, "sleep", lambda _s: None)
    run = Mock(side_effect=[SimpleNamespace(returncode=x) for x in (2, 2)])
    monkeypatch.setattr(run_with_ocr, "run_component", run)
    assert not run_with_ocr.prepare_gpu()
    assert run.call_count == 2
    assert all("pip" not in call.args[0] for call in run.call_args_list)


def test_failed_gpu_install_falls_back_without_launching_probe_again(monkeypatch, tmp_path):
    monkeypatch.setattr(run_with_ocr, "ROOT", tmp_path)
    monkeypatch.setattr(run_with_ocr.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(run_with_ocr, "gpu_runtime_available", lambda: False)
    run = Mock(side_effect=[SimpleNamespace(returncode=x) for x in (2, 1)])
    monkeypatch.setattr(run_with_ocr, "run_component", run)
    assert not run_with_ocr.prepare_gpu()
    assert run.call_count == 2
