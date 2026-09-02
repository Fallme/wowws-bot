"""Offline replay of saved screenshots through scene, OCR and movement logic.

This diagnostic intentionally has no window handle and imports no input
controller.  It can therefore show the task that *would* be dispatched for a
frame without sending keyboard or mouse events to World of Warships.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ocr import RapidOcrBackend
from core.results import ResultRewardReader
from core.ui import ScreenState
from core.vision import Vision
from port_navigator import in_battle_type_selector
from strategy.secondary_movement import (
    SecondaryMovementController,
    SecondaryMovementInput,
)


DEFAULT_CAPTURE_ROOT = PROJECT_ROOT / "training_assets" / "user_captures"
DEFAULT_REPORT = (
    PROJECT_ROOT / "runtime" / "ocr_reports" / "screenshot_scenario_report.json"
)

# User-provided full-window frames with known ground truth.  Missing files are
# simply skipped so a clean clone can still run the tool against explicit
# paths or portable test fixtures.
CURATED_REFERENCES = {
    "codex-clipboard-c90a1b76-f657-4d96-a2ca-37ca2d20aec3.png": {
        "scene": "battle",
        "speed": 0.0,
        "health": 1.0,
    },
    "codex-clipboard-419a7b62-8d81-423c-852c-069d24003e63.png": {
        "scene": "battle",
        "speed": 29.2,
        "health": 1.0,
    },
    "codex-clipboard-6613f42b-9cbe-4e1a-8fc5-ec66609c60e0.png": {
        "scene": "battle",
        "speed": 0.0,
        "health": 1.0,
    },
    "codex-clipboard-cb092149-6bdf-4f46-b978-4d4f7f5902e5.png": {
        "scene": "battle",
        "speed": 30.3,
        "health": 1.0,
    },
    "codex-clipboard-e37a2640-219c-4923-bc6f-d8ec2e9af225.png": {
        "scene": "battle",
        "speed": 0.0,
        "health": 1.0,
    },
    "codex-clipboard-ad940697-6313-400d-849e-f4e339282e3a.png": {
        "scene": "battle",
        "speed": 28.5,
        "health": 1.0,
    },
    "codex-clipboard-71a29d1f-d8a2-47e0-820f-4cbcc68e1488.png": {
        "scene": "port",
    },
    "codex-clipboard-5c773886-8a84-4805-8ef2-e15df79f49bf.png": {
        "scene": "battle_mode_selection",
    },
    "codex-clipboard-1368a367-a004-456d-b56d-619661d8cbc5.png": {
        "scene": "results_evidence",
        "credits": 198_363,
        "ship_xp": 1_602,
        "free_xp": 198,
    },
}


def _distance(minimap, first, second):
    return Vision.minimap_pixels_to_km(minimap, math.dist(first, second))


def _round(value, digits=3):
    return None if value is None else round(float(value), digits)


def _battle_replay(image, vision, backend, movement):
    minimap = vision.find_minimap(image)
    health = vision.read_health_fraction(image, backend)
    speed = vision.read_speed_knots(image, backend)
    autopilot_color_hint = bool(vision.is_autopilot_enabled(image))
    autopilot_confirmed = bool(
        vision.read_autopilot_enabled_text(image, backend)
    )
    payload = {
        "health_fraction": _round(health, 4),
        "speed_knots": _round(speed, 1),
        "on_fire": bool(vision.is_on_fire(image)),
        "flooding": bool(vision.is_flooding(image)),
        "autopilot": autopilot_confirmed,
        "autopilot_color_hint": autopilot_color_hint,
        "rudder_indicator": vision.detect_rudder_indicator(image),
        "minimap": {"available": minimap is not None},
    }
    if minimap is None:
        payload["proposed_task"] = {
            "action": "hold_controls_and_retry_vision",
            "reason": "小地图不可用，禁止盲目打舵",
        }
        return payload

    enemies, torpedoes = vision.analyze_minimap(minimap)
    pose = vision.find_player_pose_on_minimap(minimap)
    zones = vision.find_capture_zones(
        minimap,
        player=None if pose is None else pose.position,
    )
    outlines = vision.find_minimap_island_outlines(minimap)
    minimap_info = {
        "size": [int(minimap.shape[1]), int(minimap.shape[0])],
        "player": None,
        "heading": None,
        "enemy_count": len(enemies),
        "enemies": [list(point) for point in enemies],
        "torpedo_pixels_seen": bool(torpedoes),
        "capture_zones": [
            {
                "label": zone.label,
                "state": zone.state,
                "center": list(zone.center),
                "radius": round(zone.radius, 1),
            }
            for zone in zones
        ],
        "island_shapes": len(outlines),
    }
    payload["minimap"] = minimap_info
    if pose is None:
        payload["proposed_task"] = {
            "action": "full_speed_straight_and_retry_pose",
            "throttle": 1.0,
            "rudder": 0.0,
            "reason": "未可靠识别白色舰船箭头，不按错误航向打舵",
        }
        return payload

    minimap_info["player"] = list(pose.position)
    minimap_info["heading"] = [_round(value, 4) for value in pose.heading]
    center = (minimap.shape[1] / 2.0, minimap.shape[0] / 2.0)
    center_bearing = vision.relative_bearing(pose, center)
    center_distance = _distance(minimap, pose.position, center)
    selected_zone = vision.select_navigation_capture_zone(zones, pose.position)
    zone_bearing = None
    zone_distance = None
    inside_zone = False
    if selected_zone is not None:
        zone_bearing = vision.relative_bearing(pose, selected_zone.center)
        zone_distance = _distance(minimap, pose.position, selected_zone.center)
        inside_zone = math.dist(pose.position, selected_zone.center) <= (
            selected_zone.radius * 0.96
        )

    nearest_enemy = None
    enemy_bearing = None
    enemy_distance = None
    if enemies:
        nearest_enemy = min(enemies, key=lambda point: math.dist(pose.position, point))
        enemy_bearing = vision.relative_bearing(pose, nearest_enemy)
        enemy_distance = _distance(minimap, pose.position, nearest_enemy)
    island = vision.find_island_risk(
        minimap,
        pose,
        island_outlines=outlines,
    )

    minimap_info.update(
        {
            "map_center_bearing": _round(center_bearing),
            "map_center_distance_km": _round(center_distance, 1),
            "selected_zone": None
            if selected_zone is None
            else selected_zone.label or "unlabelled",
            "selected_zone_bearing": _round(zone_bearing),
            "selected_zone_distance_km": _round(zone_distance, 1),
            "inside_selected_zone": inside_zone,
            "nearest_enemy": None if nearest_enemy is None else list(nearest_enemy),
            "nearest_enemy_bearing": _round(enemy_bearing),
            "nearest_enemy_distance_km": _round(enemy_distance, 1),
            "island_risk": None
            if island is None
            else {
                "distance": _round(island.distance),
                "avoidance_rudder": _round(island.avoidance_rudder),
            },
        }
    )
    if payload["autopilot"]:
        payload["proposed_task"] = {
            "action": "keep_native_autopilot",
            "throttle": None,
            "rudder": None,
            "reason": "自动航行已开启，互锁Q/E直到其自然结束",
        }
        return payload

    command = movement.plan(
        SecondaryMovementInput(
            elapsed=60.0,
            health=health if health is not None else 1.0,
            visible_target=bool(enemies),
            minimap_distance_km=enemy_distance,
            minimap_target_bearing=enemy_bearing,
            map_center_bearing=center_bearing,
            map_center_distance_km=center_distance,
            capture_point_bearing=zone_bearing,
            capture_point_distance_km=zone_distance,
            inside_capture_point=inside_zone,
            route_phase="transit" if not inside_zone else "arrived",
            route_arrived=inside_zone,
            enemy_count=len(enemies),
            # Offline replay reports raw yellow evidence but deliberately does
            # not promote it to evasive control without a typed nearby threat.
            torpedoes_incoming=False,
            island_distance=None if island is None else island.distance,
            island_avoidance_rudder=None
            if island is None
            else island.avoidance_rudder,
        )
    )
    payload["proposed_task"] = {
        "action": "movement_command",
        "mode": command.mode.value,
        "throttle": _round(command.throttle, 2),
        "rudder": _round(command.rudder, 2),
        "reason": command.reason,
    }
    return payload


def replay_image(path, vision, backend, reward_reader, movement, expected=None):
    started = time.perf_counter()
    # Curated screenshots are independent scenarios (and may even come from
    # different battles).  Do not leak the previous image's fixed-side U-turn
    # commitment into the next report. Runtime battle control deliberately
    # keeps that state; this offline tool reports the decision for this frame.
    movement.reset()
    image = cv2.imread(str(path))
    if image is None:
        return {"path": str(path), "error": "image_read_failed", "passed": False}
    height, width = image.shape[:2]
    full_window = width >= 1400 and height >= 800 and width / max(height, 1) >= 1.35
    state = vision.classify_screen(image) if full_window else ScreenState.UNKNOWN
    selector = bool(full_window and in_battle_type_selector(image))
    rewards = None
    if state in {ScreenState.RESULTS, ScreenState.PORT, ScreenState.UNKNOWN}:
        rewards = reward_reader.read(image)
    if rewards is not None and rewards.recognized:
        extended_scene = "results_evidence"
    elif selector:
        extended_scene = "battle_mode_selection"
    else:
        extended_scene = state.value

    if extended_scene == "battle":
        analysis = _battle_replay(image, vision, backend, movement)
    elif extended_scene == "results_evidence":
        analysis = {
            "rewards": rewards.to_dict(),
            "proposed_task": {
                "action": "record_result_then_continue",
                "reason": "结算三项资源已形成完整证据",
            },
        }
    elif extended_scene == "port":
        analysis = {
            "proposed_task": {
                "action": "verify_ship_and_mode_then_join",
                "reason": "已确认港口页面",
            }
        }
    elif extended_scene == "battle_mode_selection":
        analysis = {
            "proposed_task": {
                "action": "select_configured_battle_mode",
                "reason": "已确认战斗模式选择页",
            }
        }
    elif extended_scene == "loading":
        analysis = {"proposed_task": {"action": "wait_for_battle_hud"}}
    elif extended_scene in {"escape_menu", "exit_confirmation"}:
        analysis = {
            "proposed_task": {
                "action": "resume_battle",
                "reason": "明确识别到战斗内菜单",
            }
        }
    else:
        analysis = {
            "proposed_task": {
                "action": "retry_scene_before_global_recovery",
                "reason": "未知页面先重新截图确认，连续失败才执行全局ESC恢复",
            }
        }

    checks = []
    expected = expected or {}
    if "scene" in expected:
        checks.append(
            {
                "field": "scene",
                "expected": expected["scene"],
                "actual": extended_scene,
                "passed": extended_scene == expected["scene"],
            }
        )
    for field in ("credits", "ship_xp", "free_xp"):
        if field in expected:
            actual = 0 if rewards is None else getattr(rewards, field)
            checks.append(
                {
                    "field": field,
                    "expected": expected[field],
                    "actual": actual,
                    "passed": actual == expected[field],
                }
            )
    for field, source in (("health", "health_fraction"), ("speed", "speed_knots")):
        if field in expected:
            actual = analysis.get(source)
            tolerance = 0.015 if field == "health" else 0.35
            checks.append(
                {
                    "field": field,
                    "expected": expected[field],
                    "actual": actual,
                    "passed": actual is not None
                    and abs(float(actual) - float(expected[field])) <= tolerance,
                }
            )
    return {
        "path": str(path),
        "dimensions": [width, height],
        "full_window": full_window,
        "base_scene": state.value,
        "scene": extended_scene,
        "analysis": analysis,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def default_paths():
    paths = []
    for filename in CURATED_REFERENCES:
        path = DEFAULT_CAPTURE_ROOT / filename
        if path.is_file():
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="离线回放截图：场景识别、OCR与拟下发任务（不会操作游戏）"
    )
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--no-expectations",
        action="store_true",
        help="只生成识别报告，不按内置代表样本判定通过/失败",
    )
    args = parser.parse_args()
    paths = args.images or default_paths()
    if not paths:
        parser.error("没有可回放截图；请传入图片路径")

    backend = RapidOcrBackend(prefer_gpu=True)
    vision = Vision(screen_capture=object())
    reward_reader = ResultRewardReader(backend)
    movement = SecondaryMovementController()
    results = []
    for path in paths:
        expected = None
        if not args.no_expectations:
            expected = CURATED_REFERENCES.get(Path(path).name)
        result = replay_image(
            Path(path), vision, backend, reward_reader, movement, expected=expected
        )
        results.append(result)
        task = result.get("analysis", {}).get("proposed_task", {}).get("action", "-")
        print(
            f"{Path(path).name}: scene={result.get('scene')} "
            f"task={task} passed={result.get('passed')}"
        )

    payload = {
        "created_at": time.time(),
        "safe_offline_replay": True,
        "execution_provider": backend.execution_provider,
        "images": results,
        "summary": {
            "total": len(results),
            "passed": sum(bool(result.get("passed")) for result in results),
            "failed": sum(not bool(result.get("passed")) for result in results),
            "scene_counts": {
                scene: sum(result.get("scene") == scene for result in results)
                for scene in sorted({result.get("scene", "error") for result in results})
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report={args.output}")
    print(f"provider={backend.execution_provider}")
    return 0 if payload["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
