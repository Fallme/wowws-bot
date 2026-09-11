"""Per-round trajectory journal and periodically refreshed map, without game input."""
import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np


class TrajectoryRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / "trajectory.jsonl").open("a", encoding="utf-8", buffering=1)
        self.points = []
        self.islands = []
        self.zones = []
        self.last_render = 0.0
        self.closed = False

    def __call__(self, event):
        if self.closed:
            return
        try:
            if event.topic == "battle.navigation_handoff":
                self.stream.write(json.dumps(dict(time=event.emitted_at, event=event.topic, **event.payload), ensure_ascii=False) + "\n")
                return
            if event.topic != "battle.tick":
                return
            payload = event.payload
            position = payload.get("minimap_player")
            valid = bool(position is not None and len(position) == 2
                         and all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in position)
                         and not payload.get("player_pose_cached", False))
            record = dict(time=event.emitted_at, tick=payload.get("tick"),
                          position=position if valid else None,
                          mode="native" if payload.get("autopilot_enabled") else payload.get("movement_mode", "qe"),
                          reason=payload.get("movement_reason"), speed_knots=payload.get("speed_knots"),
                          rudder=payload.get("rudder"), throttle=payload.get("throttle"),
                          target=payload.get("navigation_target"), heading=payload.get("minimap_heading"),
                          selected_enemy=payload.get("selected_enemy"),
                          steering_target=payload.get("steering_target"),
                          steering_target_kind=payload.get("steering_target_kind"),
                          enemy_bearing=payload.get("minimap_target_bearing"))
            self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.points.append(record)
            self.islands = payload.get("minimap_islands") or self.islands
            self.zones = payload.get("capture_zones") or self.zones
            if event.emitted_at - self.last_render >= 15:
                self.render()
                self.last_render = event.emitted_at
        except Exception:
            logging.getLogger(__name__).exception("轨迹记录失败；继续驾驶")

    @staticmethod
    def connected(previous, current):
        return bool(previous and previous["position"] is not None and current["position"] is not None
                    and 0 < current["time"] - previous["time"] <= 5
                    and math.dist(previous["position"], current["position"]) <= 0.04)

    def render(self):
        canvas = np.full((850, 800, 3), (35, 30, 20), dtype=np.uint8)
        def pixel(point):
            return tuple(int(30 + float(v) * 740) for v in point)
        for i in range(11):
            n = 30 + i * 74
            cv2.line(canvas, (n, 30), (n, 770), (58, 56, 43), 1)
            cv2.line(canvas, (30, n), (770, n), (58, 56, 43), 1)
        for island in self.islands:
            polygon = island.get("points", island.get("polygon", [])) if isinstance(island, dict) else island
            if len(polygon) >= 3:
                cv2.fillPoly(canvas, [np.array([pixel(p) for p in polygon], dtype=np.int32)], (95, 105, 115))
        for zone in self.zones:
            if zone.get("position") is None:
                continue
            center = pixel(zone["position"])
            cv2.circle(canvas, center, max(2, int(float(zone.get("radius", 0.08)) * 740)), (120, 150, 125), 1)
            cv2.putText(canvas, zone.get("label") or "?", center, cv2.FONT_HERSHEY_SIMPLEX, .5, (210, 220, 210), 1)
        previous = None
        for record in self.points:
            pos = record["position"]
            if pos is not None:
                color = (70, 210, 100) if record["mode"] == "native" else ((90, 100, 250) if "recovery" in str(record["mode"]) else (240, 190, 60))
                if self.connected(previous, record):
                    cv2.line(canvas, pixel(previous["position"]), pixel(pos), color, 2, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, pixel(pos), 3, color, -1)
                if previous and previous["mode"] != record["mode"]:
                    cv2.circle(canvas, pixel(pos), 5, (255, 255, 255), 1)
            previous = record
        cv2.putText(canvas, "Green: native   Blue: Q/E   Red: recovery   White: mode change", (25, 805), cv2.FONT_HERSHEY_SIMPLEX, .52, (225, 225, 225), 1)
        cv2.putText(canvas, f"Samples: {len(self.points)} | gaps/jumps are not connected", (25, 833), cv2.FONT_HERSHEY_SIMPLEX, .5, (180, 180, 180), 1)
        temporary = self.directory / "trajectory.tmp.png"
        success, data = cv2.imencode(".png", canvas)
        if not success:
            raise OSError("Unable to encode trajectory map")
        data.tofile(str(temporary))
        temporary.replace(self.directory / "trajectory.png")

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.render()
        except Exception:
            logging.getLogger(__name__).exception("轨迹图封存失败")
        finally:
            self.stream.close()
