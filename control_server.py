"""Local HTTP control panel for the World of Warships automation runner."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import yaml

from core.calibration import CalibrationStore
from web_workflow import WebCalibrationWorkflow, game_status

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "control_panel.db"
STATE_PATH = DATA_DIR / "runtime_state.json"
STOP_PATH = DATA_DIR / "stop.request"
PAUSE_PATH = DATA_DIR / "pause.request"
RESUME_PATH = DATA_DIR / "resume.request"
LOG_PATH = DATA_DIR / "runtime.log"
MODES = {
    "cooperative": "联合作战",
    "asymmetric": "非对称作战",
}
SUPPORTED_SHIPS = ("pommern", "napoli")
RESOURCES = (
    "credits",
    "ship_xp",
    "commander_xp",
    "coal",
    "steel",
    "doubloons",
    "free_xp",
    "elite_xp",
)


class ControlStore:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._initialize()

    def _initialize(self):
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    ship TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    limit_type TEXT NOT NULL,
                    limit_value REAL NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL
                );
                CREATE TABLE IF NOT EXISTS resource_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    round_no INTEGER NOT NULL DEFAULT 0,
                    credits INTEGER NOT NULL DEFAULT 0,
                    ship_xp INTEGER NOT NULL DEFAULT 0,
                    commander_xp INTEGER NOT NULL DEFAULT 0,
                    coal INTEGER NOT NULL DEFAULT 0,
                    steel INTEGER NOT NULL DEFAULT 0,
                    doubloons INTEGER NOT NULL DEFAULT 0,
                    free_xp INTEGER NOT NULL DEFAULT 0,
                    elite_xp INTEGER NOT NULL DEFAULT 0,
                    note TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'manual',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );
                """
            )
            columns = {
                row[1]
                for row in self.connection.execute(
                    "PRAGMA table_info(resource_entries)"
                )
            }
            for column in ("ship_xp", "commander_xp"):
                if column not in columns:
                    self.connection.execute(
                        f"ALTER TABLE resource_entries ADD COLUMN {column} "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
            if "source" not in columns:
                self.connection.execute(
                    "ALTER TABLE resource_entries ADD COLUMN source "
                    "TEXT NOT NULL DEFAULT 'manual'"
                )
            run_columns = {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(runs)")
            }
            if "completed_rounds" not in run_columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN completed_rounds "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            if "duration_seconds" not in run_columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN duration_seconds "
                    "REAL NOT NULL DEFAULT 0"
                )
            # Preserve useful history created before progress columns existed.
            self.connection.execute(
                """UPDATE runs SET completed_rounds = COALESCE(
                       (SELECT MAX(round_no) FROM resource_entries e
                        WHERE e.run_id = runs.id AND e.round_no > 0), 0)
                   WHERE completed_rounds = 0"""
            )
            self.connection.execute(
                """UPDATE runs SET duration_seconds = MAX(0, ended_at - started_at)
                   WHERE duration_seconds = 0 AND ended_at IS NOT NULL"""
            )

    def close(self):
        with self.lock:
            self.connection.close()

    def create_run(self, run_id, ship, mode, limit_type, limit_value):
        with self.lock, self.connection:
            self.connection.execute(
                """INSERT INTO runs
                   (id, ship, mode, limit_type, limit_value, status,
                    started_at, ended_at, completed_rounds, duration_seconds)
                   VALUES (?, ?, ?, ?, ?, 'running', ?, NULL, 0, 0)""",
                (run_id, ship, mode, limit_type, limit_value, time.time()),
            )

    def update_run_progress(self, run_id, completed_rounds, duration_seconds):
        if not run_id:
            return
        with self.lock, self.connection:
            self.connection.execute(
                """UPDATE runs SET completed_rounds = ?, duration_seconds = ?
                   WHERE id = ?""",
                (
                    max(0, int(completed_rounds or 0)),
                    max(0.0, float(duration_seconds or 0.0)),
                    run_id,
                ),
            )

    def finish_run(self, run_id, status):
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
                (status, time.time(), run_id),
            )

    def add_resources(self, run_id, round_no, values, note, *, source="manual"):
        amounts = [int(values.get(name, 0) or 0) for name in RESOURCES]
        with self.lock, self.connection:
            cursor = self.connection.execute(
                """INSERT INTO resource_entries
                   (run_id, round_no, credits, ship_xp, commander_xp,
                    coal, steel, doubloons, free_xp, elite_xp,
                    note, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    int(round_no or 0),
                    *amounts,
                    note[:200],
                    str(source)[:50],
                    time.time(),
                ),
            )
            return cursor.lastrowid

    def upsert_auto_rewards(self, run_id, round_no, values):
        """Record one OCR result per run/round without polling duplicates."""
        amounts = [int(values.get(name, 0) or 0) for name in RESOURCES]
        round_no = int(round_no or 0)
        if not run_id or round_no <= 0 or not any(amounts):
            return None
        with self.lock, self.connection:
            existing = self.connection.execute(
                """SELECT id FROM resource_entries
                   WHERE run_id = ? AND round_no = ? AND source = 'auto_result_ocr'
                   LIMIT 1""",
                (run_id, round_no),
            ).fetchone()
            if existing:
                self.connection.execute(
                    """UPDATE resource_entries SET
                       credits=?, ship_xp=?, commander_xp=?, coal=?, steel=?,
                       doubloons=?, free_xp=?, elite_xp=?, created_at=?
                       WHERE id=?""",
                    (*amounts, time.time(), existing["id"]),
                )
                return existing["id"]
            cursor = self.connection.execute(
                """INSERT INTO resource_entries
                   (run_id, round_no, credits, ship_xp, commander_xp,
                    coal, steel, doubloons, free_xp, elite_xp,
                    note, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    round_no,
                    *amounts,
                    "结算页 OCR 自动统计",
                    "auto_result_ocr",
                    time.time(),
                ),
            )
            return cursor.lastrowid

    def dashboard(self):
        with self.lock:
            totals = dict(
                self.connection.execute(
                    """SELECT COUNT(*) AS entries,
                    COALESCE(SUM(credits), 0) AS credits,
                    COALESCE(SUM(ship_xp), 0) AS ship_xp,
                    COALESCE(SUM(commander_xp), 0) AS commander_xp,
                    COALESCE(SUM(coal), 0) AS coal,
                    COALESCE(SUM(steel), 0) AS steel,
                    COALESCE(SUM(doubloons), 0) AS doubloons,
                    COALESCE(SUM(free_xp), 0) AS free_xp,
                    COALESCE(SUM(elite_xp), 0) AS elite_xp
                    FROM resource_entries"""
                ).fetchone()
            )
            run_totals = dict(
                self.connection.execute(
                    """SELECT COUNT(*) AS tasks,
                    COALESCE(SUM(completed_rounds), 0) AS completed_rounds,
                    COALESCE(SUM(duration_seconds), 0) AS duration_seconds
                    FROM runs"""
                ).fetchone()
            )
            totals.update(run_totals)
            runs = [
                dict(row)
                for row in self.connection.execute(
                    """SELECT r.*,
                    COALESCE(SUM(e.credits), 0) AS credits,
                    COALESCE(SUM(e.ship_xp), 0) AS ship_xp,
                    COALESCE(SUM(e.commander_xp), 0) AS commander_xp,
                    COALESCE(SUM(e.coal), 0) AS coal,
                    COALESCE(SUM(e.steel), 0) AS steel,
                    COALESCE(SUM(e.doubloons), 0) AS doubloons,
                    COALESCE(SUM(e.free_xp), 0) AS free_xp,
                    COALESCE(SUM(e.elite_xp), 0) AS elite_xp
                    FROM runs r LEFT JOIN resource_entries e ON e.run_id = r.id
                    GROUP BY r.id ORDER BY r.started_at DESC LIMIT 20"""
                )
            ]
        return {"totals": totals, "runs": runs}


class RunnerManager:
    def __init__(self, store: ControlStore):
        self.store = store
        self.process = None
        self.run_id = None
        self.log_stream = None
        self.lock = threading.Lock()

    def _reconcile(self):
        if self.process is None or self.process.poll() is None:
            return
        runtime_state = self._read_runtime_state()
        final_state = str(runtime_state.get("state") or "")
        status = (
            final_state
            if final_state in {"completed", "stopped", "failed"}
            else "completed"
            if self.process.returncode == 0
            else "failed"
        )
        if self.run_id:
            self.store.update_run_progress(
                self.run_id,
                runtime_state.get("completed_rounds", 0),
                runtime_state.get("elapsed_seconds", 0),
            )
            self.store.finish_run(self.run_id, status)
        if self.log_stream:
            self.log_stream.close()
        self.process = None
        self.log_stream = None

    @staticmethod
    def _read_runtime_state():
        if not STATE_PATH.exists():
            return {}
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _watch_stop_request(self, process):
        """Escalate a cooperative stop if the runner cannot exit promptly."""
        try:
            process.wait(timeout=12)
            return
        except subprocess.TimeoutExpired:
            pass
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.terminate()

    def start(self, payload):
        with self.lock:
            self._reconcile()
            if self.process is not None:
                raise RuntimeError("已有任务正在运行")
            ship = str(payload.get("ship", "pommern"))
            mode = str(payload.get("mode", "asymmetric"))
            limit_type = str(payload.get("limit_type", "continuous"))
            limit_value = float(payload.get("limit_value", 0))
            ships = load_ships()
            if ship not in ships:
                raise ValueError("未知舰船配置")
            if mode not in MODES:
                raise ValueError("未知战斗模式")
            if limit_type not in {"rounds", "duration", "continuous"}:
                raise ValueError("限制类型必须是轮次、时间或持续运行")
            if limit_type == "continuous":
                limit_value = 0
            elif not 1 <= limit_value <= (100 if limit_type == "rounds" else 1440):
                raise ValueError("运行限制超出允许范围")

            run_id = uuid.uuid4().hex[:12]
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for path in (STATE_PATH, STOP_PATH, PAUSE_PATH, RESUME_PATH):
                if path.exists():
                    path.unlink()
            env = os.environ.copy()
            env.update(
                {
                    "WOWS_SHIP": ship,
                    "WOWS_MODE": mode,
                    "WOWS_RUN_ID": run_id,
                    "WOWS_STATE_FILE": str(STATE_PATH),
                    "WOWS_STOP_FILE": str(STOP_PATH),
                    "WOWS_PAUSE_FILE": str(PAUSE_PATH),
                    "WOWS_RESUME_FILE": str(RESUME_PATH),
                    "WOWS_MAX_ROUNDS": str(int(limit_value)) if limit_type == "rounds" else "0",
                    "WOWS_DURATION_MINUTES": str(limit_value) if limit_type == "duration" else "0",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONUTF8": "1",
                }
            )
            self.log_stream = LOG_PATH.open("w", encoding="utf-8")
            self.process = subprocess.Popen(
                [sys.executable, str(BASE_DIR / "main.py")],
                cwd=BASE_DIR,
                env=env,
                stdout=self.log_stream,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.run_id = run_id
            self.store.create_run(run_id, ship, mode, limit_type, limit_value)
            return {"run_id": run_id}

    def stop(self):
        with self.lock:
            self._reconcile()
            if self.process is None:
                return False
            STOP_PATH.write_text("stop", encoding="utf-8")
            threading.Thread(
                target=self._watch_stop_request,
                args=(self.process,),
                daemon=True,
            ).start()
            return True

    def resume(self):
        """Release either a Web pause or a keyboard-intervention hold."""
        with self.lock:
            self._reconcile()
            if self.process is None:
                return False
            PAUSE_PATH.unlink(missing_ok=True)
            RESUME_PATH.write_text("resume", encoding="utf-8")
            return True

    def pause(self):
        """Cooperatively freeze new system commands without stopping the ship."""
        with self.lock:
            self._reconcile()
            if self.process is None:
                return False
            RESUME_PATH.unlink(missing_ok=True)
            PAUSE_PATH.write_text("pause", encoding="utf-8")
            return True

    def status(self):
        with self.lock:
            self._reconcile()
            running = self.process is not None
            process_id = self.process.pid if self.process else None
            run_id = self.run_id
        state = self._read_runtime_state()
        if not state:
            state = {"state": "idle", "message": "等待启动"}
        defaults = {
            "completed_rounds": 0,
            "current_round": 0,
            "elapsed_seconds": 0.0,
            "route_phase": "unplanned",
            "route_progress": 0.0,
            "route_waypoint": 0,
            "route_arrived": False,
            "stop_after_current": False,
            "manual_intervention_latched": False,
            "manual_intervention_seconds": 0.0,
        }
        for key, value in defaults.items():
            state.setdefault(key, value)
        paused_by_user = bool(running and PAUSE_PATH.exists())
        state["paused_by_user"] = paused_by_user
        if paused_by_user:
            state["manual_intervention_latched"] = True
            state["movement_mode"] = "manual_pause"
            state["movement_reason"] = "网页手动暂停；保持当前船速和舵位，不再下发新指令"
        if state.get("rewards_status") == "recognized":
            rewards = state.get("last_rewards") or {}
            reward_round = int(state.get("rewards_round") or 0)
            reward_run_id = str(state.get("run_id") or run_id or "")
            if isinstance(rewards, dict) and reward_run_id and reward_round > 0:
                self.store.upsert_auto_rewards(
                    reward_run_id,
                    reward_round,
                    rewards,
                )
        elapsed_seconds = float(state.get("elapsed_seconds") or 0.0)
        if running and state.get("started_at"):
            elapsed_seconds = max(
                elapsed_seconds,
                time.time() - float(state["started_at"]),
            )
            state["elapsed_seconds"] = elapsed_seconds
        if run_id:
            self.store.update_run_progress(
                run_id,
                state.get("completed_rounds", 0),
                elapsed_seconds,
            )
        if not running and state.get("state") in {
            "starting",
            "launching_game",
            "entering_game",
            "preparing",
            "battle",
            "requeueing",
            "returning",
            "collecting_rewards",
        }:
            state.update(
                {
                    "state": "stopped",
                    "message": "运行进程已结束，未保留活动控制",
                    "safety_state": "idle",
                    "movement_verified": False,
                    "target_distance_km": None,
                    "distance_source": "unknown",
                    "minimap_distance_km": None,
                    "distance_confidence": 0.0,
                    "target_track_id": "",
                    "ocr_status": "no_target",
                    "ocr_provider": "uninitialized",
                    "movement_mode": "idle",
                    "movement_reason": "",
                    "capture_point_distance_km": None,
                    "inside_capture_point": False,
                    "route_phase": "unplanned",
                    "route_progress": 0.0,
                    "route_waypoint": 0,
                    "route_arrived": False,
                    "stop_after_current": False,
                }
            )
        if not running and state.get("state") in {"completed", "stopped"}:
            state.update(
                {
                    "movement_mode": "idle",
                    "movement_reason": "控制已安全释放",
                    "route_phase": "unplanned",
                    "route_progress": 0.0,
                    "route_waypoint": 0,
                    "route_arrived": False,
                    "inside_capture_point": False,
                    "stop_after_current": False,
                }
            )
        state.update(
            {
                "running": running,
                "pid": process_id,
                # Keep the latest completed task selected across a control
                # panel restart; a new start replaces it with the new run id.
                "run_id": run_id or state.get("run_id"),
            }
        )
        state["calibration"] = CalibrationStore().status().to_dict()
        state["log"] = tail_log()
        return state


def load_ships():
    with (BASE_DIR / "config" / "ship.yaml").open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return {
        key: {
            "name": value.get("display_name", value.get("name", key)),
            "type": value.get("type", ""),
            "nation": value.get("nation", ""),
            "secondary_range": value.get("secondary", {}).get("range", 0),
            "speed": value.get("navigation", {}).get("speed", 0),
        }
        for key, value in raw.items()
        if key in SUPPORTED_SHIPS
    }


def tail_log(lines=80):
    if not LOG_PATH.exists():
        return []
    try:
        return LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


STORE = ControlStore()
RUNNER = RunnerManager(STORE)
CALIBRATION = WebCalibrationWorkflow()


class ControlHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        return

    def _json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _payload(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 64_000:
            raise ValueError("请求体过大")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/config":
            self._json(
                {
                    "ships": load_ships(),
                    "modes": MODES,
                    "calibration": CalibrationStore().status().to_dict(),
                    "game": game_status(),
                }
            )
            return
        if path == "/api/calibration":
            self._json(CalibrationStore().status().to_dict())
            return
        if path == "/api/calibration/session":
            self._json(CALIBRATION.status())
            return
        if path == "/api/game/status":
            self._json(game_status())
            return
        if path == "/api/status":
            self._json(RUNNER.status())
            return
        if path == "/api/dashboard":
            self._json(STORE.dashboard())
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._payload()
            if path == "/api/run/start":
                self._json({"ok": True, **RUNNER.start(payload)}, HTTPStatus.CREATED)
                return
            if path == "/api/run/stop":
                self._json({"ok": RUNNER.stop()})
                return
            if path == "/api/run/pause":
                self._json({"ok": RUNNER.pause()})
                return
            if path == "/api/run/resume":
                self._json({"ok": RUNNER.resume()})
                return
            if path == "/api/game/launch":
                self._json({"ok": True, **CALIBRATION.launch()})
                return
            if path == "/api/calibration/prepare":
                if RUNNER.status().get("running"):
                    raise RuntimeError("自动任务运行中，不能开始校准")
                self._json(
                    {"ok": True, **CALIBRATION.prepare(payload.get("ship", "pommern"))},
                    HTTPStatus.ACCEPTED,
                )
                return
            if path == "/api/calibration/action":
                self._json({"ok": True, **CALIBRATION.run_action()})
                return
            if path == "/api/calibration/confirm":
                confirmed = payload.get("confirmed")
                if not isinstance(confirmed, bool):
                    raise ValueError("confirmed 必须是布尔值")
                self._json({"ok": True, **CALIBRATION.confirm(confirmed)})
                return
            if path == "/api/calibration/cancel":
                self._json({"ok": True, **CALIBRATION.cancel()})
                return
            if path == "/api/resources":
                run_id = str(payload.get("run_id") or RUNNER.run_id or "")
                if not run_id:
                    raise ValueError("没有可关联的运行任务")
                entry_id = STORE.add_resources(
                    run_id,
                    payload.get("round_no", 0),
                    payload,
                    str(payload.get("note", "")),
                )
                self._json({"ok": True, "entry_id": entry_id}, HTTPStatus.CREATED)
                return
            self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def main():
    host = os.environ.get("WOWS_PANEL_HOST", "127.0.0.1")
    port = int(os.environ.get("WOWS_PANEL_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), ControlHandler)
    url = f"http://{host}:{port}"
    print(f"战舰控制台已启动: {url}")
    if os.environ.get("WOWS_PANEL_NO_BROWSER") != "1":
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
