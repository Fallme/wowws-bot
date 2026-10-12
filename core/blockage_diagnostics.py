"""Bounded, game-window-only evidence before and after recovery actions."""

import json
import logging
import re
import time
from collections import deque
from pathlib import Path

import cv2


logger = logging.getLogger("blockage")


class BlockageTrace:
    """Keep recovery pairs and terminal evidence even when sampling fills up.

    Ordinary observations are throttled. Explicit ``before``, ``after`` and
    ``result`` frames are never throttled, and can displace older evidence when
    the disk-frame limit is reached. Eviction removes an old action pair as a
    unit, so the latest recovery attempt remains explainable.
    """

    def __init__(self, root, label, *, maximum=60, interval=0.7,
                 max_per_action=12, context_frames=6, lossless=False):
        self.root = None if root is None else Path(root)
        self.label = re.sub(r"[^a-zA-Z0-9_]", "_", str(label)).strip("_") or "blockage"
        # A before/after pair plus the terminal result must fit together.
        self.maximum = max(3, int(maximum))
        self.interval = max(0.0, float(interval))
        self.max_per_action = max(1, int(max_per_action))
        self.lossless = bool(lossless)
        self.count = 0
        self.action_counts = {}
        self.last_saved = float("-inf")
        self.folder = None
        self._sequence = 0
        self._observations = []
        self._started = None
        self._context = deque(maxlen=max(0, int(context_frames)))
        self._last_context = float("-inf")
        self._dropped = 0

    @staticmethod
    def _valid_image(image):
        return image is not None and getattr(image, "size", 0) > 0

    def remember(self, image, *, state, attempt=0, details=None):
        """Keep a few spaced frames in memory until a blockage is recorded.

        Callers can feed already-captured game frames during normal navigation;
        this does not create files or require extra game captures.
        """
        if self.root is None or not self._context.maxlen or not self._valid_image(image):
            return False
        now = time.monotonic()
        if now - self._last_context < self.interval:
            return False
        self._context.append((image.copy(), state, attempt, details, now, time.time()))
        self._last_context = now
        return True

    def record(self, image, *, state, action, attempt, phase="observation", details=None):
        """Save a frame and decision metadata; return whether it was saved.

        Use the same action and attempt for the before/after frames. ``details``
        accepts a metadata dictionary, for example a recovery reason or outcome.
        The original call signature remains valid for sampled observations.
        """
        if self.root is None or not self._valid_image(image):
            return False
        if phase not in {"observation", "context", "before", "after", "result"}:
            raise ValueError(f"Unsupported blockage evidence phase: {phase}")
        now = time.monotonic()
        critical = phase in {"before", "after", "result"}
        if not critical and (
            self.count >= self.maximum
            or self.action_counts.get(action, 0) >= self.max_per_action
            or now - self.last_saved < self.interval
        ):
            return False
        if self._context:
            context = list(self._context)[-(self.maximum - 1):]
            self._context.clear()
            for frame, old_state, old_attempt, old_details, captured, wall_time in context:
                self._save(frame, state=old_state, action="pre_blockage_context",
                           attempt=old_attempt, phase="context", details=old_details,
                           now=captured, wall_time=wall_time)
        return self._save(image, state=state, action=action, attempt=attempt,
                          phase=phase, details=details, now=now, wall_time=time.time())

    def finish(self, image, *, state, action="recovery_result", attempt=0,
               outcome=None, details=None):
        """Record the final success, failure or abort frame without throttling."""
        metadata = dict(details or {})
        if outcome is not None:
            metadata["outcome"] = outcome
        return self.record(image, state=state, action=action, attempt=attempt,
                           phase="result", details=metadata or None)

    def _make_room(self, phase, pair_id):
        if self.count < self.maximum:
            return True
        if phase in {"observation", "context"}:
            return False
        protected = {pair_id} if pair_id is not None else set()
        if phase == "result":
            # A result should retain the action that immediately preceded it.
            pairs = {}
            for row in self._observations:
                if row["pair_id"] is not None:
                    pairs.setdefault(row["pair_id"], set()).add(row["phase"])
            complete = {key for key, phases in pairs.items() if {"before", "after"} <= phases}
            for row in reversed(self._observations):
                if row["pair_id"] in complete:
                    protected.add(row["pair_id"])
                    break
        candidates = [row for row in self._observations if row["phase"] in {"context", "observation"}]
        if not candidates:
            candidates = [row for row in self._observations
                          if row["pair_id"] is not None and row["pair_id"] not in protected]
        if not candidates:
            candidates = [row for row in self._observations if row["phase"] == "result"]
        if not candidates:
            # Repeated frames for the current pair may be replaced individually.
            candidates = [row for row in self._observations if row["phase"] == phase]
        if not candidates:
            return False
        oldest = candidates[0]
        if oldest["pair_id"] is None or oldest["pair_id"] in protected:
            removed = [oldest]
        else:
            removed = [row for row in self._observations if row["pair_id"] == oldest["pair_id"]]
        for row in removed:
            (self.folder / row["frame"]).unlink(missing_ok=True)
            self._observations.remove(row)
            self._dropped += 1
        self.count = len(self._observations)
        return self.count < self.maximum

    def _save(self, image, *, state, action, attempt, phase, details, now, wall_time):
        pair_id = f"{action}:{attempt}" if phase in {"before", "after"} else None
        pending = None
        try:
            if self.folder is None:
                self.folder = self.root / (
                    f"{self.label}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
                )
                self.folder.mkdir(parents=True, exist_ok=True)
            if self.count >= self.maximum and phase in {"observation", "context"}:
                return False
            self._sequence += 1
            extension = ".png" if self.lossless else ".jpg"
            encoding = ([int(cv2.IMWRITE_PNG_COMPRESSION), 1] if self.lossless else
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            name = f"frame_{self._sequence:04d}_{phase}{extension}"
            pending = self.folder / f"pending_frame{extension}"
            if not cv2.imwrite(
                str(pending), image, encoding,
            ):
                raise OSError("OpenCV could not save the captured frame")
            # A failed capture write must not discard the evidence it replaces.
            if not self._make_room(phase, pair_id):
                pending.unlink(missing_ok=True)
                return False
            pending.replace(self.folder / name)
            if self._started is None:
                self._started = now
            row = {
                "frame": name,
                "time": wall_time,
                "elapsed": round(now - self._started, 3),
                "state": getattr(state, "value", str(state)),
                "action": action,
                "attempt": attempt,
                "phase": phase,
                "pair_id": pair_id,
                "evicted_frames": self._dropped,
                "width": image.shape[1],
                "height": image.shape[0],
            }
            if details is not None:
                row["details"] = details
            self._observations.append(row)
            self.count = len(self._observations)
            # The manifest describes retained files, including any frame eviction.
            manifest = self.folder / "observations.jsonl"
            temporary = self.folder / "observations_tmp.jsonl"
            temporary.write_text("".join(json.dumps(item, ensure_ascii=False, default=str) + "\n"
                                         for item in self._observations), encoding="utf-8")
            temporary.replace(manifest)
            if phase == "observation":
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
            self.last_saved = now
            if self._sequence == 1:
                logger.warning("阻塞多帧诊断已开始: %s", self.folder)
            return True
        except (OSError, cv2.error):
            logger.warning("阻塞诊断截图保存失败", exc_info=True)
            if pending is not None:
                try:
                    pending.unlink(missing_ok=True)
                except OSError:
                    pass
            return False
