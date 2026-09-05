"""Resolution-independent UI geometry shared by recognition and clicking."""

from dataclasses import dataclass
from enum import Enum


class ScreenState(str, Enum):
    PORT = "port"
    DAILY_REWARD = "daily_reward"
    LOADING = "loading"
    BATTLE = "battle"
    ESCAPE_MENU = "escape_menu"
    EXIT_CONFIRMATION = "exit_confirmation"
    PORT_EXIT_CONFIRMATION = "port_exit_confirmation"
    SURVEY = "survey"
    RESULTS = "results"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RelativeRegion:
    left: float
    top: float
    right: float
    bottom: float

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(width * self.left),
            round(height * self.top),
            round(width * self.right),
            round(height * self.bottom),
        )

    def center(self, width: int, height: int) -> tuple[int, int]:
        x1, y1, x2, y2 = self.pixels(width, height)
        return ((x1 + x2) // 2, (y1 + y2) // 2)


PORT_BATTLE_BUTTON = RelativeRegion(0.455, 0.002, 0.545, 0.043)
# Current-mode emblem and label immediately to the right of ``加入战斗``.
# Keep the region tight: the previous wide box clicked the empty header area
# to the right of the label at 2560x1600.
PORT_MODE_SELECTOR = RelativeRegion(0.535, 0.002, 0.625, 0.050)
# Battle-type selection screen.  The standard PvE card is the first card in
# the upper row.  Event cards can move between slots, so asymmetric battles
# are located by their purple emblem inside the search area instead.
BATTLE_TYPE_COOPERATIVE_CARD = RelativeRegion(0.340, 0.255, 0.425, 0.430)
BATTLE_TYPE_SEARCH_AREA = RelativeRegion(0.120, 0.170, 0.880, 0.580)
BATTLE_TYPE_BACK_BUTTON = RelativeRegion(0.000, 0.010, 0.075, 0.075)

# Reference templates captured at 2560x1440.  Runtime matching searches the
# whole visible carousel and clicks the actual match, not a fixed coordinate.
SHIP_REFERENCE_SIZE = (2560, 1440)
SHIP_NAME_TEMPLATES = {
    "napoli": "napoli_name.png",
    "pommern": "pommern_name.png",
}
SELECTED_SHIP_NAME_TEMPLATES = {
    "napoli": "napoli_selected_name.png",
    "pommern": "pommern_selected_name.png",
}
PORT_DIALOG_CLOSE = RelativeRegion(0.585, 0.650, 0.620, 0.690)
# Commander-less ships show a blocking confirmation after ``加入战斗``.
# The affirmative button is stable in the center-left of the dialog.
NO_COMMANDER_CONFIRM_BUTTON = RelativeRegion(0.420, 0.535, 0.495, 0.615)
LOADING_START_BUTTON = RelativeRegion(0.455, 0.945, 0.545, 0.995)
ESCAPE_RESUME_BUTTON = RelativeRegion(0.445, 0.465, 0.555, 0.515)
# Quick-battle early exit uses a two-step menu. These regions are fallbacks
# only after OCR has positively identified the live-battle menu/confirmation.
QUICK_EXIT_BATTLE_BUTTON = RelativeRegion(0.445, 0.385, 0.555, 0.435)
QUICK_EXIT_CONFIRM_YES_BUTTON = RelativeRegion(0.425, 0.590, 0.500, 0.640)
EXIT_CONTINUE_BUTTON = RelativeRegion(0.535, 0.455, 0.615, 0.510)
# Result screen: the left teal button returns to port; the right orange button
# immediately queues another battle.  The runner prefers the right button
# between rounds and falls back to the verified port workflow when unavailable.
RESULTS_RETURN_TO_PORT_BUTTON = RelativeRegion(0.455, 0.895, 0.545, 0.935)
RESULTS_REQUEUE_BUTTON = RelativeRegion(0.765, 0.895, 0.850, 0.935)
# Backwards-compatible alias for callers/tests written before the distinction.
RESULTS_CONTINUE_BUTTON = RESULTS_RETURN_TO_PORT_BUTTON
HEALTH_BAR_REGION = RelativeRegion(0.014, 0.785, 0.1015, 0.817)
MINIMAP_REGION = RelativeRegion(0.730, 0.565, 1.000, 1.000)
# Fire markers appear as orange flame-droplets on the speed-dial ship
# silhouette (bottom-left) and on the viewport ship model.  The viewport band
# stops where the minimap begins: the minimap draws its own fire markers for
# every burning spotted ship, which would otherwise pass every droplet test.
FIRE_DIAL_ROI = RelativeRegion(0.020, 0.840, 0.120, 0.940)
FIRE_SHIP_ROI = RelativeRegion(0.150, 0.550, MINIMAP_REGION.left, 0.900)
# Active damage timers (fire, flooding or a disabled critical module) appear
# as compact coloured cards immediately above the consumable row. This anchor
# is much more stable than flames rendered on a distant/occluded ship model.
FIRE_STATUS_ROI = RelativeRegion(0.440, 0.760, 0.530, 0.840)
