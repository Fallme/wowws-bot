# Curated visual regression fixtures

`wgc_cold_start_20261012.log` is a separate nonvisual startup fixture. It retains
only the session header, target installation Ready state, initialization and
postload timestamps from the October 12 WG log. The failed retry occurred
between initialization and postload. It contains no account or login payloads.

These seven frames are copied from the local historical capture archive and
are the minimum portable set used by the automated regression suite. They
cover port selection, mode selection, loading, battle HUD, escape menu, and
results states.

The full-resolution source archive remains under `training_assets/` and is not
loaded by the production runtime.

`battle_low_contrast_1440.png` is a local 2560x1440 battle capture from
2026-09-12. The minimap contrast is below the old fixed HUD threshold despite
a visible player marker and independent HUD anchors. It reproduces the
PORT/LOADING recovery loop without sending game input.

`battle_port_color_conflict.png` is the unmodified 2560x1494 `full_0040.png`
from the local `run_20260921_124701_1789966021467861100` archive. It shows
an active match at 14:43, but broad port/button colours veto the ordinary
HUD detector. The continuity regression requires independent numeric HP
and match-clock evidence to recover BATTLE, without changing port navigation.

`assets/ui/fire_warning_mask.png` is a 40x40 binary glyph reference extracted
from the own-ship HP warning in the 2026-10-05 Rupprecht trial's retained
`health_reading_20261005_180808` lossless capture. It includes the triangle and
the internal flame, without the surrounding account or battle information.
The detector compares both the whole shape and its interior in two HUD
regions; one warning or a different triangle symbol is insufficient.

`torpedo_prediction_live_2560.png` is the unmodified 2560x1600 observation
from the 2026-10-06 11:34 Rupprecht co-op trial. The visible faint white
prediction fan crosses shell trails; raw foreground/background correlation
previously rejected it. Geometry-only regressions inject weapon identity and
verify the native and 75% images recover the lead while remaining unaligned
and unsafe to fire. Production still requires actual weapon/cooldown evidence.
