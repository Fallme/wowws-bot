# Curated visual regression fixtures

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
