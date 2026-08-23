# Legacy assets retained from `wows-assistant`

`wowws-bot` is the only runnable project and the only codebase for future
development.  The assets below were retained because they can support future
vision work; they are not imported by the runtime.

| Location | Contents | Intended future use |
| --- | --- | --- |
| `training_assets/legacy_202607/training_data` | 1,735 historical screenshots | Vision-regression samples and model training |
| `training_assets/legacy_202607/yolo` | YOLO dataset, labels, scripts, and weights | Optional object-detection experiments |
| `training_assets/legacy_202607/dataset_v3` | Screenshot and JSON state pairs | Offline state-recognition analysis |
| `reference/legacy_202607` | Ship statistics and secondary-battery ballistics | Expanding ship configurations |
| `assets/legacy_202607_templates` | Historical UI templates | Comparing UI changes and adding template checks |
| `docs/legacy_202607_operation_guide.md` | Previous operation notes | Reference only |
| `training_assets/legacy_runtime/debug` | Historical debug frames | Offline visual regression and future relabeling |
| `training_assets/legacy_runtime/snapshots` | Historical port and menu frames | Port-state regression |
| `training_assets/legacy_runtime/root_screenshots` | Unsorted early captures | Archive only; relabel before reuse |

The archived YOLO scripts retain their original relative paths and are not a
supported entry point.  If we adopt YOLO in this project, create a tested
adapter under `training/` instead of importing the old runtime code.
