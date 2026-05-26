# Results Directory Guide

This folder keeps only the current optimized 300-frame result set at the top level. Historical non-smoke outputs are archived under `Old/`; early smoke outputs were removed because they are development checks rather than paper evidence.

Important boundary: these outputs evaluate Python research baselines inspired by ORB-SLAM2, DSO, and SVO. They are not official paper-level reproductions.

## Current Top-Level Files

| Path | Role |
| --- | --- |
| `final_seq00_300_optimized_f800_p800/` | Latest 300-frame ORB / DSO / SVO benchmark with the optimized DSO implementation. |
| `dso_ablation_300_optimized_p800/` | Latest 300-frame DSO ablation across strict/full and disabled-component configurations. |
| `seq00_300_optimized_results.zip` | Zip archive containing the latest benchmark, latest ablation, and this guide. |
| `README.md` | This results guide. |
| `Old/` | Archived historical results and `Goal.md`. These are not the current main evidence. |

## File Types

| File type | Role |
| --- | --- |
| `benchmark_seq00.json` | Main machine-readable benchmark summary: metrics, runtime, parameters, implementation manifest, robustness stats. |
| `benchmark_trajectory_seq00.png` | Trajectory plot comparing estimates against KITTI ground truth. |
| `*_slam_seq00.txt` | KITTI-format trajectory file for one algorithm. |
| `dso_diagnostics_seq00.json` / `.csv` | Per-frame DSO tracking diagnostics: residuals, valid projections, failures, fallback, BA, loop fields. |
| `dso_diagnostics_seq00.png` | DSO diagnostic plot for valid projected points, inlier ratio, and failure flags. |
| `dso_ablation_seq00.json` / `.csv` | DSO-only ablation summary across configurations. |
| `dso_ablation_seq00.png` | Ablation summary figure. |
| `*.zip` | Packaged rerun artifacts for transfer or archival. |

## Latest 300-Frame Benchmark

Source: `final_seq00_300_optimized_f800_p800/benchmark_seq00.json`.

| Algorithm | ATE RMSE m | RPE trans % | KITTI mean trans % | Failures | Fallbacks | Loop closures | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ORB | 2.9913 | 2.7487 | 2.6843 | 0 | n/a | 2 | Feature baseline unchanged; slow Python local map / BA path. |
| DSO | 3.4279 | 2.0376 | 2.0259 | 12 | 12 | 0 | Optimized strict DSO; 17 candidates, 0 verified/corrected closures, 47/64 BA accepted, 1282 active points culled. |
| SVO | 2.9621 | 1.7526 | 1.7257 | 0 | 0 | 0 | Semi-direct baseline unchanged. |

Compared with the archived `Old/final_seq00_300_f800_p800`, the optimized DSO run changes the interpretation substantially: old DSO had ATE 36.3696 m, 41 failures, and 58 counted loop closures; optimized DSO has ATE 3.4279 m, 12 failures, and 0 true loop closures. The new result is therefore both more accurate and more honest about loop semantics.

## Latest 300-Frame DSO Ablation

Source: `dso_ablation_300_optimized_p800/dso_ablation_seq00.json`.

| Config | ATE RMSE m | RPE trans % | Failures | Fallbacks | Loop candidates | Loop closures | BA accepted / runs | Culled points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_full | 3.4279 | 2.0376 | 12 | 12 | 17 | 0 | 47 / 64 | 1282 |
| no_left_right_check | 3.7716 | 2.8142 | 6 | 6 | 16 | 0 | 53 / 68 | 1558 |
| no_quality_gates | 3.4419 | 2.0374 | 11 | 11 | 17 | 0 | 48 / 65 | 1255 |
| no_joint_ba | 59.3614 | 36.0059 | 12 | 12 | 17 | 0 | 62 / 64 | 511 |
| no_affine_brightness | 3.3253 | 2.0561 | 7 | 7 | 16 | 0 | 64 / 66 | 1746 |
| loop_candidates_only | 3.4279 | 2.0376 | 12 | 12 | 17 | 0 | 47 / 64 | 1282 |
| lk_pnp_only | 3.7976 | 3.0281 | 9 | 9 | 15 | 0 | 49 / 65 | 1416 |
| photometric_with_motion_gate | 2.8182 | 1.7395 | 25 | 25 | 16 | 0 | 42 / 59 | 2480 |
| grid_uniform_selection | 4.0708 | 2.5954 | 16 | 16 | 13 | 0 | 47 / 63 | 1238 |
| outlier_culling_on | 3.6680 | 2.1737 | 17 | 17 | 13 | 0 | 46 / 61 | 2582 |
| outlier_culling_off | 2.8182 | 1.7395 | 25 | 25 | 16 | 0 | 42 / 59 | 2480 |

Key reading order:

1. Start with `strict_full` as the default optimized DSO.
2. Compare `no_joint_ba` against `strict_full`: this is the clearest evidence that active-window BA is essential in the new implementation.
3. Compare `loop_candidates_only` against `strict_full`: both have 0 loop closures, confirming strict loop verification prevents candidates from being reported as true closures.
4. Compare `Old/final_seq00_300_f800_p800` against `final_seq00_300_optimized_f800_p800`: this is the before/after story for the six DSO enhancements.

## Archived Content

`Old/` contains historical non-smoke outputs such as earlier 20/100-frame runs, pre-optimization 300-frame results, intermediate DSO validation outputs, and the historical `Goal.md` task file. These files can help explain development history, but the current paper tables should use the top-level optimized 300-frame outputs.

Early directories whose names contained `smoke` were deleted from `results/` because they were transient runtime checks and should not be cited as final evidence.

