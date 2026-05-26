# KITTI ORB / DSO / SVO Python SLAM Baselines

本仓库是在 KITTI Odometry stereo 数据上运行的 Python 视觉 SLAM / VO 研究基线。当前统一比较三条路线：

- `ORB-SLAM2-inspired stereo SLAM`
- `DSO-inspired direct sparse stereo SLAM`
- `SVO-style semi-direct stereo visual odometry/SLAM`

重要边界：这里提供的是 **reproducible Python research baselines**，不是 ORB-SLAM2、DSO 或 SVO 的官方 C++ 完整论文级复现。`run_benchmark.py` 输出的 `implementation_manifest.paper_level_claim` 必须保持 `false`。

## 当前状态

当前代码已经从早期“两算法 VO demo”演进为三算法统一 benchmark。旧入口 `orb_vo.py` 和 `direct_vo.py` 已经不再存在，也不需要恢复；完整运行入口是 `run_benchmark.py`，DSO 消融入口是 `dso_ablation.py`。

本次梳理后的文档判断：

| 文件 | 状态 | 处理建议 |
| --- | --- | --- |
| `README.md` | 当前主说明文件 | 已按实际仓库更新为完整版 |
| `PAPER_LEVEL_STATUS.md` | 有用 | 保留，作为论文级复现边界说明 |
| `Goal.md` | 阶段性任务书 | 可保留归档，不作为用户使用文档 |
| `AI.md` | 早期课程规划 | 内容已过时，可保留归档或移入 `docs/archive/` |
| `00000_Report/slam_report_Latex/README.md` | 报告编译说明 | 有用，仅服务 LaTeX 报告目录 |

算法文件判断：

| 文件 | 作用 | 是否保留 |
| --- | --- | --- |
| `run_benchmark.py` | ORB / DSO / SVO 统一 benchmark、JSON 和图输出 | 保留，主入口 |
| `dso_ablation.py` | DSO 消融实验入口 | 保留 |
| `orb_slam.py` | ORB-SLAM2-inspired 主实现 | 保留 |
| `dso_slam.py` | DSO-inspired 主实现、diagnostics、motion gate、fallback | 保留 |
| `svo_slam.py` | SVO-style semi-direct 主实现 | 保留 |
| `kitti_utils.py` | KITTI 数据加载、标定、轨迹保存 | 保留 |
| `evaluate.py` | 指标、轨迹读取、绘图、JSON safe | 保留；CLI 仍偏传统双算法，三算法请用 `run_benchmark.py` |
| `slam_profiles.py` | manifest 与 paper-level claim | 保留 |
| `slam_base.py` | KeyFrame、MapPoint、图结构、BA / pose graph 基础 | 保留 |
| `local_mapping.py` | ORB/local mapping 共享后端 | 保留 |
| `loop_detector.py` | BoW / loop candidate / Sim3 工具 | 保留 |
| `orb_advanced.py` | ORB local map、relocalization、Sim3 fusion、global BA | 保留 |
| `dso_advanced.py` | DSO photometric calibration、inverse-depth active window、prior | 保留 |
| `orb_vo.py` | 旧 ORB VO demo 文件，已删除 | 无需恢复 |
| `direct_vo.py` | 旧 direct VO demo 文件，已删除 | 无需恢复 |

## 数据目录

当前仓库按 KITTI Odometry 结构读取双目灰度图像：

```text
data/kitti_odometry/
  sequences/
    00/
      image_0/
      image_1/
      calib.txt
      times.txt
    poses/
      00.txt
```

`KITTIOdometryLoader` 也兼容真值位于 `data/kitti_odometry/poses/00.txt` 的常见布局。当前本地 `sequence 00` 左右目各有 4541 帧。

## 安装

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖见 `requirements.txt`：

```text
opencv-python
numpy
scipy
matplotlib
scikit-learn
```

## 推荐运行

20 帧三算法 smoke benchmark：

```powershell
.\.venv\Scripts\python.exe -B run_benchmark.py --data-dir data/kitti_odometry --seq 00 --max-frames 20 --algorithms orb,dso,svo --features 800 --points 800 --svo-points 1500 --output-dir results\tri_slam_20
```

100 帧三算法 benchmark：

```powershell
.\.venv\Scripts\python.exe -B run_benchmark.py --data-dir data/kitti_odometry --seq 00 --max-frames 100 --algorithms orb,dso,svo --features 1500 --points 1500 --svo-points 1500 --output-dir results\tri_slam_100
```

DSO 100 帧消融实验：

```powershell
.\.venv\Scripts\python.exe -B dso_ablation.py --data-dir data/kitti_odometry --seq 00 --max-frames 100 --points 1500 --output-dir results\dso_ablation_100
```

结构验证：

```powershell
.\.venv\Scripts\python.exe -m py_compile dso_ablation.py dso_advanced.py dso_slam.py evaluate.py kitti_utils.py local_mapping.py loop_detector.py orb_advanced.py orb_slam.py run_benchmark.py slam_base.py slam_profiles.py svo_slam.py
```

注意：当前根目录没有 `tests/` 目录。若需要正式回归测试，建议补回 `tests/test_reliability.py`，覆盖 manifest、JSON safe、trajectory length mismatch、DSO diagnostics 和 SVO short-sequence。

## Benchmark 输出

`run_benchmark.py` 会生成：

```text
orb_slam_seq00.txt
dso_slam_seq00.txt
svo_slam_seq00.txt
benchmark_seq00.json
benchmark_trajectory_seq00.png
dso_diagnostics_seq00.json
dso_diagnostics_seq00.csv
dso_diagnostics_seq00.png
```

`dso_ablation.py` 会生成：

```text
dso_ablation_seq00.json
dso_ablation_seq00.csv
dso_ablation_seq00.png
dso_full_simplified_dso_seq00.txt
dso_lk_pnp_only_seq00.txt
dso_photometric_with_motion_gate_seq00.txt
dso_grid_uniform_selection_seq00.txt
dso_outlier_culling_on_seq00.txt
dso_outlier_culling_off_seq00.txt
```

## 已有结果

以下数值来自当前仓库已有 JSON 文件，不是手填推测。

`results/tri_slam_20/benchmark_seq00.json`

| Algorithm | ATE RMSE (m) | Runtime avg/median/p95 (ms) | Wall time (s) | Keyframes | Tracking failures | Fallbacks | Loop closures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ORB | 0.9367 | 6078.94 / 91.78 / 28712.37 | 121.92 | 7 | 0 | 0 | 0 |
| DSO | 1.6427 | 610.33 / 386.47 / 1216.79 | 12.16 | 7 | 0 | 0 | 0 |
| SVO | 1.1066 | 118.25 / 163.89 / 189.46 | 2.76 | 12 | 0 | 0 | 0 |

`results/tri_slam_100/benchmark_seq00.json`

| Algorithm | ATE RMSE (m) | Runtime avg/median/p95 (ms) | Wall time (s) | Keyframes | Tracking failures | Fallbacks | Loop closures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ORB | 2.1349 | 108781.18 / 149.90 / 63623.81 | 10881.14 | 34 | 0 | 0 | 0 |
| DSO | 87.8995 | 441.29 / 261.63 / 1250.25 | 44.97 | 40 | 6 | 6 | 0 |
| SVO | 2.2858 | 69.35 / 82.55 / 104.25 | 7.87 | 75 | 0 | 0 | 0 |

说明：

- 20 / 100 帧短序列没有足够的 100m+ path segment，因此 `rpe_trans_percent`、`rpe_rot_deg_per_m` 和 KITTI segment mean 可能为 `null`。
- `tri_slam_100` 中 ORB 的 wall time 和 runtime 平均值存在明显长尾，应在写报告前重新运行或单独 profiling，不建议直接把该耗时作为最终效率结论。
- `tri_slam_100` 中 DSO drift 很明显，应解释为当前 Python baseline 的实现限制，不代表原始 DSO 方法本身失败。

`results/dso_ablation_100/dso_ablation_seq00.json`

| Config | ATE RMSE (m) | Tracking failures | Fallbacks | Motion gate rejections | Mean valid points | Mean inlier ratio | Avg runtime (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_simplified_dso | 40.0404 | 1 | 1 | 0 | 1113.22 | 0.7338 | 734.29 |
| lk_pnp_only | 77.2572 | 0 | 0 | 0 | 1115.00 | 0.6717 | 616.25 |
| photometric_with_motion_gate | 75.7913 | 17 | 17 | 10 | 1346.86 | 0.4826 | 534.43 |
| grid_uniform_selection | 11.6835 | 1 | 1 | 0 | 1113.49 | 0.7381 | 836.57 |
| outlier_culling_on | 9.5661 | 14 | 14 | 10 | 1361.77 | 0.4819 | 593.63 |
| outlier_culling_off | 75.7913 | 17 | 17 | 10 | 1346.86 | 0.4826 | 535.78 |

## Notebook

`slam_comparison.ipynb` 是当前推荐的交互式查看入口。它会读取：

```text
results/tri_slam_20/benchmark_seq00.json
results/tri_slam_100/benchmark_seq00.json
results/dso_ablation_100/dso_ablation_seq00.json
```

并展示三算法结果表、鲁棒性统计、轨迹图、DSO diagnostics 图和 DSO ablation 表。旧 notebook 中引用的 `orb_vo.py`、`direct_vo.py`、`orb_seq00.txt`、`direct_seq00.txt` 已过时。

## 实现边界

### ORB-SLAM2-inspired

已包含 ORB 特征、双目深度、PnP RANSAC、关键帧、局部建图、local map tracking、BoW/PnP relocalization、loop candidate、Sim3 loop fusion、pose graph 和 bounded BA。仍缺少官方级离线大词袋、完整 spanning tree / essential graph 策略和成熟 map point replacement / loop fusion 细节。

### DSO-inspired

已包含高梯度点、stereo depth 初始化、coarse-to-fine photometric tracking、LK+PnP 初始化、Huber photometric residual、affine brightness、inverse-depth active window、轻量 prior、motion gate、fallback、逐帧 diagnostics 和 ablation。仍不是完整 DSO 的联合 photometric BA、Schur complement marginalization 和数据集标定文件驱动的 photometric calibration。

### SVO-style

已包含 stereo depth 初始化、grid-uniform sparse point selection、pyramidal LK patch tracking、PnP RANSAC、motion gate、fallback、关键帧插入和稀疏点维护。未实现原始 SVO 的完整概率深度滤波、不确定性传播、成熟重定位和回环后端。

## 建议清理

不建议删除当前根目录 `.py` 算法文件；它们都仍有引用或承担入口/共享模块职责。可以考虑后续做两类整理：

- 把 `AI.md`、`Goal.md` 移入 `docs/archive/`，避免和当前 README 冲突。
- 把 `evaluate.py` 的 CLI 扩展到直接支持 SVO，或在 README 中继续明确三算法评测统一走 `run_benchmark.py`。

## 报告表述建议

建议写作时使用：

```text
Python research baselines inspired by ORB-SLAM2, DSO, and SVO.
```

不要写成：

```text
official/full paper-level reproduction of ORB-SLAM2, DSO, and SVO
```

对 DSO 结果应谨慎表述：

```text
The simplified DSO-inspired Python baseline remains sensitive to photometric assumptions, initialization quality, and incomplete joint photometric optimization. The observed drift should be interpreted as an implementation-level limitation rather than a failure of the original DSO method.
```
