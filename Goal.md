# Codex 执行任务：三算法 SLAM baseline 工程补齐与实验输出

你现在负责代码实现、调试、实验运行、指标输出和测试补齐。不要写论文。最终论文由 GPT Pro 在你完成实验后基于真实结果撰写。

本项目是 Python SLAM research baseline，基于 KITTI Odometry stereo 数据集。当前已有两个算法：

1. `ORB-SLAM2-inspired stereo SLAM`
2. `DSO-inspired direct sparse stereo SLAM`

现在需要完成两件核心工作：

1. 诊断并稳定当前 DSO baseline。
2. 新增第三算法：`SVO-style semi-direct stereo visual odometry/SLAM`，并集成到统一 benchmark。

必须遵守以下硬约束：

* 不得声称当前实现是 ORB-SLAM2、DSO、SVO 的官方完整复现。
* `implementation_manifest.paper_level_claim` 必须保持 `false`。
* 所有结果必须来自实际运行，不得伪造。
* 不得删除失败帧、静默截断轨迹或修改评估协议来美化结果。
* 轨迹长度与 ground truth 不一致时必须报错，除非显式实验说明。
* 所有 JSON 必须是标准 JSON，`NaN`、`inf` 必须通过 `json_safe` 转换为 `null`。
* 修改要小步、可测试、可回滚，不能重写整个项目。
* 所有新增算法都必须进入统一 `run_benchmark.py`，不能只做孤立 demo。

---

## Phase 0：先读代码并建立改动边界

先阅读以下文件，确认现有接口和数据流：

* `kitti_utils.py`
* `evaluate.py`
* `run_benchmark.py`
* `slam_profiles.py`
* `slam_base.py`
* `local_mapping.py`
* `orb_slam.py`
* `orb_advanced.py`
* `dso_slam.py`
* `dso_advanced.py`
* `tests/test_reliability.py`
* `README.md`
* `PAPER_LEVEL_STATUS.md`

当前仓库状态要点：

* `run_benchmark.py` 当前只运行 ORB 和 DSO。
* `slam_profiles.py` 当前只包含 ORB 和 DSO profile。
* `tests/test_reliability.py` 当前已有 manifest、JSON safe、trajectory mismatch、ORB/DSO short sequence 等测试。
* 当前没有 `svo_slam.py`。
* 当前 DSO 已有 LK+PnP 初始化、photometric tracking、active inverse-depth window、local mapping、loop detector 等简化组件，但缺少逐帧 diagnostics、ablation 输出、强 motion gate 和系统性 fallback 记录。

执行前先跑一次现有测试，记录 baseline：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

如果在 Linux/macOS 环境：

```bash
python -B -m unittest discover -s tests
```

不要在第一步重写架构。先确认现有测试是否通过，若失败，先记录失败原因。

---

# Phase 1：DSO diagnostics、motion gate、fallback 与 ablation

目标：降低 DSO catastrophic drift 和 tracking failures，增加可解释诊断信息。优化后 DSO 不要求超过 ORB，但必须能解释漂移和失败来源。

## 1.1 文件级任务

重点修改：

* `dso_slam.py`
* `dso_advanced.py`
* `run_benchmark.py`
* `evaluate.py`，仅在需要新增 plotting 或 JSON-safe helper 时小改
* `tests/test_reliability.py`
* `README.md`
* `PAPER_LEVEL_STATUS.md`

可新增文件，但不强制：

* `dso_diagnostics.py`
* `dso_ablation.py`

如果新增文件，请保持接口简单，不要分散过度。

---

## 1.2 DSO 逐帧 diagnostics

在 `dso_slam.py` 中为 `DSOSLAM` 增加逐帧 diagnostics 记录。建议使用 dataclass 或普通 dict。每一帧都必须有一条记录，包括第一帧。第一帧中无法计算的运动量、residual 可以设为 `None`，不能写 `NaN`。

每帧 diagnostics 至少包含：

```python
{
    "frame_id": int,
    "timestamp": float,
    "tracking_cost": float | None,
    "valid_projected_points": int,
    "residual_mean": float | None,
    "residual_median": float | None,
    "residual_p95": float | None,
    "inlier_ratio": float | None,
    "relative_translation_m": float | None,
    "relative_rotation_deg": float | None,
    "tracking_failure": bool,
    "fallback_used": bool,
    "fallback_reason": str | None,
    "keyframe_inserted": bool,
    "active_window_keyframes": int,
    "active_inverse_depth_points": int
}
```

在 `DSOSLAM` 中新增：

```python
self.diagnostics: list[dict] = []
```

并新增方法：

```python
def get_diagnostics(self) -> list[dict]:
    ...

def get_robustness_summary(self) -> dict:
    ...
```

`get_stats()` 中也要包含 robustness summary 的关键字段，例如：

* `frames_processed`
* `tracking_failures`
* `fallbacks_used`
* `motion_gate_rejections`
* `low_projection_rejections`
* `mean_valid_projected_points`
* `median_valid_projected_points`
* `mean_inlier_ratio`
* `active_inverse_depth_points`
* `loop_candidates`
* `loop_closures`

---

## 1.3 DSO diagnostics 输出文件

在 `run_benchmark.py` 中，当运行 DSO 后，输出：

```text
results/.../dso_diagnostics_seq00.json
results/.../dso_diagnostics_seq00.csv
results/.../dso_diagnostics_seq00.png
```

JSON 必须使用：

```python
json.dump(json_safe(obj), f, indent=2, ensure_ascii=False, allow_nan=False)
```

CSV 可选，但 JSON 必须有。

Diagnostic plot 至少画出以下曲线之一，建议三条都画：

* tracking cost over frame id
* valid projected points over frame id
* relative translation / relative rotation over frame id

建议新增函数：

```python
def plot_dso_diagnostics(diagnostics: list[dict], output_path: Path) -> None:
    ...
```

可以放在 `evaluate.py` 或 `dso_slam.py`。如果放在 `evaluate.py`，不要破坏现有 `plot_trajectories` 接口。

---

## 1.4 DSO residual statistics 和 valid projection count

当前 `compute_photometric_residuals()` 只返回 residual array。请新增一个轻量 helper，用于计算 diagnostics：

```python
def compute_photometric_residual_stats(... ) -> dict:
    ...
```

或让 `DSOTacker.estimate_pose()` 在每次 tracking 后记录：

```python
self.last_valid_projected_points
self.last_residual_mean
self.last_residual_median
self.last_residual_p95
self.last_inlier_ratio
self.last_tracking_cost
self.last_tracking_success
```

inlier 可以定义为：

```python
abs(residual) < residual_inlier_threshold
```

例如 threshold 初始设为 20 或 30，并允许通过 config 调参。

注意：`valid_projected_points` 不能只等于 active points 总数，必须反映当前帧中成功投影且在图像边界内的点数。

---

## 1.5 DSO motion gate

新增 motion gate，用于拒绝明显异常的单帧相对运动。建议在 `DSOSLAM` 或 `DSOTacker` 中实现：

```python
def relative_motion_stats(T_ref_cur: np.ndarray) -> dict:
    ...
```

输出：

* `translation_m`
* `rotation_deg`

建议初始阈值：

```python
max_translation_m = 5.0
max_rotation_deg = 15.0
min_valid_projected_points = 30
min_inlier_ratio = 0.15
```

不要把阈值写死到不可调。建议新增 config dataclass：

```python
@dataclass
class DSOTrackingConfig:
    max_translation_m: float = 5.0
    max_rotation_deg: float = 15.0
    min_valid_projected_points: int = 30
    min_inlier_ratio: float = 0.15
    residual_inlier_threshold: float = 25.0
    enable_motion_gate: bool = True
    enable_outlier_culling: bool = True
    enable_grid_selection: bool = True
    enable_photometric_refinement: bool = True
    fallback_mode: str = "lk_pnp_then_constant_velocity"
```

如果改动量过大，可以放在 `dso_slam.py` 内部，不必新建文件。

当 motion gate 触发时：

* 不接受 photometric refinement 的结果。
* 回退到 LK+PnP 初值、constant velocity prediction 或上一帧安全位姿。
* 必须保证仍然 append 一个 pose，轨迹长度不能减少。
* diagnostics 中必须记录：

  * `fallback_used = True`
  * `fallback_reason = "motion_gate"` 或更具体原因
  * `tracking_failure = True` 或单独记录 rejection，二者需定义清楚。

推荐 fallback 优先级：

1. LK+PnP result 如果通过 motion gate，则使用 LK+PnP。
2. constant velocity prediction 如果通过 gate，则使用 constant velocity。
3. 否则使用上一帧安全位姿，保持轨迹长度。

不要静默接受异常大跳变。

---

## 1.6 DSO grid-uniform high-gradient point selection

当前 `select_high_gradient_pixels()` 可能导致点集中在少数高纹理区域。请实现 grid-uniform high-gradient selection。

建议保留原函数，并新增：

```python
def select_grid_uniform_high_gradient_pixels(
    gray: np.ndarray,
    depth: np.ndarray,
    num_points: int = 1500,
    grid_rows: int = 8,
    grid_cols: int = 12,
    border: int = 10,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

要求：

* 图像划分为网格。
* 每个网格选若干梯度最高点。
* depth 必须有效。
* 避免边界像素。
* 返回格式与现有 `select_high_gradient_pixels()` 兼容。
* 点数不足时允许少于 `num_points`，但不能出错。
* 保留原函数用于 ablation 对比。

在 `DSOTacker.make_keyframe()` 中根据 config 决定使用原始 selection 还是 grid-uniform selection。

---

## 1.7 DSO outlier culling

新增以下 outlier filtering：

1. Depth range filtering：

   * 默认 `0.5m < depth < 80m` 或合理 KITTI 范围。
2. Photometric residual outlier removal：

   * tracking 后根据 residual threshold 或 robust percentile 去除明显异常点。
3. Low valid-projection count rejection：

   * 如果 valid projected points 太少，不接受 photometric pose。
4. Stereo disparity consistency：

   * 如果现有 `compute_stereo_depth()` 能拿到 disparity，则过滤 disparity <= 0。
   * 如果暂时无法做 left-right consistency，必须在代码注释和 stats 中说明当前只做 disparity validity，不做完整 left-right consistency。

不要为了过滤而减少轨迹长度。过滤失败时回退。

---

## 1.8 DSO ablation 支持

至少支持以下配置：

1. `full_simplified_dso`
2. `lk_pnp_only`
3. `photometric_with_motion_gate`
4. `grid_uniform_selection`
5. `outlier_culling_on`
6. `outlier_culling_off`

可以通过 `run_benchmark.py` 新增 CLI 参数：

```bash
--dso-mode full
--dso-mode lk_pnp_only
--dso-mode photometric_gate
--dso-grid-selection
--dso-no-grid-selection
--dso-outlier-culling
--dso-no-outlier-culling
```

也可以新增单独脚本：

```bash
python -B dso_ablation.py --seq 00 --max-frames 100 --points 1500 --output-dir results/dso_ablation_100
```

但即使新增 `dso_ablation.py`，`run_benchmark.py` 中也必须保留 DSO 主配置集成。

Ablation 输出至少包括：

```text
results/dso_ablation_100/dso_ablation_seq00.json
results/dso_ablation_100/dso_ablation_seq00.csv
results/dso_ablation_100/dso_ablation_seq00.png
```

`dso_ablation_seq00.json` 中每个 config 至少包含：

* ATE RMSE
* RPE translational / rotational error
* runtime avg / median / p95
* wall_time_s
* keyframes
* tracking_failures
* fallback count
* motion gate rejection count
* mean valid projected points
* mean inlier ratio

---

# Phase 2：实现第三算法 SVO-style semi-direct stereo visual odometry/SLAM

新增算法名称必须准确写成：

```text
SVO-style semi-direct stereo visual odometry/SLAM
```

不要改成官方 SVO，不要声称 full reproduction。

---

## 2.1 新增文件

必须新增：

* `svo_slam.py`

可选新增：

* `svo_advanced.py`

如果实现较短，先把核心逻辑放在 `svo_slam.py`，避免过度拆分。

---

## 2.2 SVO 类接口

在 `svo_slam.py` 中实现以下类之一：

```python
class SVOStyleSLAM:
    ...
```

或：

```python
class SVOSLAM:
    ...
```

推荐使用：

```python
class SVOStyleSLAM:
```

必须提供接口：

```python
def __init__(
    self,
    loader: KITTIOdometryLoader,
    num_points: int = 1500,
    ...
):
    ...

def run(self) -> tuple[list[np.ndarray], list[float]]:
    ...

def get_stats(self) -> dict:
    ...
```

`run()` 必须返回：

* `trajectory: list[np.ndarray]`，每帧一个 4x4 pose。
* `runtimes: list[float]`，至少每个 processed tracking step 一个 runtime；如果第一帧不计 runtime，也要在 summary 中解释一致。

轨迹长度必须等于输入帧数。不能丢帧。

---

## 2.3 SVO 输入和初始化

使用现有：

* `KITTIOdometryLoader`
* stereo 左右目图像
* calibration
* `disparity_to_depth`
* trajectory save/evaluation infrastructure

第一帧：

1. 转灰度或直接使用 loader 输出灰度，按现有格式处理。
2. 计算 stereo disparity/depth。
3. 使用 grid-uniform feature/high-gradient point selection。
4. 初始化 sparse 3D map points。
5. 创建第一关键帧。
6. pose 为 identity。

---

## 2.4 SVO 点选择

SVO-style 不能只做 ORB descriptor matching。点选择可以结合：

* high-gradient pixels
* corners
* ORB keypoints 位置

但 tracking 必须基于 patch / optical flow，不得只基于 ORB descriptor matching。

建议实现：

```python
def select_svo_points_grid(
    gray: np.ndarray,
    depth: np.ndarray,
    num_points: int,
    grid_rows: int = 8,
    grid_cols: int = 12,
    border: int = 12,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
) -> np.ndarray:
    ...
```

要求：

* spatially uniform
* depth valid
* sufficient gradient or cornerness
* avoid border
* output `uvs` shape `(N, 2)` float32

---

## 2.5 SVO stereo depth 初始化

实现或复用 depth 计算：

```python
disparity = compute_stereo_depth(left, right)
depth = disparity_to_depth(disparity, calibration)
```

过滤：

* disparity <= 0
* depth <= min_depth
* depth >= max_depth
* NaN/inf
* border pixels

将 2D points + depth 转为 camera-frame 3D：

```python
x = (u - cx) * z / fx
y = (v - cy) * z / fy_or_fx
z = depth
```

如果 calibration 中只有统一 focal length，则沿用现有项目处理方式。

---

## 2.6 SVO 半直接 tracking

核心要求：使用 patch/photometric tracking 得到当前帧 2D 点，再用 3D-2D PnP 估计位姿。

建议 baseline：

```python
cv2.calcOpticalFlowPyrLK(ref_gray, curr_gray, ref_uvs, None, ...)
```

流程：

1. 从当前 reference keyframe 取：

   * `ref_uvs`
   * `ref_points_3d`
   * `ref_gray`
2. 使用 LK optical flow 跟踪到当前帧：

   * `curr_uvs`
   * `status`
   * `err`
3. 过滤：

   * status valid
   * in image bounds
   * LK error 小于阈值
   * optionally forward-backward check
4. 使用：

   * object points = reference 3D sparse points
   * image points = tracked current 2D points
5. 调用 `cv2.solvePnPRansac()`。
6. 使用 inlier mask 更新当前 tracks。
7. 可选使用 `cv2.solvePnPRefineLM()` refine pose。
8. 对异常 motion 使用 motion gate。
9. tracking 失败时 fallback 到 constant velocity 或上一安全位姿。

这就是 semi-direct / patch-tracking 成分。不要退化为 ORB descriptor matching。

---

## 2.7 SVO motion gate 和 fallback

SVO 也需要 motion gate，避免异常跳变。可以复用 DSO 中的 motion helper，也可以在 `svo_slam.py` 中实现简洁版本。

建议默认阈值：

```python
max_translation_m = 5.0
max_rotation_deg = 15.0
min_tracked_points = 30
min_pnp_inliers = 15
min_inlier_ratio = 0.2
```

fallback 策略：

1. PnP 成功且 motion 合理：接受。
2. 如果 PnP 失败或 motion gate 触发：使用 constant velocity prediction。
3. 如果 constant velocity 也不可用：使用上一帧 pose。

必须记录：

* `tracking_failures`
* `fallbacks_used`
* `motion_gate_rejections`
* `tracked_points`
* `pnp_inliers`
* `inlier_ratio`

---

## 2.8 SVO keyframe 策略

新增 keyframe 条件：

插入 keyframe 当满足任一条件：

* tracked point count 低于阈值
* PnP inlier ratio 低于阈值
* relative translation 超过阈值
* relative rotation 超过阈值
* average optical flow/parallax 超过阈值
* 距上一个 keyframe 超过最大帧数

插入 keyframe 时：

1. 使用当前帧 stereo 重新初始化 points。
2. 创建 keyframe。
3. 更新 local map / tracks。
4. 记录 keyframe 数量。

SVO 不要求完整后端，但至少要维护：

* keyframes list
* current reference keyframe
* sparse points for reference
* tracking stats
* `loop_closures = 0`，如果不实现 loop closure

---

## 2.9 SVO local mapping

优先复用现有结构：

* `KeyFrame`
* `MapPoint`
* `CovisibilityGraph`
* `LocalMappingConfig`

但不要为了复杂 backend 牺牲稳定性。最低要求：

* 有关键帧对象或轻量 dataclass。
* 有 sparse map points。
* 有 map point 生命周期：初始化、被 track、被 inlier 保留、关键帧重建。
* stats 中记录 `map_points` 或 `active_points`。

可以先不做 BA。若做 bounded local BA，必须保持稳定和小步改动。

---

## 2.10 SVO loop closure

SVO loop closure 不是硬性要求。当前最低要求：

```python
"loop_candidates": 0,
"loop_closures": 0
```

如果复用现有 loop detector 成本低，可以加入候选检测；否则不要强行实现。

论文中后续会说明：SVO-style baseline does not enable production-grade loop closure。

---

# Phase 3：统一 benchmark 集成

重点修改：

* `run_benchmark.py`
* `evaluate.py`
* `slam_profiles.py`
* `tests/test_reliability.py`

---

## 3.1 run_benchmark.py 支持 ORB / DSO / SVO

`run_benchmark.py` 必须支持同时运行三算法：

```text
ORB-SLAM2-inspired stereo SLAM
DSO-inspired direct sparse stereo SLAM
SVO-style semi-direct stereo visual odometry/SLAM
```

新增 import：

```python
from svo_slam import SVOStyleSLAM
```

运行顺序建议：

1. ORB
2. DSO
3. SVO

每个算法都使用新的 `LimitedLoader(base_loader, args.max_frames)`，避免 iterator 被消耗。

输出文件：

```text
orb_slam_seq00.txt
dso_slam_seq00.txt
svo_slam_seq00.txt
benchmark_seq00.json
benchmark_trajectory_seq00.png
dso_diagnostics_seq00.json
dso_diagnostics_seq00.png
```

trajectory plot 必须包含：

* Ground truth
* ORB
* DSO
* SVO

`plot_trajectories()` 调用应类似：

```python
plot_trajectories(
    gt,
    {
        "ORB-SLAM": orb_array,
        "DSO-SLAM": dso_array,
        "SVO-SLAM": svo_array,
    },
    plot_path,
    alignment=args.alignment,
)
```

---

## 3.2 benchmark JSON 结构

`benchmark_seq00.json` 至少包含：

```python
{
    "sequence": "00",
    "max_frames": 100,
    "alignment": "origin",
    "implementation_manifest": implementation_manifest(),
    "orb": {...},
    "dso": {...},
    "svo": {...}
}
```

每个 algorithm summary 至少包含：

```python
{
    "ate": ...,
    "rpe": ...,
    "kitti_segment_metrics": ...,
    "runtime": {
        "avg_ms": ...,
        "median_ms": ...,
        "p95_ms": ...
    },
    "wall_time_s": ...,
    "keyframes": ...,
    "robustness": ...
}
```

不要改变已有 `summarize_method()` 的核心评估协议，除非必须修 bug。alignment 默认继续使用 `origin`。

轨迹长度检查必须扩展为三算法：

```python
if len(orb_array) != expected_len or len(dso_array) != expected_len or len(svo_array) != expected_len:
    raise ValueError(...)
```

不能静默截断。

---

## 3.3 CLI 参数

保留现有参数：

```bash
--data-dir
--seq
--output-dir
--max-frames
--features
--points
--alignment
```

新增参数建议：

```bash
--algorithms orb,dso,svo
--svo-points 1500
--dso-mode full
--dso-grid-selection / --dso-no-grid-selection
--dso-outlier-culling / --dso-no-outlier-culling
--dso-motion-gate / --dso-no-motion-gate
```

如果为了简洁，也可以让 `--points` 同时控制 DSO active points 和 SVO points，但 JSON 中必须记录实际参数。

---

# Phase 4：slam_profiles.py manifest 更新

在 `slam_profiles.py` 中新增 SVO profile：

```python
SVO_PROFILE = AlgorithmProfile(
    name="SVO-style semi-direct stereo visual odometry/SLAM",
    paper_reference="Forster et al., SVO: Fast Semi-Direct Monocular Visual Odometry, ICRA 2014",
    implementation_level="research_baseline_not_full_paper_reproduction",
    completed_components=(
        "grid-uniform sparse point selection",
        "stereo depth initialization",
        "patch/optical-flow based semi-direct tracking",
        "3D-2D PnP RANSAC pose estimation",
        "motion-gated pose acceptance",
        "keyframe insertion",
        "sparse local map maintenance",
    ),
    missing_paper_components=(
        "probabilistic depth filter",
        "full SVO backend",
        "production-grade relocalization",
        "mature map-point lifecycle management",
        "full loop closure",
        "C++ reference-level runtime optimization",
    ),
)
```

`implementation_manifest()` 必须返回：

```python
{
    "paper_level_claim": False,
    "interpretation": "...",
    "algorithms": {
        "orb": ...,
        "dso": ...,
        "svo": ...
    }
}
```

`interpretation` 必须改为包含三算法，例如：

```text
This repository provides reproducible Python research baselines inspired by ORB-SLAM2, DSO, and SVO. It is not a complete paper-level reimplementation of any official system.
```

禁止把 `paper_level_claim` 改成 `True`。

---

# Phase 5：测试补齐

修改 `tests/test_reliability.py`，至少新增以下测试。

---

## 5.1 SVO short sequence returns one pose per frame

```python
@unittest.skipUnless(HAS_KITTI_00, "KITTI sequence 00 sample is not available")
def test_svo_slam_short_sequence_has_one_pose_per_frame(self) -> None:
    base_loader = KITTIOdometryLoader(DATA_DIR, "00")
    slam = SVOStyleSLAM(LimitedLoader(base_loader, 5), num_points=300)
    trajectory, runtimes = slam.run()
    self.assertEqual(len(trajectory), 5)
    self.assertGreaterEqual(len(runtimes), 4)
    stats = slam.get_stats()
    self.assertEqual(stats["frames_processed"], 5)
    self.assertIn("tracking_failures", stats)
    self.assertIn("loop_closures", stats)
```

---

## 5.2 SVO short sequence moves forward on KITTI 00 sample

可以合并到上一个测试中：

```python
self.assertGreater(trajectory[-1][2, 3], 0.5)
```

如果 SVO 坐标 convention 与 ORB/DSO 不完全一致，应使用更稳健的位移范数：

```python
translation_norm = np.linalg.norm(trajectory[-1][:3, 3] - trajectory[0][:3, 3])
self.assertGreater(translation_norm, 0.5)
```

---

## 5.3 benchmark JSON includes SVO

可做轻量单元测试，不一定完整跑 100 帧。若测试环境有 KITTI，跑 3 到 5 帧即可。

要求确认：

* benchmark JSON 中包含 `"svo"`
* benchmark JSON 中包含 `"implementation_manifest"`
* manifest algorithms 中包含 `"svo"`

---

## 5.4 manifest includes SVO and does not overclaim

扩展现有测试：

```python
def test_manifest_does_not_overclaim_paper_level(self) -> None:
    manifest = implementation_manifest()
    self.assertFalse(manifest["paper_level_claim"])
    self.assertIn("orb", manifest["algorithms"])
    self.assertIn("dso", manifest["algorithms"])
    self.assertIn("svo", manifest["algorithms"])
    self.assertEqual(
        manifest["algorithms"]["svo"]["name"],
        "SVO-style semi-direct stereo visual odometry/SLAM"
    )
    self.assertIn("missing_paper_components", manifest["algorithms"]["svo"])
```

---

## 5.5 trajectory mismatch still raises

保留并不要削弱当前测试：

```python
def test_length_mismatch_is_not_silently_truncated(self) -> None:
    ...
```

---

## 5.6 json_safe still converts NaN to null

保留并扩展：

```python
safe = json_safe({
    "nan": float("nan"),
    "inf": float("inf"),
    "nested": [float("-inf")]
})
self.assertIsNone(safe["nan"])
self.assertIsNone(safe["inf"])
self.assertIsNone(safe["nested"][0])
```

---

## 5.7 DSO diagnostics generated

新增测试可以不写文件，直接运行短序列检查：

```python
@unittest.skipUnless(HAS_KITTI_00, "KITTI sequence 00 sample is not available")
def test_dso_diagnostics_has_one_record_per_frame(self) -> None:
    base_loader = KITTIOdometryLoader(DATA_DIR, "00")
    slam = DSOSLAM(LimitedLoader(base_loader, 3), num_active_points=300, pyramid_levels=2)
    trajectory, _ = slam.run()
    diagnostics = slam.get_diagnostics()
    self.assertEqual(len(diagnostics), len(trajectory))
    self.assertIn("valid_projected_points", diagnostics[-1])
    self.assertIn("fallback_used", diagnostics[-1])
```

---

# Phase 6：README 和 PAPER_LEVEL_STATUS 更新

更新：

* `README.md`
* `PAPER_LEVEL_STATUS.md`

必须说明：

1. 本项目是 reproducible Python research baseline。
2. 不是 ORB-SLAM2 / DSO / SVO 的官方 C++ 复现。
3. 新增第三算法：

   * `SVO-style semi-direct stereo visual odometry/SLAM`
4. DSO 新增：

   * diagnostics
   * motion gate
   * fallback
   * ablation
   * grid-uniform point selection
   * outlier culling
5. 统一 benchmark 命令。
6. 输出文件说明。
7. `implementation_manifest.paper_level_claim=false` 的含义。

不要写论文结论，不要写未经实际运行验证的数值。

---

# Phase 7：实验运行命令

完成代码和测试后，运行以下命令。

## 7.1 Unit tests

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

Linux/macOS：

```bash
python -B -m unittest discover -s tests
```

---

## 7.2 Tri-SLAM benchmark：20 frames

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -B run_benchmark.py --seq 00 --max-frames 20 --features 800 --points 800 --output-dir results\tri_slam_20
```

Linux/macOS：

```bash
python -B run_benchmark.py --seq 00 --max-frames 20 --features 800 --points 800 --output-dir results/tri_slam_20
```

---

## 7.3 Tri-SLAM benchmark：100 frames

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -B run_benchmark.py --seq 00 --max-frames 100 --features 1500 --points 1500 --output-dir results\tri_slam_100
```

Linux/macOS：

```bash
python -B run_benchmark.py --seq 00 --max-frames 100 --features 1500 --points 1500 --output-dir results/tri_slam_100
```

---

## 7.4 Tri-SLAM benchmark：300 frames

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -B run_benchmark.py --seq 00 --max-frames 300 --features 1500 --points 1500 --output-dir results\tri_slam_300
```

Linux/macOS：

```bash
python -B run_benchmark.py --seq 00 --max-frames 300 --features 1500 --points 1500 --output-dir results/tri_slam_300
```

如果 300 帧因数据、时间或环境问题无法完成，必须说明原因，并至少完成 20 和 100 帧。

---

## 7.5 DSO ablation：100 frames

如果实现为 `dso_ablation.py`：

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -B dso_ablation.py --seq 00 --max-frames 100 --points 1500 --output-dir results\dso_ablation_100
```

Linux/macOS：

```bash
python -B dso_ablation.py --seq 00 --max-frames 100 --points 1500 --output-dir results/dso_ablation_100
```

如果通过 `run_benchmark.py` 参数实现，则至少运行以下配置并保存到不同目录：

```powershell
.\.venv\Scripts\python.exe -B run_benchmark.py --seq 00 --max-frames 100 --features 1500 --points 1500 --dso-mode full --output-dir results\dso_ablation_100\full

.\.venv\Scripts\python.exe -B run_benchmark.py --seq 00 --max-frames 100 --features 1500 --points 1500 --dso-mode lk_pnp_only --output-dir results\dso_ablation_100\lk_pnp_only

.\.venv\Scripts\python.exe -B run_benchmark.py --seq 00 --max-frames 100 --features 1500 --points 1500 --dso-mode photometric_gate --output-dir results\dso_ablation_100\photometric_gate
```

---

# Phase 8：输出文件验收清单

完成后，至少应存在以下文件：

```text
results/tri_slam_20/
  benchmark_seq00.json
  benchmark_trajectory_seq00.png
  orb_slam_seq00.txt
  dso_slam_seq00.txt
  svo_slam_seq00.txt
  dso_diagnostics_seq00.json
  dso_diagnostics_seq00.png

results/tri_slam_100/
  benchmark_seq00.json
  benchmark_trajectory_seq00.png
  orb_slam_seq00.txt
  dso_slam_seq00.txt
  svo_slam_seq00.txt
  dso_diagnostics_seq00.json
  dso_diagnostics_seq00.png

results/tri_slam_300/
  benchmark_seq00.json
  benchmark_trajectory_seq00.png
  orb_slam_seq00.txt
  dso_slam_seq00.txt
  svo_slam_seq00.txt
  dso_diagnostics_seq00.json
  dso_diagnostics_seq00.png

results/dso_ablation_100/
  dso_ablation_seq00.json
  dso_ablation_seq00.csv
  dso_ablation_seq00.png
```

如果某个文件未生成，必须说明原因。

---

# Phase 9：Codex 最终汇报格式

完成后，请按以下格式汇报，不要写论文。

## 1. 修改摘要

列出修改过的文件：

```text
Modified:
- dso_slam.py
- dso_advanced.py
- run_benchmark.py
- evaluate.py
- slam_profiles.py
- tests/test_reliability.py
- README.md
- PAPER_LEVEL_STATUS.md

Added:
- svo_slam.py
- dso_ablation.py
```

实际以你的修改为准。

---

## 2. 关键实现说明

分别说明：

### DSO

* diagnostics 增加了哪些字段。
* motion gate 如何定义。
* fallback 如何工作。
* grid-uniform point selection 如何实现。
* outlier culling 做了哪些过滤。
* ablation 支持哪些配置。

### SVO-style

* 点选择如何做。
* stereo depth 如何初始化。
* patch / optical-flow tracking 如何做。
* PnP RANSAC 如何做。
* motion gate 和 fallback 如何做。
* keyframe 策略如何做。
* 是否启用 loop closure。

---

## 3. 测试结果

给出命令和结果，例如：

```text
Command:
.\.venv\Scripts\python.exe -B -m unittest discover -s tests

Result:
OK, 14 tests passed.
```

如果失败，必须列出失败项和原因。

---

## 4. Benchmark 结果表

至少给出 20 帧和 100 帧结果。300 帧如果完成也给出。

表格格式：

```text
Sequence 00, max_frames=100, alignment=origin

Algorithm | ATE RMSE (m) | RPE trans | RPE rot | Runtime avg/median/p95 (ms) | Wall time (s) | Keyframes | Tracking failures | Fallbacks | Loop closures
ORB       | ...
DSO       | ...
SVO       | ...
```

数值必须来自 `benchmark_seq00.json`，不能手填猜测。

---

## 5. DSO ablation 结果表

```text
DSO ablation, Sequence 00, max_frames=100, alignment=origin

Config | ATE RMSE (m) | Tracking failures | Fallbacks | Motion gate rejections | Mean valid points | Mean inlier ratio | Runtime avg (ms)
full_simplified_dso | ...
lk_pnp_only | ...
photometric_with_motion_gate | ...
grid_uniform_selection | ...
outlier_culling_on | ...
outlier_culling_off | ...
```

如果某个配置无法完成，必须说明原因。

---

## 6. 生成的结果文件

列出实际生成路径：

```text
- results/tri_slam_20/benchmark_seq00.json
- results/tri_slam_20/benchmark_trajectory_seq00.png
- results/tri_slam_20/dso_diagnostics_seq00.json
- results/tri_slam_20/dso_diagnostics_seq00.png
...
```

---

## 7. 剩余问题和建议

必须客观说明：

* DSO 是否仍有明显 drift。
* SVO 是否稳定。
* 是否有 tracking failures。
* 是否有某些指标异常。
* 300 帧是否完成。
* 是否仍有需要 GPT Pro 在论文中谨慎解释的限制。

不要把 DSO 差解释成 “direct methods are worse than feature-based methods”。正确表述应是：

```text
The simplified DSO-inspired Python baseline remains sensitive to photometric assumptions, initialization quality, and incomplete joint photometric optimization. The observed drift should be interpreted as an implementation-level limitation rather than a failure of the original DSO method.
```

---

# 最终验收标准

Codex 任务完成必须满足：

1. 所有 unit tests 通过。
2. `run_benchmark.py` 至少能在 KITTI 00 的 20 帧和 100 帧上完成 ORB、DSO、SVO 三算法运行。
3. 三个算法轨迹长度与 ground truth 对齐。
4. `benchmark_seq00.json` 中包含 `"orb"`、`"dso"`、`"svo"`。
5. `benchmark_trajectory_seq00.png` 中包含 ground truth、ORB、DSO、SVO。
6. SVO 不能只是复制 ORB descriptor matching，必须有 patch / optical-flow / semi-direct tracking 成分。
7. DSO 必须有 diagnostics 输出和至少一个 ablation 输出。
8. `README.md` 或 `PAPER_LEVEL_STATUS.md` 已更新第三算法和实验命令。
9. 不得删除现有 ORB/DSO 功能。
10. `implementation_manifest.paper_level_claim` 必须保持 `false`。
11. JSON 输出必须 `allow_nan=False` 可成功写入。
12. 轨迹长度 mismatch 必须继续报错，不能静默截断。

---

完成上述任务后，把实际 benchmark JSON、trajectory plot、DSO diagnostics、ablation 结果和测试输出交回。GPT Pro 将只基于这些真实结果进入论文阶段；若结果不完整，会先继续补齐工程和实验，不会编造论文数值。
