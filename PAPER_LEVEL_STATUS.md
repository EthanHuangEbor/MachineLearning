# 论文级复现状态

本仓库当前提供的是 ORB-SLAM2-inspired、DSO-inspired 和 SVO-style semi-direct stereo visual odometry / SLAM 的 Python 实验 baseline，也就是 **reproducible Python research baselines**，不是三个原始系统的官方完整论文级复现。

所有 benchmark JSON 都应包含：

```json
{
  "implementation_manifest": {
    "paper_level_claim": false
  }
}
```

## 已补齐的工程基础

- 统一 KITTI stereo 数据加载、标定读取、真值读取和三算法 benchmark。
- 统一输出 ATE、RPE、KITTI segment metrics、轨迹图和 JSON summary。
- JSON 输出通过 `json_safe` 处理 NaN / inf，保证机器可读。
- ORB、DSO、SVO 在同一数据、同一帧数、同一评估协议下对比。
- `run_benchmark.py` 是三算法主入口，`dso_ablation.py` 是 DSO 消融入口。

## ORB-SLAM2-inspired baseline

当前已支持：

- stereo ORB feature extraction
- stereo depth from left-right matches
- frame-to-frame PnP RANSAC tracking
- keyframe insertion
- cross-keyframe map point association
- bounded local bundle adjustment
- local map tracking against projected map points
- BoW/PnP relocalization
- online visual vocabulary for loop candidates
- essential-matrix loop verification
- Sim3 map-point loop fusion
- pose graph optimization
- bounded global bundle adjustment after loop closure

仍不是官方 ORB-SLAM2 的原因：

- 没有大规模离线训练 ORB vocabulary。
- covisibility spanning tree、essential graph 和关键帧策略仍是简化 baseline。
- relocalization、Sim3 fusion、map point replacement 和 global BA 不是官方 C++ 系统的完整实现。
- 当前目标是统一 benchmark 中的特征法 baseline，而不是替代 ORB-SLAM2 官方实现。

## DSO-inspired baseline

当前已支持：

- high-gradient active pixel selection
- stereo SGBM depth initialization
- left-right stereo consistency depth filtering
- coarse-to-fine photometric tracking
- LK+PnP initialization for direct alignment
- Huber-robust photometric residuals
- affine brightness gain / bias
- residual p95、cost jump、projection ratio、LK consistency tracking gates
- inverse-depth active window
- bounded active-window photometric BA over poses, affine brightness, and inverse-depth points
- active point lifecycle culling
- lightweight marginalization pose prior
- response/vignetting-style photometric pre-calibration
- optional CLAHE and gradient-normalized residual ablations
- fallback keyframe refresh and lightweight reinitialization
- strict loop candidate / verified / correction accounting

仍不是官方 DSO 的原因：

- marginalization prior 是轻量 pose prior，不是完整 Schur complement marginalization。
- 光度标定是 response/vignetting-style 工程接口，不是由数据集曝光、响应和 vignette 标定文件完整驱动。
- active point activation、marginalization、immature point handling 和稀疏 Hessian 后端仍是 bounded Python baseline。
- 当前 BA 为 bounded sliding-window photometric BA，目标是论文可信和可运行，不是复现官方 DSO C++ 性能。
- loop closure 是严格诊断和候选验证逻辑，不声称等价于成熟 direct sparse map loop backend。

## SVO-style baseline

当前已支持：

- stereo SGBM depth initialization
- grid-uniform high-gradient sparse point selection
- patch / optical-flow based semi-direct tracking
- pyramidal LK patch tracking
- PnP RANSAC pose estimation
- motion gate and constant-velocity fallback
- keyframe insertion
- bounded sparse map maintenance
- unified KITTI benchmark integration

仍不是官方 SVO 的原因：

- 没有完整 probabilistic depth-filter update。
- 没有完整 sparse patch direct alignment -> depth uncertainty propagation -> mature map point lifecycle。
- 没有成熟 relocalization、loop closure 和 pose graph backend。
- 当前重点是提供第三条半直接路线，用于和 ORB/DSO 在同一协议下对比。

## 使用建议

本仓库适合：

- Python 视觉 SLAM 教学和实验。
- ORB / DSO / SVO 风格路线的统一协议对比。
- 消融实验、鲁棒性统计和可解释 baseline。
- 论文中讨论 bounded research baseline 的工程改进。

不适合直接声称：

- “完整复现 ORB-SLAM2 / DSO / SVO 论文”
- “达到官方实现精度和鲁棒性”
- “DSO 实现具有完整 Schur complement marginalization 后端”
- “当前结果可以替代官方 C++ 系统”

建议论文表述：

> We evaluate reproducible Python research baselines inspired by ORB-SLAM2, DSO, and SVO under a unified KITTI stereo protocol. These implementations are not claimed to be complete paper-level reproductions of the original systems.

