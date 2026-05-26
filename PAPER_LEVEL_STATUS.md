# 论文级复现状态

本仓库当前提供的是 ORB-SLAM2-inspired、DSO-inspired 和 SVO-style semi-direct stereo visual odometry/SLAM 的 Python 实验 baseline，也就是 **reproducible Python research baselines**，而不是三个原始系统的完整论文级复现。For paper wording, this is **not official ORB-SLAM2/DSO/SVO reproduction**. 所有 benchmark JSON 都会包含 `implementation_manifest.paper_level_claim=false`：

```json
{
  "implementation_manifest": {
    "paper_level_claim": false
  }
}
```

## 已补齐的工程基础

- 统一 KITTI stereo 数据加载、标定读取、真值读取和三算法 benchmark。
- 轨迹长度不一致默认报错，不再静默截断。
- ATE、RPE、KITTI segment metrics 和轨迹图统一输出。
- JSON 使用标准 JSON，NaN / inf 通过 `json_safe` 写为 `null`。
- ORB 侧包含 local map tracking、BoW/PnP relocalization、Sim3 loop fusion 和 bounded global BA。
- DSO 侧包含 inverse-depth active window、轻量 marginalization pose prior、光度预校正、motion gate、outlier culling、LK+PnP / constant-velocity fallback 和逐帧 diagnostics。
- SVO 侧包含双目深度初始化、grid 高梯度点、LK patch tracking、PnP、motion gate、fallback 和关键帧维护。
- DSO ablation 已有独立入口，可比较 LK+PnP、光度门控、grid selection 和 outlier culling。

## 仍不是论文级完整复现的原因

### ORB-SLAM2

- 缺少大规模离线训练 ORB vocabulary。
- covisibility spanning tree、essential graph 和关键帧策略仍是简化 baseline。
- relocalization、Sim3 fusion、map point replacement 和 global BA 不是官方 C++ 系统的完整实现。

### DSO

- active poses、inverse depths 和 affine brightness 尚未做完整联合优化。
- marginalization prior 是轻量 pose prior，不是完整 Schur complement prior。
- 光度标定是响应/暗角风格接口，不是数据集标定文件驱动的曝光、响应和 vignette 完整模型。
- 点激活、生命周期、边缘化策略和 photometric BA 仍是简化版本。

### SVO

- 当前实现是 SVO-style semi-direct stereo visual odometry/SLAM baseline，不是原始 SVO 的完整概率深度滤波系统。
- 未实现完整 map point uncertainty、深度滤波收敛策略和成熟重定位/回环后端。
- 当前重点是提供第三条半直接路线，用于和 ORB/DSO 在同一协议下对比。

## 使用建议

本仓库适合：

- Python 视觉 SLAM 教学和实验。
- ORB / DSO / SVO 风格路线的公平协议对比。
- 消融实验、鲁棒性统计和可解释 baseline。

不适合直接声称：

- “100% 复现 ORB-SLAM2 / DSO / SVO 论文”。
- “达到官方实现精度和鲁棒性”。
- “代码没有任何问题”。

论文或报告中建议表述为：`Python research baselines inspired by ORB-SLAM2, DSO, and SVO`。
