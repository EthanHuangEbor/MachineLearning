# KITTI 视觉 SLAM 算法复现与对比

基于 KITTI Odometry 数据集，复现两种视觉 SLAM 算法并进行对比分析。

## 项目概述

**研究任务**：在 KITTI Odometry 数据集上复现 ORB-SLAM2 与 DSO 两种视觉定位/建图算法，从轨迹精度、运行效率和鲁棒性三个方面进行对比分析。

**技术路线**：
- **ORB-SLAM2**（特征点法）— 基于 ORB 特征提取 + PnP 姿态估计 + 局部 BA + 回环检测
- **DSO**（直接法）— 基于高梯度像素光度误差 + LM 多层优化 + 局部光度 BA + 回环检测

两种方法在前端跟踪策略上不同（特征匹配 vs 直接光度对齐），后端和回环使用相同基础设施，保证公平对比。

---

## 目录结构

```
2-MashineLearning/
├── slam_base.py          # 共用基础设施：KeyFrame, MapPoint, CovisibilityGraph, BA求解器
├── local_mapping.py      # 局部建图：地图点创建、局部BA、地图点筛选
├── loop_detector.py      # 回环检测：词袋(BoW) + 几何校验 + Sim3
│
├── kitti_utils.py        # KITTI数据集加载、标定、深度转换
├── orb_vo.py            # ORB特征点法视觉里程计（前端，仅供参考对比）
├── direct_vo.py         # DSO风格直接法里程计（前端，仅供参考对比）
│
├── orb_slam.py          # 完整ORB-SLAM2系统（前端+后端+回环）
├── dso_slam.py          # 完整DSO-SLAM系统（前端+后端+回环）
│
├── evaluate.py          # 轨迹评估：ATE RMSE、RPE、轨迹可视化
│
├── requirements.txt     # Python依赖
├── AI.md                # 项目详细规划文档（中文）
└── README.md
```

---

## 算法原理

### ORB-SLAM2（特征点法）

**核心思想**：提取 ORB 特征点 → 双目匹配三角化 → PnP+RANSAC 估计相机位姿

**系统组成**：
1. **Tracking**：ORB 特征检测 → 左右目匹配 → 3D 点三角化 → PnP 姿态估计 → 关键帧判断
2. **Local Mapping**：新地图点三角化 → 局部 Bundle Adjustment → 地图点筛选
3. **Loop Closing**：BoW 词袋匹配 → 几何校验（RANSAC+本质矩阵）→ Pose Graph 优化

### DSO（直接稀疏里程计）

**核心思想**：不提取特征，直接利用高梯度像素的光度误差优化相机位姿

**系统组成**：
1. **Tracking**：高梯度像素选择 → 双目深度估计 → 光度误差构建 → 多层金字塔 LM 优化
2. **Local Mapping**：光度 BA（EnergyFunctional）→ 关键帧管理 → 共视图更新
3. **Loop Closing**：ORB 特征匹配找回环候选 → 几何校验 → Pose Graph 优化

### 关键对比

| 维度 | ORB-SLAM2 | DSO |
|------|-----------|-----|
| 前端方法 | 特征点 + PnP | 直接光度对齐 |
| 误差类型 | 重投影几何误差 | 光度误差 |
| 优化变量 | 稀疏特征点 | 稀疏像素点 |
| 光照敏感性 | 中等 | 高 |
| 纹理要求 | 需要角点/边缘 | 需要梯度信息 |

---

## 数据准备

### 下载地址

KITTI Odometry Benchmark：https://www.cvlibs.net/datasets/kitti/eval_odometry.php

**需要下载**：
1. **odometry data set (grayscale)** — 22 GB（双目灰度图像序列）
2. **ground truth poses** — 4 MB（序列 00-10 的真值轨迹）
3. **calibration files** — 1 MB（相机标定参数）

### 目录结构

下载解压后按以下结构放置：

```
data/
└── kitti_odometry/
    ├── sequences/
    │   └── 00/
    │       ├── image_0/       ← 双目左图（000000.png ~ NNNNNN.png）
    │       ├── image_1/       ← 双目右图
    │       ├── calib.txt      ← 相机标定
    │       └── times.txt      ← 时间戳
    └── poses/
        └── 00.txt             ← 真值轨迹
```

**验证数据是否就绪**：
```bash
ls data/kitti_odometry/sequences/00/image_0/ | head -3
ls data/kitti_odometry/poses/00.txt
```

---

## 运行方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行完整系统对比

**1. 运行 ORB-SLAM2**：
```bash
python orb_slam.py --seq 00 --output results/orb_slam_seq00.txt
```

**2. 运行 DSO-SLAM**：
```bash
python dso_slam.py --seq 00 --output results/dso_slam_seq00.txt
```

**3. 评估对比**：
```bash
python evaluate.py --seq 00
```

**可选参数**：
```bash
--seq        # KITTI序列号，默认00
--data-dir   # 数据根目录，默认data/kitti_odometry
--output     # 输出轨迹路径
--features   # ORB特征数量，默认1500
--points     # DSO活跃像素数量，默认1500
```

### 运行旧版前端对比（仅前端里程计，无后端回环）

```bash
python orb_vo.py --seq 00
python direct_vo.py --seq 00
python evaluate.py --seq 00
```

### 输出结果

运行后在 `results/` 目录下生成：
- `orb_slam_seq00.txt` / `dso_slam_seq00.txt` — 轨迹文件（KITTI 格式）
- `summary_seq00.json` — ATE/RPE 误差统计
- `trajectory_seq00.png` — 轨迹可视化对比图

---

## 评价指标

| 指标 | 全称 | 说明 |
|------|------|------|
| **ATE** | Absolute Trajectory Error | 绝对轨迹误差，RMSE（米） |
| **RPE_trans** | Relative Pose Error (translation) | 相对平移误差，百分比 |
| **RPE_rot** | Relative Pose Error (rotation) | 相对旋转误差，度/米 |
| **Runtime** | 平均每帧处理时间 | 毫秒/帧 |

---

## 依赖环境

```
pykitti          # KITTI数据加载
evo              # 轨迹评估工具
opencv-python    # 图像处理、特征提取
numpy            # 数值计算
scipy             # 非线性优化（BA求解）
matplotlib       # 可视化
open3d           # 3D点云可视化
scikit-learn     # 词袋聚类
```

---

## 参考论文

- **ORB-SLAM2**: Mur-Artal, Raul, and Juan D. Tardós. "ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras." IEEE Transactions on Robotics, 2017.
- **DSO**: Engel, Jakob, et al. "Direct Sparse Odometry." IEEE TPAMI, 2018.

---

## 报告结构建议

1. **引言** — SLAM 背景、自动驾驶应用
2. **理论基础** — SLAM 原理、前端/后端/回环/建图各模块
3. **算法介绍** — ORB-SLAM2 和 DSO 分别详述
4. **实验设置** — 硬件、软件、数据集、参数
5. **结果分析** — 轨迹精度、运行时间、鲁棒性表格/图
6. **总结** — 两种算法优缺点、适用场景
