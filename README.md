# KITTI ORB / DSO / SVO Python SLAM Baselines

长期下载链接：
- 北航云盘：
https://bhpan.buaa.edu.cn/link/AA970A42705B6547778B7B345B92AEADA8
文件夹名：MashineLearning
- Github：
https://github.com/EthanHuangEbor/MachineLearning.git


本仓库是在 KITTI Odometry stereo 数据集上运行的 Python 视觉 SLAM / VO 研究基线。当前统一比较三条路线：

- ORB-SLAM2-inspired stereo SLAM
- DSO-inspired direct sparse stereo SLAM
- SVO-style semi-direct stereo visual odometry / SLAM

重要边界：这里提供的是 **reproducible Python research baselines**，不是 ORB-SLAM2、DSO 或 SVO 官方 C++ 系统的完整论文级复现。所有 benchmark JSON 都通过 `implementation_manifest.paper_level_claim=false` 明确声明这一点。

## 当前状态

当前主入口是：

| 文件 | 作用 |
| --- | --- |
| `run_benchmark.py` | ORB / DSO / SVO 三算法统一 benchmark，输出 JSON、轨迹、诊断图。 |
| `dso_ablation.py` | DSO-only 消融实验入口。 |
| `slam_comparison.ipynb` | 读取最新结果并生成交互式分析表格和图像展示。 |
| `evaluate.py` | ATE、RPE、KITTI segment metrics、轨迹绘图、JSON 安全转换。 |
| `kitti_utils.py` | KITTI stereo 数据、标定、真值轨迹读取与 KITTI 格式轨迹保存。 |
| `slam_profiles.py` | 三算法实现边界和 `paper_level_claim=false` manifest。 |

旧入口 `orb_vo.py` 和 `direct_vo.py` 已不存在，也不需要恢复。

## 算法文件

| 文件 | 当前用途 | 是否保留 |
| --- | --- | --- |
| `orb_slam.py` | ORB-SLAM2-inspired 主实现：特征跟踪、PnP、关键帧、局部建图、回环。 | 保留 |
| `orb_advanced.py` | ORB local map tracking、relocalization、Sim3 fusion、global BA。 | 保留 |
| `dso_slam.py` | DSO-inspired 主实现：光度 tracking、质量门控、fallback、active-window BA、loop diagnostics。 | 保留 |
| `dso_advanced.py` | DSO 光度校正、inverse-depth active window、轻量 prior。 | 保留 |
| `svo_slam.py` | SVO-style 半直接 stereo baseline。 | 保留 |
| `slam_base.py` | KeyFrame、MapPoint、covisibility graph、local BA、pose graph。 | 保留 |
| `local_mapping.py` | ORB 共享局部建图后端。 | 保留 |
| `loop_detector.py` | BoW-style loop candidate、Sim3 求解工具。 | 保留 |

根目录的算法 `.py` 文件当前都仍然有实际用途；不建议删除。可以清理的是 `__pycache__/`、工具私有目录、以及旧实验输出。

## DSO 当前增强

当前 DSO 仍是 bounded DSO-inspired baseline，但已从早期简化版升级为更可信的工程版本：

- left-right SGBM consistency depth filtering
- grid active point selection with spatial diversity
- affine brightness gain / bias
- residual p95、cost jump、valid projection ratio、LK consistency tracking gates
- fallback 顺序：LK+PnP -> constant velocity -> last safe pose
- fallback 后强制关键帧刷新，连续失败后 lightweight reinitialization
- bounded active-window photometric BA，联合优化 pose、affine brightness、inverse-depth points
- active point age / observation / bad-count / residual lifecycle culling
- loop candidates、verified loops、applied corrections 分开统计
- `loop_closures` 只在几何验证通过且 pose graph correction 实际应用后增加

这修正了早期 DSO 把候选回环直接记成真实闭环的问题。KITTI 00 前 300 帧的最新 DSO 结果中，`loop_candidates=17`，但 `loop_closures=0`，更符合短序列实验语义。

## 数据目录

默认读取 KITTI Odometry 结构：

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

最新 300 帧三算法 benchmark：

```powershell
.\.venv\Scripts\python.exe -B run_benchmark.py --data-dir data/kitti_odometry --seq 00 --max-frames 300 --algorithms orb,dso,svo --features 800 --points 800 --svo-points 1500 --output-dir results\final_seq00_300_optimized_f800_p800
```

最新 300 帧 DSO 消融：

```powershell
.\.venv\Scripts\python.exe -B dso_ablation.py --data-dir data/kitti_odometry --seq 00 --max-frames 300 --points 800 --output-dir results\dso_ablation_300_optimized_p800
```

单独重新绘图或计算指标时，优先使用 `run_benchmark.py` 和 `dso_ablation.py`，不要回退到早期 demo 入口。

## 最新结果

当前最新主结果位于：

| 路径 | 作用 |
| --- | --- |
| `results/final_seq00_300_optimized_f800_p800/` | 最新 300 帧 ORB / DSO / SVO 三算法 benchmark。 |
| `results/dso_ablation_300_optimized_p800/` | 最新 300 帧 DSO 消融实验。 |
| `results/seq00_300_optimized_results.zip` | 上面两个结果目录和结果说明的压缩包。 |
| `results/README.md` | 结果目录说明、指标表和阅读顺序。 |

最新 300 帧主 benchmark 摘要：

| Algorithm | ATE RMSE m | RPE trans % | KITTI mean trans % | Failures | Fallbacks | Loop closures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORB | 2.9913 | 2.7487 | 2.6843 | 0 | n/a | 2 |
| DSO | 3.4279 | 2.0376 | 2.0259 | 12 | 12 | 0 |
| SVO | 2.9621 | 1.7526 | 1.7257 | 0 | 0 | 0 |

旧结果已归档到 `results/Old/`。带有 `smoke` 的早期临时运行结果不作为论文证据，已从 `results/` 中清理。

## 文档状态

| 文件 | 状态 |
| --- | --- |
| `README.md` | 当前主说明文档。 |
| `PAPER_LEVEL_STATUS.md` | 论文级复现边界说明。 |
| `results/README.md` | 实验结果目录说明。 |
| `slam_comparison.ipynb` | 最新结果分析 notebook。 |
| `results/Old/Goal.md` | 历史阶段任务书，仅归档，不作为当前使用说明。 |
| `00000_Report/` | LaTeX/PDF 报告资料；若用于最终论文，需要同步最新 300 帧结果。 |





