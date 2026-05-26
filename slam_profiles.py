from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AlgorithmProfile:
    name: str
    paper_reference: str
    implementation_level: str
    completed_components: tuple[str, ...]
    missing_paper_components: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


ORB_SLAM2_PROFILE = AlgorithmProfile(
    name="ORB-SLAM2-inspired stereo SLAM",
    paper_reference="Mur-Artal and Tardos, ORB-SLAM2, IEEE T-RO 2017",
    implementation_level="research_baseline_not_full_paper_reproduction",
    completed_components=(
        "stereo ORB feature extraction",
        "stereo depth from left-right matches",
        "frame-to-frame PnP RANSAC tracking",
        "keyframe insertion",
        "cross-keyframe map-point association",
        "bounded local bundle adjustment",
        "local map tracking against projected map points",
        "BoW/PnP relocalization",
        "online visual vocabulary for loop candidates",
        "essential-matrix loop verification",
        "Sim3 map-point loop fusion",
        "pose graph optimization",
        "bounded global bundle adjustment after loop closure",
    ),
    missing_paper_components=(
        "production ORB vocabulary trained offline at scale",
        "full covisibility spanning-tree and essential-graph policy",
        "full ORB-SLAM2 relocalization scoring and recovery policy",
        "full Sim3 loop fusion with robust duplicate map-point replacement",
        "unbounded global bundle adjustment comparable to the C++ reference system",
    ),
)


DSO_PROFILE = AlgorithmProfile(
    name="DSO-inspired direct sparse stereo SLAM",
    paper_reference="Engel et al., Direct Sparse Odometry, IEEE TPAMI 2018",
    implementation_level="research_baseline_not_full_paper_reproduction",
    completed_components=(
        "high-gradient active pixel selection",
        "stereo SGBM depth initialization",
        "coarse-to-fine photometric tracking",
        "LK+PnP initialization for direct alignment",
        "Huber-robust photometric residuals",
        "affine brightness transfer in direct tracking",
        "residual p95, cost-jump, projection-ratio, and LK-consistency tracking gates",
        "left-right stereo consistency depth filtering",
        "inverse-depth active window",
        "bounded active-window photometric BA over poses, affine brightness, and inverse-depth points",
        "active point lifecycle culling",
        "lightweight marginalization pose prior",
        "response/vignetting-style photometric pre-calibration",
        "optional CLAHE and gradient-normalized residual ablations",
        "bounded photometric local refinement",
        "strict loop-candidate verification",
        "candidate-only loop accounting unless pose graph correction is actually applied",
        "fallback keyframe refresh and lightweight reinitialization",
    ),
    missing_paper_components=(
        "Schur complement marginalization priors matching DSO",
        "dataset-calibrated exposure/response/vignetting files",
        "full DSO point activation and marginalization policy",
        "unbounded production photometric bundle adjustment comparable to the C++ reference system",
        "production-grade loop closure for direct sparse maps",
    ),
)


SVO_PROFILE = AlgorithmProfile(
    name="SVO-style semi-direct stereo visual odometry/SLAM",
    paper_reference="Forster et al., SVO: Fast Semi-Direct Monocular Visual Odometry, ICRA 2014",
    implementation_level="research_baseline_not_full_paper_reproduction",
    completed_components=(
        "stereo SGBM depth initialization",
        "grid-uniform high-gradient sparse point selection",
        "patch/optical-flow based semi-direct tracking",
        "pyramidal LK patch tracking",
        "PnP RANSAC pose estimation",
        "motion gate and constant-velocity fallback",
        "keyframe insertion",
        "bounded sparse map maintenance",
        "unified KITTI benchmark integration",
    ),
    missing_paper_components=(
        "full SVO probabilistic depth-filter update",
        "direct image alignment of sparse patches before feature alignment",
        "mature map point life-cycle and uncertainty propagation",
        "loop closure and pose graph backend",
        "production-grade initialization and recovery policies",
    ),
)


def implementation_manifest() -> dict:
    return {
        "paper_level_claim": False,
        "interpretation": (
            "This repository provides reproducible research baselines inspired by "
            "ORB-SLAM2, DSO, and SVO. It is not a complete paper-level reimplementation."
        ),
        "algorithms": {
            "orb": ORB_SLAM2_PROFILE.to_dict(),
            "dso": DSO_PROFILE.to_dict(),
            "svo": SVO_PROFILE.to_dict(),
        },
    }
