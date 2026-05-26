from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dso_slam import DSOTrackingConfig, DSOSLAM
from evaluate import json_safe, plot_trajectories, summarize_method
from kitti_utils import KITTIOdometryLoader, save_trajectory_kitti
from orb_slam import ORBSLAM2
from slam_profiles import implementation_manifest
from svo_slam import SVOStyleSLAM


class LimitedLoader:
    def __init__(self, base_loader: KITTIOdometryLoader, max_frames: int | None):
        self.base_loader = base_loader
        self.calibration = base_loader.calibration
        self.max_frames = max_frames

    def iter_frames(self):
        frames = self.base_loader.iter_frames()
        if self.max_frames is None:
            return frames
        return itertools.islice(frames, self.max_frames)


def runtime_summary(runtimes_ms: list[float]) -> dict[str, float]:
    if not runtimes_ms:
        return {"avg_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    values = np.asarray(runtimes_ms, dtype=np.float64)
    return {
        "avg_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def parse_algorithms(value: str) -> list[str]:
    algorithms = [part.strip().lower() for part in value.split(",") if part.strip()]
    valid = {"orb", "dso", "svo"}
    unknown = sorted(set(algorithms) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown algorithms: {', '.join(unknown)}")
    if not algorithms:
        raise argparse.ArgumentTypeError("At least one algorithm is required")
    return algorithms


def build_dso_config(args: argparse.Namespace) -> DSOTrackingConfig:
    config = DSOTrackingConfig()
    if args.dso_mode in {"full", "strict_full"}:
        pass
    elif args.dso_mode == "lk_pnp_only":
        config.enable_photometric_refinement = False
        config.enable_outlier_culling = False
    elif args.dso_mode in {"photometric_with_motion_gate", "photometric_gate"}:
        config.enable_photometric_refinement = True
        config.enable_motion_gate = True
        config.enable_grid_selection = False
        config.enable_outlier_culling = False
    elif args.dso_mode == "grid_uniform_selection":
        config.enable_grid_selection = True
        config.enable_outlier_culling = False
    elif args.dso_mode == "outlier_culling_on":
        config.enable_grid_selection = False
        config.enable_outlier_culling = True
    elif args.dso_mode == "outlier_culling_off":
        config.enable_grid_selection = False
        config.enable_outlier_culling = False
    elif args.dso_mode == "no_left_right_check":
        config.enable_left_right_depth_check = False
    elif args.dso_mode == "no_quality_gates":
        config.enable_residual_p95_gate = False
        config.enable_cost_jump_gate = False
        config.enable_lk_consistency_gate = False
        config.min_valid_projected_points = 0
        config.min_valid_projected_ratio = 0.0
        config.min_inlier_ratio = 0.0
    elif args.dso_mode == "no_joint_ba":
        config.enable_joint_window_ba = False
    elif args.dso_mode == "no_affine_brightness":
        config.enable_affine_brightness = False
    elif args.dso_mode == "loop_candidates_only":
        config.enable_loop_correction = False

    if args.dso_motion_gate is not None:
        config.enable_motion_gate = args.dso_motion_gate
    if args.dso_grid_selection is not None:
        config.enable_grid_selection = args.dso_grid_selection
    if args.dso_outlier_culling is not None:
        config.enable_outlier_culling = args.dso_outlier_culling
    if args.dso_left_right_depth_check is not None:
        config.enable_left_right_depth_check = args.dso_left_right_depth_check
    if args.dso_strict_loop_verification is not None:
        config.enable_strict_loop_verification = args.dso_strict_loop_verification
    if args.dso_loop_correction is not None:
        config.enable_loop_correction = args.dso_loop_correction
    if args.dso_residual_p95_gate is not None:
        config.enable_residual_p95_gate = args.dso_residual_p95_gate
    if args.dso_cost_jump_gate is not None:
        config.enable_cost_jump_gate = args.dso_cost_jump_gate
    if args.dso_lk_consistency_gate is not None:
        config.enable_lk_consistency_gate = args.dso_lk_consistency_gate
    if args.dso_joint_window_ba is not None:
        config.enable_joint_window_ba = args.dso_joint_window_ba
    if args.dso_affine_brightness is not None:
        config.enable_affine_brightness = args.dso_affine_brightness
    if args.dso_clahe is not None:
        config.enable_clahe = args.dso_clahe
    if args.dso_gradient_normalized_residual is not None:
        config.enable_gradient_normalized_residual = args.dso_gradient_normalized_residual

    if args.disable_dso_motion_gate:
        config.enable_motion_gate = False
    if args.disable_dso_grid_selection:
        config.enable_grid_selection = False
    if args.disable_dso_outlier_culling:
        config.enable_outlier_culling = False
    if args.disable_dso_photometric_refinement:
        config.enable_photometric_refinement = False
    if args.disable_dso_left_right_depth_check:
        config.enable_left_right_depth_check = False
    if args.disable_dso_quality_gates:
        config.enable_residual_p95_gate = False
        config.enable_cost_jump_gate = False
        config.enable_lk_consistency_gate = False
        config.min_valid_projected_ratio = 0.0

    if args.dso_max_residual_p95 is not None:
        config.max_residual_p95 = args.dso_max_residual_p95
    if args.dso_max_cost_jump_ratio is not None:
        config.max_cost_jump_ratio = args.dso_max_cost_jump_ratio
    if args.dso_ba_max_points_per_keyframe is not None:
        config.ba_max_points_per_keyframe = args.dso_ba_max_points_per_keyframe
    if args.dso_ba_max_nfev is not None:
        config.ba_max_nfev = args.dso_ba_max_nfev
    if args.dso_max_consecutive_failures is not None:
        config.max_consecutive_failures = args.dso_max_consecutive_failures
    return config


def write_diagnostics_json_csv(
    diagnostics: list[dict[str, object]],
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(diagnostics), f, indent=2, ensure_ascii=False, allow_nan=False)

    if not diagnostics:
        csv_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(diagnostics[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in json_safe(diagnostics):
            writer.writerow(row)


def plot_dso_diagnostics(diagnostics: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not diagnostics:
        output_path.write_bytes(b"")
        return

    frames = [int(item["frame_id"]) for item in diagnostics]
    def value_or_nan(item: dict[str, object], key: str) -> float:
        value = item.get(key)
        if value is None:
            return float("nan")
        return float(value)

    valid_points = [value_or_nan(item, "valid_projected_points") for item in diagnostics]
    inlier_ratio = [value_or_nan(item, "inlier_ratio") for item in diagnostics]
    failures = [1.0 if item["tracking_failure"] else 0.0 for item in diagnostics]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(frames, valid_points, label="valid projected points")
    axes[0].set_ylabel("points")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(frames, inlier_ratio, label="inlier ratio", color="tab:green")
    axes[1].set_ylabel("inlier")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[2].step(frames, failures, where="mid", label="tracking failure", color="tab:red")
    axes[2].set_ylabel("failure")
    axes[2].set_xlabel("frame")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("DSO Tracking Diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fair ORB/DSO/SVO KITTI benchmark")
    parser.add_argument("--data-dir", default="data/kitti_odometry")
    parser.add_argument("--seq", default="00")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--features", type=int, default=1500)
    parser.add_argument("--points", type=int, default=1500, help="DSO active point budget")
    parser.add_argument("--svo-points", type=int, default=1500)
    parser.add_argument("--alignment", choices=["origin", "se3", "sim3"], default="origin")
    parser.add_argument("--algorithms", type=parse_algorithms, default=parse_algorithms("orb,dso,svo"))
    parser.add_argument(
        "--dso-mode",
        choices=[
            "full",
            "strict_full",
            "lk_pnp_only",
            "photometric_with_motion_gate",
            "photometric_gate",
            "grid_uniform_selection",
            "outlier_culling_on",
            "outlier_culling_off",
            "no_left_right_check",
            "no_quality_gates",
            "no_joint_ba",
            "no_affine_brightness",
            "loop_candidates_only",
        ],
        default="full",
    )
    parser.add_argument("--dso-motion-gate", dest="dso_motion_gate", action="store_true", default=None)
    parser.add_argument("--dso-no-motion-gate", dest="dso_motion_gate", action="store_false")
    parser.add_argument("--dso-grid-selection", dest="dso_grid_selection", action="store_true", default=None)
    parser.add_argument("--dso-no-grid-selection", dest="dso_grid_selection", action="store_false")
    parser.add_argument("--dso-outlier-culling", dest="dso_outlier_culling", action="store_true", default=None)
    parser.add_argument("--dso-no-outlier-culling", dest="dso_outlier_culling", action="store_false")
    parser.add_argument("--dso-left-right-depth-check", dest="dso_left_right_depth_check", action="store_true", default=None)
    parser.add_argument("--dso-no-left-right-depth-check", dest="dso_left_right_depth_check", action="store_false")
    parser.add_argument("--dso-strict-loop-verification", dest="dso_strict_loop_verification", action="store_true", default=None)
    parser.add_argument("--dso-no-strict-loop-verification", dest="dso_strict_loop_verification", action="store_false")
    parser.add_argument("--dso-loop-correction", dest="dso_loop_correction", action="store_true", default=None)
    parser.add_argument("--dso-no-loop-correction", dest="dso_loop_correction", action="store_false")
    parser.add_argument("--dso-residual-p95-gate", dest="dso_residual_p95_gate", action="store_true", default=None)
    parser.add_argument("--dso-no-residual-p95-gate", dest="dso_residual_p95_gate", action="store_false")
    parser.add_argument("--dso-cost-jump-gate", dest="dso_cost_jump_gate", action="store_true", default=None)
    parser.add_argument("--dso-no-cost-jump-gate", dest="dso_cost_jump_gate", action="store_false")
    parser.add_argument("--dso-lk-consistency-gate", dest="dso_lk_consistency_gate", action="store_true", default=None)
    parser.add_argument("--dso-no-lk-consistency-gate", dest="dso_lk_consistency_gate", action="store_false")
    parser.add_argument("--dso-joint-window-ba", dest="dso_joint_window_ba", action="store_true", default=None)
    parser.add_argument("--dso-no-joint-window-ba", dest="dso_joint_window_ba", action="store_false")
    parser.add_argument("--dso-affine-brightness", dest="dso_affine_brightness", action="store_true", default=None)
    parser.add_argument("--dso-no-affine-brightness", dest="dso_affine_brightness", action="store_false")
    parser.add_argument("--dso-clahe", dest="dso_clahe", action="store_true", default=None)
    parser.add_argument("--dso-no-clahe", dest="dso_clahe", action="store_false")
    parser.add_argument("--dso-gradient-normalized-residual", dest="dso_gradient_normalized_residual", action="store_true", default=None)
    parser.add_argument("--dso-no-gradient-normalized-residual", dest="dso_gradient_normalized_residual", action="store_false")
    parser.add_argument("--dso-max-residual-p95", type=float, default=None)
    parser.add_argument("--dso-max-cost-jump-ratio", type=float, default=None)
    parser.add_argument("--dso-ba-max-points-per-keyframe", type=int, default=None)
    parser.add_argument("--dso-ba-max-nfev", type=int, default=None)
    parser.add_argument("--dso-max-consecutive-failures", type=int, default=None)
    parser.add_argument("--disable-dso-motion-gate", action="store_true")
    parser.add_argument("--disable-dso-grid-selection", action="store_true")
    parser.add_argument("--disable-dso-outlier-culling", action="store_true")
    parser.add_argument("--disable-dso-photometric-refinement", action="store_true")
    parser.add_argument("--disable-dso-left-right-depth-check", action="store_true")
    parser.add_argument("--disable-dso-quality-gates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq = f"{int(args.seq):02d}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_loader = KITTIOdometryLoader(args.data_dir, seq)
    if base_loader.gt_poses is None:
        raise FileNotFoundError(f"Missing ground truth for sequence {seq}")

    expected_len = args.max_frames or len(base_loader)
    gt = base_loader.gt_poses[:expected_len]
    dso_config = build_dso_config(args)

    results: dict[str, dict[str, object]] = {}
    trajectories: dict[str, np.ndarray] = {}
    plot_inputs: dict[str, np.ndarray] = {}
    saved_paths: dict[str, str] = {}
    dso_slam: DSOSLAM | None = None

    if "orb" in args.algorithms:
        start = time.perf_counter()
        orb_slam = ORBSLAM2(LimitedLoader(base_loader, args.max_frames), num_features=args.features)
        orb_traj, orb_runtimes = orb_slam.run()
        wall_s = time.perf_counter() - start
        path = output_dir / f"orb_slam_seq{seq}.txt"
        save_trajectory_kitti(orb_traj, path)
        orb_array = np.stack(orb_traj)
        trajectories["orb"] = orb_array
        plot_inputs["ORB-SLAM"] = orb_array
        saved_paths["orb"] = str(path)
        results["orb"] = {
            "trajectory_path": str(path),
            **summarize_method(gt, orb_array, alignment=args.alignment),
            "runtime": runtime_summary(orb_runtimes),
            "wall_time_s": wall_s,
            "keyframes": len(orb_slam.keyframes),
            "robustness": orb_slam.get_stats(),
        }

    if "dso" in args.algorithms:
        start = time.perf_counter()
        dso_slam = DSOSLAM(
            LimitedLoader(base_loader, args.max_frames),
            num_active_points=args.points,
            tracking_config=dso_config,
        )
        dso_traj, dso_runtimes = dso_slam.run()
        wall_s = time.perf_counter() - start
        path = output_dir / f"dso_slam_seq{seq}.txt"
        save_trajectory_kitti(dso_traj, path)
        dso_array = np.stack(dso_traj)
        trajectories["dso"] = dso_array
        plot_inputs["DSO-SLAM"] = dso_array
        saved_paths["dso"] = str(path)
        results["dso"] = {
            "trajectory_path": str(path),
            **summarize_method(gt, dso_array, alignment=args.alignment),
            "runtime": runtime_summary(dso_runtimes),
            "wall_time_s": wall_s,
            "keyframes": len(dso_slam.keyframes),
            "robustness": dso_slam.get_stats(),
        }

    if "svo" in args.algorithms:
        start = time.perf_counter()
        svo_slam = SVOStyleSLAM(LimitedLoader(base_loader, args.max_frames), num_points=args.svo_points)
        svo_traj, svo_runtimes = svo_slam.run()
        wall_s = time.perf_counter() - start
        path = output_dir / f"svo_slam_seq{seq}.txt"
        save_trajectory_kitti(svo_traj, path)
        svo_array = np.stack(svo_traj)
        trajectories["svo"] = svo_array
        plot_inputs["SVO-SLAM"] = svo_array
        saved_paths["svo"] = str(path)
        results["svo"] = {
            "trajectory_path": str(path),
            **summarize_method(gt, svo_array, alignment=args.alignment),
            "runtime": runtime_summary(svo_runtimes),
            "wall_time_s": wall_s,
            "keyframes": len(svo_slam.keyframes),
            "robustness": svo_slam.get_stats(),
        }

    mismatched = {name: len(array) for name, array in trajectories.items() if len(array) != expected_len}
    if mismatched:
        lengths = ", ".join(f"{name.upper()}={length}" for name, length in sorted(mismatched.items()))
        raise ValueError(
            f"Benchmark produced misaligned trajectories: expected {expected_len}, {lengths}"
        )

    summary = {
        "sequence": seq,
        "max_frames": args.max_frames,
        "alignment": args.alignment,
        "dataset": {
            "data_dir": str(args.data_dir),
            "sequence": seq,
            "expected_frames": expected_len,
        },
        "parameters": {
            "algorithms": args.algorithms,
            "features": args.features,
            "dso_points": args.points,
            "svo_points": args.svo_points,
            "dso_mode": args.dso_mode,
            "dso_tracking_config": asdict(dso_config),
        },
        "implementation_manifest": implementation_manifest(),
        **results,
    }
    safe_summary = json_safe(summary)

    summary_path = output_dir / f"benchmark_seq{seq}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(safe_summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    plot_path = output_dir / f"benchmark_trajectory_seq{seq}.png"
    plot_trajectories(gt, plot_inputs, plot_path, alignment=args.alignment)

    if dso_slam is not None:
        diagnostics = dso_slam.get_diagnostics()
        diag_json = output_dir / f"dso_diagnostics_seq{seq}.json"
        diag_csv = output_dir / f"dso_diagnostics_seq{seq}.csv"
        diag_png = output_dir / f"dso_diagnostics_seq{seq}.png"
        write_diagnostics_json_csv(diagnostics, diag_json, diag_csv)
        plot_dso_diagnostics(diagnostics, diag_png)
        saved_paths["dso_diagnostics_json"] = str(diag_json)
        saved_paths["dso_diagnostics_csv"] = str(diag_csv)
        saved_paths["dso_diagnostics_png"] = str(diag_png)

    print(json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False))
    for name, path in saved_paths.items():
        print(f"Saved {name} to {path}")
    print(f"Saved benchmark summary to {summary_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
