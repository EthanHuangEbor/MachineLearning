from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dso_slam import DSOTrackingConfig, DSOSLAM
from evaluate import json_safe, summarize_method
from kitti_utils import KITTIOdometryLoader, save_trajectory_kitti
from run_benchmark import LimitedLoader, runtime_summary


def ablation_configs() -> dict[str, DSOTrackingConfig]:
    return {
        "strict_full": DSOTrackingConfig(),
        "no_left_right_check": DSOTrackingConfig(
            enable_left_right_depth_check=False,
        ),
        "no_quality_gates": DSOTrackingConfig(
            enable_residual_p95_gate=False,
            enable_cost_jump_gate=False,
            enable_lk_consistency_gate=False,
            min_valid_projected_points=0,
            min_valid_projected_ratio=0.0,
            min_inlier_ratio=0.0,
        ),
        "no_joint_ba": DSOTrackingConfig(
            enable_joint_window_ba=False,
        ),
        "no_affine_brightness": DSOTrackingConfig(
            enable_affine_brightness=False,
        ),
        "loop_candidates_only": DSOTrackingConfig(
            enable_loop_correction=False,
        ),
        "lk_pnp_only": DSOTrackingConfig(
            enable_photometric_refinement=False,
            enable_outlier_culling=False,
        ),
        "photometric_with_motion_gate": DSOTrackingConfig(
            enable_grid_selection=False,
            enable_outlier_culling=False,
            enable_motion_gate=True,
        ),
        "grid_uniform_selection": DSOTrackingConfig(
            enable_grid_selection=True,
            enable_outlier_culling=False,
        ),
        "outlier_culling_on": DSOTrackingConfig(
            enable_grid_selection=False,
            enable_outlier_culling=True,
        ),
        "outlier_culling_off": DSOTrackingConfig(
            enable_grid_selection=False,
            enable_outlier_culling=False,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DSO tracking ablations on KITTI")
    parser.add_argument("--data-dir", default="data/kitti_odometry")
    parser.add_argument("--seq", default="00")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--points", type=int, default=800)
    parser.add_argument("--alignment", choices=["origin", "se3", "sim3"], default="origin")
    return parser.parse_args()


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in json_safe(rows):
            writer.writerow(row)


def plot_ablation(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        output_path.write_bytes(b"")
        return
    names = [str(row["config"]) for row in rows]
    ate = [float(row["ate_rmse_m"]) if row["ate_rmse_m"] is not None else np.nan for row in rows]
    failures = [float(row["tracking_failures"]) for row in rows]
    fallbacks = [float(row["fallbacks_used"]) for row in rows]

    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(x, ate, color="tab:blue")
    axes[0].set_ylabel("ATE RMSE (m)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x - 0.18, failures, width=0.36, label="failures", color="tab:red")
    axes[1].bar(x + 0.18, fallbacks, width=0.36, label="fallbacks", color="tab:orange")
    axes[1].set_ylabel("count")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("DSO Ablation Summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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

    summaries: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    for name, config in ablation_configs().items():
        start = time.perf_counter()
        slam = DSOSLAM(
            LimitedLoader(base_loader, args.max_frames),
            num_active_points=args.points,
            tracking_config=config,
        )
        trajectory, runtimes = slam.run()
        wall_s = time.perf_counter() - start
        if len(trajectory) != expected_len:
            raise ValueError(
                f"Ablation {name} produced misaligned trajectory: expected {expected_len}, got {len(trajectory)}"
            )

        traj_path = output_dir / f"dso_{name}_seq{seq}.txt"
        save_trajectory_kitti(trajectory, traj_path)
        trajectory_array = np.stack(trajectory)
        method_summary = summarize_method(gt, trajectory_array, alignment=args.alignment)
        stats = slam.get_stats()
        summaries[name] = {
            "trajectory_path": str(traj_path),
            "tracking_config": asdict(config),
            **method_summary,
            "runtime": runtime_summary(runtimes),
            "wall_time_s": wall_s,
            "keyframes": len(slam.keyframes),
            "robustness": stats,
        }
        csv_rows.append(
            {
                "config": name,
                "ate_rmse_m": method_summary["ate_rmse_m"],
                "rpe_trans_percent": method_summary["rpe_trans_percent"],
                "rpe_rot_deg_per_m": method_summary["rpe_rot_deg_per_m"],
                "kitti_mean_trans_percent": method_summary["kitti_segments"]["mean_trans_percent"],
                "tracking_failures": stats["tracking_failures"],
                "fallbacks_used": stats["fallbacks_used"],
                "motion_gate_rejections": stats["motion_gate_rejections"],
                "low_projection_rejections": stats["low_projection_rejections"],
                "residual_p95_gate_rejections": stats["residual_p95_gate_rejections"],
                "cost_jump_rejections": stats["cost_jump_rejections"],
                "lk_consistency_rejections": stats["lk_consistency_rejections"],
                "reinitializations": stats["reinitializations"],
                "loop_candidates": stats["loop_candidates"],
                "loop_verified": stats["loop_verified"],
                "loop_corrections_applied": stats["loop_corrections_applied"],
                "ba_runs": stats["ba_runs"],
                "ba_accepted": stats["ba_accepted"],
                "ba_rejected": stats["ba_rejected"],
                "active_points_culled": stats["active_points_culled"],
                "keyframes": len(slam.keyframes),
                "wall_time_s": wall_s,
                "avg_runtime_ms": runtime_summary(runtimes)["avg_ms"],
            }
        )

    summary = {
        "sequence": seq,
        "max_frames": args.max_frames,
        "alignment": args.alignment,
        "points": args.points,
        "configs": summaries,
    }
    summary_path = output_dir / f"dso_ablation_seq{seq}.json"
    csv_path = output_dir / f"dso_ablation_seq{seq}.csv"
    plot_path = output_dir / f"dso_ablation_seq{seq}.png"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False, allow_nan=False)
    write_csv(csv_rows, csv_path)
    plot_ablation(json_safe(csv_rows), plot_path)

    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved DSO ablation summary to {summary_path}")
    print(f"Saved DSO ablation CSV to {csv_path}")
    print(f"Saved DSO ablation plot to {plot_path}")


if __name__ == "__main__":
    main()
