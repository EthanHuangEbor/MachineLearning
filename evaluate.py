from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_kitti_poses(file_path: str | Path) -> np.ndarray:
    file_path = Path(file_path)
    poses = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            values = np.fromstring(line.strip(), sep=" ")
            if values.size != 12:
                continue
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = values.reshape(3, 4)
            poses.append(pose)
    if not poses:
        raise ValueError(f"No poses found in {file_path}")
    return np.stack(poses)


def normalize_trajectory(poses: np.ndarray) -> np.ndarray:
    origin_inv = np.linalg.inv(poses[0])
    return np.stack([origin_inv @ pose for pose in poses])


def truncate_pair(gt: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(gt), len(est))
    return gt[:length], est[:length]


def compute_ate_rmse(gt: np.ndarray, est: np.ndarray) -> float:
    gt, est = truncate_pair(normalize_trajectory(gt), normalize_trajectory(est))
    diff = gt[:, :3, 3] - est[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def accumulated_distances(poses: np.ndarray) -> np.ndarray:
    positions = poses[:, :3, 3]
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def compute_rpe_metrics(gt: np.ndarray, est: np.ndarray, delta_m: float = 100.0) -> dict[str, float]:
    gt, est = truncate_pair(normalize_trajectory(gt), normalize_trajectory(est))
    distances = accumulated_distances(gt)
    trans_errors = []
    rot_errors = []

    for start in range(len(gt) - 1):
        target_distance = distances[start] + delta_m
        end = int(np.searchsorted(distances, target_distance, side="left"))
        if end >= len(gt):
            continue

        gt_rel = np.linalg.inv(gt[start]) @ gt[end]
        est_rel = np.linalg.inv(est[start]) @ est[end]
        error_rel = np.linalg.inv(gt_rel) @ est_rel

        translation_error = np.linalg.norm(error_rel[:3, 3]) / delta_m * 100.0
        rotation_error = rotation_angle_deg(error_rel[:3, :3]) / delta_m
        trans_errors.append(float(translation_error))
        rot_errors.append(float(rotation_error))

    if not trans_errors:
        return {"rpe_trans_percent": float("nan"), "rpe_rot_deg_per_m": float("nan")}

    return {
        "rpe_trans_percent": float(np.mean(trans_errors)),
        "rpe_rot_deg_per_m": float(np.mean(rot_errors)),
    }


def plot_trajectories(gt: np.ndarray, trajectories: dict[str, np.ndarray], output_path: str | Path) -> None:
    plt.figure(figsize=(8, 6))
    gt_pos = normalize_trajectory(gt)[:, :3, 3]
    plt.plot(gt_pos[:, 0], gt_pos[:, 2], label="Ground Truth", linewidth=2)

    for label, trajectory in trajectories.items():
        pos = normalize_trajectory(trajectory)[:, :3, 3]
        plt.plot(pos[:, 0], pos[:, 2], label=label)

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("KITTI Trajectory Comparison")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_method(gt: np.ndarray, est: np.ndarray) -> dict[str, float]:
    summary = {"ate_rmse_m": compute_ate_rmse(gt, est)}
    summary.update(compute_rpe_metrics(gt, est))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KITTI trajectories")
    parser.add_argument("--data-dir", default="data/kitti_odometry", help="KITTI odometry root directory")
    parser.add_argument("--seq", default="00", help="KITTI sequence number")
    parser.add_argument("--orb", default=None, help="ORB trajectory path")
    parser.add_argument("--direct", default=None, help="Direct VO trajectory path")
    parser.add_argument("--output-dir", default="results", help="Directory for summary and plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq = f"{int(args.seq):02d}"
    gt_path = Path(args.data_dir) / "poses" / f"{seq}.txt"
    orb_path = Path(args.orb) if args.orb else Path(args.output_dir) / f"orb_seq{seq}.txt"
    direct_path = Path(args.direct) if args.direct else Path(args.output_dir) / f"direct_seq{seq}.txt"

    gt = load_kitti_poses(gt_path)
    orb = load_kitti_poses(orb_path)
    direct = load_kitti_poses(direct_path)

    summary = {
        "sequence": seq,
        "orb": summarize_method(gt, orb),
        "direct": summarize_method(gt, direct),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"summary_seq{seq}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_path = output_dir / f"trajectory_seq{seq}.png"
    plot_trajectories(gt, {"ORB Stereo VO": orb, "Direct Sparse VO": direct}, plot_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
