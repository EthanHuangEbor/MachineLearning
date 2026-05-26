from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from slam_profiles import implementation_manifest


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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


def resolve_existing_path(candidates: list[Path], description: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Tried:\n  {tried}")


def resolve_ground_truth_path(data_dir: str | Path, seq: str) -> Path:
    data_dir = Path(data_dir)
    return resolve_existing_path(
        [
            data_dir / "poses" / f"{seq}.txt",
            data_dir / "sequences" / "poses" / f"{seq}.txt",
        ],
        f"ground-truth poses for sequence {seq}",
    )


def resolve_trajectory_path(
    explicit_path: str | None,
    output_dir: str | Path,
    seq: str,
    default_names: list[str],
    description: str,
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"{description} does not exist: {path}")
        return path
    output_dir = Path(output_dir)
    return resolve_existing_path([output_dir / name.format(seq=seq) for name in default_names], description)


def normalize_trajectory(poses: np.ndarray) -> np.ndarray:
    origin_inv = np.linalg.inv(poses[0])
    return np.stack([origin_inv @ pose for pose in poses])


def align_pair(gt: np.ndarray, est: np.ndarray, allow_truncate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if len(gt) != len(est):
        if not allow_truncate:
            raise ValueError(
                f"Trajectory length mismatch: ground truth has {len(gt)} poses, "
                f"estimate has {len(est)} poses. Fix frame alignment or pass --allow-truncate."
            )
        length = min(len(gt), len(est))
        return gt[:length], est[:length]
    return gt, est


def umeyama_align(src: np.ndarray, dst: np.ndarray, with_scale: bool = False) -> np.ndarray:
    """Align src points to dst points with SE(3) or Sim(3) Umeyama alignment."""
    if len(src) < 3:
        return src.copy()

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = dst_centered.T @ src_centered / len(src)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt

    scale = 1.0
    if with_scale:
        variance = np.mean(np.sum(src_centered ** 2, axis=1))
        if variance > 1e-12:
            scale = np.trace(np.diag(np.linalg.svd(covariance, compute_uv=False)) @ correction) / variance

    translation = dst_mean - scale * rotation @ src_mean
    return (scale * (rotation @ src.T)).T + translation


def trajectory_positions_for_ate(gt: np.ndarray, est: np.ndarray, alignment: str) -> tuple[np.ndarray, np.ndarray]:
    gt_pos = gt[:, :3, 3]
    est_pos = est[:, :3, 3]
    if alignment == "se3":
        est_pos = umeyama_align(est_pos, gt_pos, with_scale=False)
    elif alignment == "sim3":
        est_pos = umeyama_align(est_pos, gt_pos, with_scale=True)
    return gt_pos, est_pos


def compute_ate_rmse(
    gt: np.ndarray,
    est: np.ndarray,
    *,
    allow_truncate: bool = False,
    alignment: str = "origin",
) -> float:
    gt, est = align_pair(normalize_trajectory(gt), normalize_trajectory(est), allow_truncate)
    gt_pos, est_pos = trajectory_positions_for_ate(gt, est, alignment)
    diff = gt_pos - est_pos
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def accumulated_distances(poses: np.ndarray) -> np.ndarray:
    positions = poses[:, :3, 3]
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def compute_rpe_metrics(
    gt: np.ndarray,
    est: np.ndarray,
    delta_m: float = 100.0,
    *,
    allow_truncate: bool = False,
) -> dict[str, float]:
    gt, est = align_pair(normalize_trajectory(gt), normalize_trajectory(est), allow_truncate)
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


def compute_kitti_segment_metrics(
    gt: np.ndarray,
    est: np.ndarray,
    *,
    segment_lengths: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800),
    allow_truncate: bool = False,
) -> dict[str, object]:
    """KITTI-style average relative errors over fixed path-length segments."""
    gt, est = align_pair(normalize_trajectory(gt), normalize_trajectory(est), allow_truncate)
    distances = accumulated_distances(gt)
    per_length: dict[str, dict[str, float | int]] = {}
    all_trans_errors = []
    all_rot_errors = []

    for length_m in segment_lengths:
        trans_errors = []
        rot_errors = []
        for start in range(len(gt) - 1):
            end = int(np.searchsorted(distances, distances[start] + length_m, side="left"))
            if end >= len(gt):
                continue

            gt_rel = np.linalg.inv(gt[start]) @ gt[end]
            est_rel = np.linalg.inv(est[start]) @ est[end]
            error_rel = np.linalg.inv(gt_rel) @ est_rel
            trans_errors.append(float(np.linalg.norm(error_rel[:3, 3]) / length_m * 100.0))
            rot_errors.append(float(rotation_angle_deg(error_rel[:3, :3]) / length_m))

        per_length[str(length_m)] = {
            "num_segments": len(trans_errors),
            "trans_percent": float(np.mean(trans_errors)) if trans_errors else float("nan"),
            "rot_deg_per_m": float(np.mean(rot_errors)) if rot_errors else float("nan"),
        }
        all_trans_errors.extend(trans_errors)
        all_rot_errors.extend(rot_errors)

    return {
        "segment_lengths_m": list(segment_lengths),
        "per_length": per_length,
        "mean_trans_percent": float(np.mean(all_trans_errors)) if all_trans_errors else float("nan"),
        "mean_rot_deg_per_m": float(np.mean(all_rot_errors)) if all_rot_errors else float("nan"),
        "num_segments": len(all_trans_errors),
    }


def plot_trajectories(
    gt: np.ndarray,
    trajectories: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    allow_truncate: bool = False,
    alignment: str = "origin",
) -> None:
    plt.figure(figsize=(8, 6))
    gt_pos = normalize_trajectory(gt)[:, :3, 3]
    plt.plot(gt_pos[:, 0], gt_pos[:, 2], label="Ground Truth", linewidth=2)

    for label, trajectory in trajectories.items():
        gt_aligned, trajectory_aligned = align_pair(
            normalize_trajectory(gt), normalize_trajectory(trajectory), allow_truncate
        )
        _, pos = trajectory_positions_for_ate(gt_aligned, trajectory_aligned, alignment)
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


def summarize_method(
    gt: np.ndarray,
    est: np.ndarray,
    *,
    allow_truncate: bool = False,
    alignment: str = "origin",
) -> dict[str, object]:
    summary = {"ate_rmse_m": compute_ate_rmse(gt, est, allow_truncate=allow_truncate, alignment=alignment)}
    summary.update(compute_rpe_metrics(gt, est, allow_truncate=allow_truncate))
    summary["kitti_segments"] = compute_kitti_segment_metrics(gt, est, allow_truncate=allow_truncate)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KITTI trajectories")
    parser.add_argument("--data-dir", default="data/kitti_odometry", help="KITTI odometry root directory")
    parser.add_argument("--seq", default="00", help="KITTI sequence number")
    parser.add_argument("--orb", default=None, help="ORB/ORB-SLAM trajectory path")
    parser.add_argument("--dso", default=None, help="DSO/Direct trajectory path")
    parser.add_argument("--direct", default=None, help="Legacy alias for --dso")
    parser.add_argument("--output-dir", default="results", help="Directory for summary and plots")
    parser.add_argument(
        "--alignment",
        choices=["origin", "se3", "sim3"],
        default="origin",
        help="Global alignment used for ATE and plots",
    )
    parser.add_argument(
        "--allow-truncate",
        action="store_true",
        help="Allow evaluation after truncating mismatched trajectories to the shorter length",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq = f"{int(args.seq):02d}"
    gt_path = resolve_ground_truth_path(args.data_dir, seq)
    orb_path = resolve_trajectory_path(
        args.orb,
        args.output_dir,
        seq,
        ["orb_slam_seq{seq}.txt", "orb_seq{seq}.txt"],
        "ORB trajectory",
    )
    dso_path = resolve_trajectory_path(
        args.dso or args.direct,
        args.output_dir,
        seq,
        ["dso_slam_seq{seq}.txt", "direct_seq{seq}.txt"],
        "DSO trajectory",
    )

    gt = load_kitti_poses(gt_path)
    orb = load_kitti_poses(orb_path)
    dso = load_kitti_poses(dso_path)

    summary = {
        "sequence": seq,
        "alignment": args.alignment,
        "allow_truncate": args.allow_truncate,
        "ground_truth_path": str(gt_path),
        "orb_path": str(orb_path),
        "dso_path": str(dso_path),
        "implementation_manifest": implementation_manifest(),
        "orb": summarize_method(gt, orb, allow_truncate=args.allow_truncate, alignment=args.alignment),
        "dso": summarize_method(gt, dso, allow_truncate=args.allow_truncate, alignment=args.alignment),
    }
    safe_summary = json_safe(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"summary_seq{seq}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(safe_summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    plot_path = output_dir / f"trajectory_seq{seq}.png"
    plot_trajectories(
        gt,
        {"ORB-SLAM": orb, "DSO-SLAM": dso},
        plot_path,
        allow_truncate=args.allow_truncate,
        alignment=args.alignment,
    )

    print(json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
