from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from dso_slam import (
    DSOTrackingConfig,
    bilinear_interpolate,
    compute_stereo_depth,
    finite_float_or_none,
    relative_motion_stats,
    select_grid_uniform_high_gradient_pixels,
)
from kitti_utils import KITTIOdometryLoader, disparity_to_depth, save_trajectory_kitti


@dataclass
class SVOKeyFrame:
    id: int
    timestamp: float
    pose: np.ndarray
    gray: np.ndarray
    depth: np.ndarray
    active_uvs: np.ndarray
    active_points_3d: np.ndarray
    active_intensities: np.ndarray

    @staticmethod
    def next_id() -> int:
        SVOKeyFrame._counter += 1
        return SVOKeyFrame._counter


SVOKeyFrame._counter = 0


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
    """Select spatially uniform high-gradient points for SVO-style patch tracking."""
    uvs, _, _ = select_grid_uniform_high_gradient_pixels(
        gray,
        depth,
        num_points=num_points,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        border=border,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return uvs.astype(np.float32)


class SVOStyleSLAM:
    """SVO-style semi-direct stereo visual odometry/SLAM baseline."""

    def __init__(
        self,
        loader: KITTIOdometryLoader,
        num_points: int = 1500,
        tracking_config: Optional[DSOTrackingConfig] = None,
        keyframe_interval: int = 5,
    ):
        self.loader = loader
        self.calib = loader.calibration
        self.num_points = num_points
        self.tracking_config = tracking_config or DSOTrackingConfig(
            enable_photometric_refinement=False,
            enable_grid_selection=True,
            enable_outlier_culling=True,
        )
        self.keyframe_interval = keyframe_interval
        self.keyframes: list[SVOKeyFrame] = []
        self.current_kf: Optional[SVOKeyFrame] = None
        self.frames_processed = 0
        self.frames_since_keyframe = 0
        self.tracking_failures = 0
        self.fallbacks_used = 0
        self.motion_gate_rejections = 0
        self.low_projection_rejections = 0
        self.loop_candidates = 0
        self.loop_closures = 0
        self.last_velocity = np.eye(4, dtype=np.float64)
        self.last_safe_pose = np.eye(4, dtype=np.float64)
        self.last_report = self._empty_report(None)
        self.diagnostics: list[dict[str, object]] = []

    def _empty_report(self, failure_reason: str | None) -> dict[str, object]:
        return {
            "success": False,
            "failure_reason": failure_reason,
            "valid_projected_points": 0,
            "inlier_ratio": None,
            "residual_mean": None,
            "residual_median": None,
            "residual_p95": None,
            "relative_translation_m": None,
            "relative_rotation_deg": None,
            "mean_flow_px": 0.0,
            "pnp_inliers": 0,
        }

    def _make_keyframe(self, left: np.ndarray, right: np.ndarray, pose: np.ndarray, timestamp: float | None) -> SVOKeyFrame:
        disparity = compute_stereo_depth(left, right)
        depth = disparity_to_depth(disparity, self.calib)
        uvs = select_svo_points_grid(left, depth, self.num_points)
        if len(uvs) == 0:
            points_3d = np.empty((0, 3), dtype=np.float32)
            intensities = np.empty(0, dtype=np.float32)
        else:
            z = bilinear_interpolate(depth, uvs[:, 0], uvs[:, 1])
            valid = np.isfinite(z) & (z > 0.5) & (z < 80.0)
            uvs = uvs[valid]
            z = z[valid]
            x = (uvs[:, 0] - self.calib.cx) * z / self.calib.focal_length
            y = (uvs[:, 1] - self.calib.cy) * z / self.calib.focal_length
            points_3d = np.column_stack([x, y, z]).astype(np.float32)
            intensities = bilinear_interpolate(left.astype(np.float32), uvs[:, 0], uvs[:, 1]).astype(np.float32)

        return SVOKeyFrame(
            id=SVOKeyFrame.next_id(),
            timestamp=float(timestamp or 0.0),
            pose=pose.copy(),
            gray=left.copy(),
            depth=depth.copy(),
            active_uvs=uvs.astype(np.float32),
            active_points_3d=points_3d,
            active_intensities=intensities,
        )

    def _track(self, ref_kf: SVOKeyFrame, curr_gray: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        if len(ref_kf.active_uvs) < 6:
            return np.eye(4, dtype=np.float64), self._empty_report("insufficient_points")

        max_points = min(self.num_points, len(ref_kf.active_uvs))
        step = max(1, len(ref_kf.active_uvs) // max_points)
        indices = np.arange(0, len(ref_kf.active_uvs), step, dtype=int)[:max_points]
        prev_pts = ref_kf.active_uvs[indices].reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            ref_kf.gray,
            curr_gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            return np.eye(4, dtype=np.float64), self._empty_report("optical_flow_failed")

        flow_mask = status.ravel() > 0
        object_pts = ref_kf.active_points_3d[indices][flow_mask].astype(np.float32)
        image_pts = next_pts.reshape(-1, 2)[flow_mask].astype(np.float32)
        ref_uvs = ref_kf.active_uvs[indices][flow_mask]
        ref_intensities = ref_kf.active_intensities[indices][flow_mask]
        valid_flow = len(object_pts)
        if valid_flow < 6:
            report = self._empty_report("low_projection")
            report["valid_projected_points"] = int(valid_flow)
            return np.eye(4, dtype=np.float64), report

        h, w = curr_gray.shape
        in_bounds = (
            (image_pts[:, 0] > 0)
            & (image_pts[:, 0] < w - 1)
            & (image_pts[:, 1] > 0)
            & (image_pts[:, 1] < h - 1)
        )
        object_pts = object_pts[in_bounds]
        image_pts = image_pts[in_bounds]
        ref_uvs = ref_uvs[in_bounds]
        ref_intensities = ref_intensities[in_bounds]
        if len(object_pts) < 6:
            report = self._empty_report("low_projection")
            report["valid_projected_points"] = int(len(object_pts))
            return np.eye(4, dtype=np.float64), report

        curr_intensities = bilinear_interpolate(curr_gray.astype(np.float32), image_pts[:, 0], image_pts[:, 1])
        residuals = np.abs(ref_intensities - curr_intensities)
        if self.tracking_config.enable_outlier_culling:
            keep = residuals <= self.tracking_config.residual_inlier_threshold
            if int(keep.sum()) >= 6:
                object_pts = object_pts[keep]
                image_pts = image_pts[keep]
                ref_uvs = ref_uvs[keep]
                residuals = residuals[keep]

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_pts,
            image_pts,
            self.calib.k_left,
            None,
            reprojectionError=4.0,
            iterationsCount=100,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None or len(inliers) < 6:
            report = self._empty_report("pnp_failed")
            report["valid_projected_points"] = int(len(object_pts))
            report["residual_mean"] = finite_float_or_none(np.mean(residuals)) if len(residuals) else None
            report["residual_median"] = finite_float_or_none(np.median(residuals)) if len(residuals) else None
            report["residual_p95"] = finite_float_or_none(np.percentile(residuals, 95)) if len(residuals) else None
            return np.eye(4, dtype=np.float64), report

        rotation, _ = cv2.Rodrigues(rvec)
        relative_pose = np.eye(4, dtype=np.float64)
        relative_pose[:3, :3] = rotation
        relative_pose[:3, 3] = tvec.reshape(3)
        translation_m, rotation_deg = relative_motion_stats(relative_pose)
        inlier_ratio = float(len(inliers) / max(len(object_pts), 1))
        flow = np.linalg.norm(image_pts - ref_uvs, axis=1)

        failure_reason = None
        if self.tracking_config.enable_motion_gate and (
            translation_m > self.tracking_config.max_translation_m
            or rotation_deg > self.tracking_config.max_rotation_deg
        ):
            failure_reason = "motion_gate"
        elif len(object_pts) < self.tracking_config.min_valid_projected_points:
            failure_reason = "low_projection"
        elif inlier_ratio < self.tracking_config.min_inlier_ratio:
            failure_reason = "low_inlier_ratio"

        report = {
            "success": failure_reason is None,
            "failure_reason": failure_reason,
            "valid_projected_points": int(len(object_pts)),
            "inlier_ratio": finite_float_or_none(inlier_ratio),
            "residual_mean": finite_float_or_none(np.mean(residuals)) if len(residuals) else None,
            "residual_median": finite_float_or_none(np.median(residuals)) if len(residuals) else None,
            "residual_p95": finite_float_or_none(np.percentile(residuals, 95)) if len(residuals) else None,
            "relative_translation_m": finite_float_or_none(translation_m),
            "relative_rotation_deg": finite_float_or_none(rotation_deg),
            "mean_flow_px": float(np.mean(flow)) if len(flow) else 0.0,
            "pnp_inliers": int(len(inliers)),
        }
        return relative_pose, report

    def _fallback_pose(self, previous_pose: np.ndarray) -> tuple[np.ndarray, str]:
        translation_m, rotation_deg = relative_motion_stats(self.last_velocity)
        if (
            not self.tracking_config.enable_motion_gate
            or (
                translation_m <= self.tracking_config.max_translation_m
                and rotation_deg <= self.tracking_config.max_rotation_deg
            )
        ):
            return previous_pose @ self.last_velocity, "constant_velocity"
        return self.last_safe_pose.copy(), "last_safe_pose"

    def _should_insert_keyframe(self, report: dict[str, object]) -> bool:
        if len(self.keyframes) == 0:
            return True
        if self.frames_since_keyframe >= self.keyframe_interval:
            return True
        if int(report.get("pnp_inliers", 0) or 0) < 40:
            return True
        return float(report.get("mean_flow_px", 0.0) or 0.0) > 25.0

    def _active_sparse_points(self) -> int:
        return int(sum(len(kf.active_points_3d) for kf in self.keyframes[-5:]))

    def _record_diagnostic(
        self,
        frame,
        report: dict[str, object],
        *,
        tracking_failure: bool,
        fallback_used: bool,
        fallback_reason: str | None,
        keyframe_inserted: bool,
    ) -> None:
        self.diagnostics.append(
            {
                "frame_id": int(frame.index),
                "timestamp": frame.timestamp,
                "valid_projected_points": int(report.get("valid_projected_points", 0) or 0),
                "tracked_points": int(report.get("valid_projected_points", 0) or 0),
                "pnp_inliers": int(report.get("pnp_inliers", 0) or 0),
                "residual_mean": finite_float_or_none(report.get("residual_mean")),
                "residual_median": finite_float_or_none(report.get("residual_median")),
                "residual_p95": finite_float_or_none(report.get("residual_p95")),
                "inlier_ratio": finite_float_or_none(report.get("inlier_ratio")),
                "relative_translation_m": finite_float_or_none(report.get("relative_translation_m")),
                "relative_rotation_deg": finite_float_or_none(report.get("relative_rotation_deg")),
                "tracking_failure": bool(tracking_failure),
                "fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
                "keyframe_inserted": bool(keyframe_inserted),
                "active_sparse_points": self._active_sparse_points(),
            }
        )

    def run(self) -> tuple[list[np.ndarray], list[float]]:
        trajectory = [np.eye(4, dtype=np.float64)]
        runtimes: list[float] = []
        frames = iter(self.loader.iter_frames())
        try:
            first_frame = next(frames)
        except StopIteration:
            return trajectory, runtimes

        start = time.perf_counter()
        current_pose = np.eye(4, dtype=np.float64)
        previous_pose = current_pose.copy()
        self.last_safe_pose = current_pose.copy()
        self.last_velocity = np.eye(4, dtype=np.float64)
        first_kf = self._make_keyframe(first_frame.left, first_frame.right, current_pose, first_frame.timestamp)
        self.keyframes.append(first_kf)
        self.current_kf = first_kf
        self.frames_processed = 1
        self._record_diagnostic(
            first_frame,
            self._empty_report(None),
            tracking_failure=False,
            fallback_used=False,
            fallback_reason=None,
            keyframe_inserted=True,
        )
        runtimes.append((time.perf_counter() - start) * 1000.0)

        for frame in frames:
            start = time.perf_counter()
            relative_pose, report = self._track(self.current_kf, frame.left)
            tracking_failure = not bool(report.get("success", False))
            fallback_used = False
            fallback_reason = None

            if tracking_failure:
                self.tracking_failures += 1
                reason = report.get("failure_reason")
                if reason == "motion_gate":
                    self.motion_gate_rejections += 1
                elif reason == "low_projection":
                    self.low_projection_rejections += 1
                current_pose, fallback_reason = self._fallback_pose(previous_pose)
                fallback_used = True
                self.fallbacks_used += 1
            else:
                current_pose = self.current_kf.pose @ np.linalg.inv(relative_pose)

            keyframe_inserted = False
            self.frames_since_keyframe += 1
            if not tracking_failure and self._should_insert_keyframe(report):
                kf = self._make_keyframe(frame.left, frame.right, current_pose, frame.timestamp)
                self.keyframes.append(kf)
                self.current_kf = kf
                self.frames_since_keyframe = 0
                keyframe_inserted = True

            if not tracking_failure or fallback_reason == "constant_velocity":
                try:
                    self.last_velocity = np.linalg.inv(previous_pose) @ current_pose
                except np.linalg.LinAlgError:
                    self.last_velocity = np.eye(4, dtype=np.float64)
                self.last_safe_pose = current_pose.copy()
            previous_pose = current_pose.copy()
            trajectory.append(current_pose.copy())
            self.last_report = report
            self._record_diagnostic(
                frame,
                report,
                tracking_failure=tracking_failure,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                keyframe_inserted=keyframe_inserted,
            )
            runtimes.append((time.perf_counter() - start) * 1000.0)
            self.frames_processed += 1

        return trajectory, runtimes

    def get_diagnostics(self) -> list[dict[str, object]]:
        return [dict(item) for item in self.diagnostics]

    def get_stats(self) -> dict[str, object]:
        valid_counts = np.asarray(
            [item["valid_projected_points"] for item in self.diagnostics if item["valid_projected_points"] > 0],
            dtype=np.float64,
        )
        inlier_values = [finite_float_or_none(item.get("inlier_ratio")) for item in self.diagnostics]
        inlier_ratios = np.asarray([value for value in inlier_values if value is not None], dtype=np.float64)
        pnp_inliers = np.asarray(
            [item["pnp_inliers"] for item in self.diagnostics if item.get("pnp_inliers", 0) > 0],
            dtype=np.float64,
        )
        return {
            "frames_processed": self.frames_processed,
            "keyframes": len(self.keyframes),
            "tracking_failures": self.tracking_failures,
            "fallbacks_used": self.fallbacks_used,
            "motion_gate_rejections": self.motion_gate_rejections,
            "low_projection_rejections": self.low_projection_rejections,
            "relocalization_attempts": 0,
            "relocalization_successes": 0,
            "loop_candidates": self.loop_candidates,
            "loop_closures": self.loop_closures,
            "active_sparse_points": self._active_sparse_points(),
            "mean_tracked_points": float(np.mean(valid_counts)) if len(valid_counts) else 0.0,
            "mean_pnp_inliers": float(np.mean(pnp_inliers)) if len(pnp_inliers) else 0.0,
            "mean_valid_projected_points": float(np.mean(valid_counts)) if len(valid_counts) else 0.0,
            "median_valid_projected_points": float(np.median(valid_counts)) if len(valid_counts) else 0.0,
            "mean_inlier_ratio": float(np.mean(inlier_ratios)) if len(inlier_ratios) else 0.0,
            "loop_closure_enabled": False,
        }


def run_svo_slam(
    data_dir: str,
    seq: str,
    output_path: str | None = None,
    num_points: int = 1500,
) -> tuple[list[np.ndarray], list[float]]:
    loader = KITTIOdometryLoader(data_dir, seq)
    slam = SVOStyleSLAM(loader, num_points=num_points)
    trajectory, runtimes = slam.run()
    out_path = output_path or f"results/svo_slam_seq{int(seq):02d}.txt"
    save_trajectory_kitti(trajectory, out_path)
    return trajectory, runtimes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SVO-style semi-direct stereo visual odometry/SLAM on KITTI")
    parser.add_argument("--data-dir", default="data/kitti_odometry")
    parser.add_argument("--seq", default="00")
    parser.add_argument("--output", default=None)
    parser.add_argument("--points", type=int, default=1500)
    args = parser.parse_args()

    trajectory, runtimes = run_svo_slam(args.data_dir, args.seq, args.output, args.points)
    print(f"SVO-style semi-direct stereo VO/SLAM: {len(trajectory)} frames, avg {np.mean(runtimes):.2f} ms/frame")
