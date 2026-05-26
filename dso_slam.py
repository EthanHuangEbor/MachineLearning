"""
DSO-SLAM complete system: integrates frontend VO + local mapping + loop closing.
Frontend: direct photometric method (high-gradient pixels + photometric error + LM optimization)
Backend: photometric Bundle Adjustment (EnergyFunctional style)
Loop: ORB feature matching + geometric verification + Pose Graph optimization

Comparison baseline: ORB-SLAM2 (orb_slam.py) uses the same backend/loop infrastructure,
only frontend tracking method differs (feature-point PnP vs direct photometric alignment).
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from dataclasses import dataclass
from typing import Optional

from kitti_utils import KITTIOdometryLoader, CameraCalibration, disparity_to_depth, save_trajectory_kitti
from slam_base import KeyFrame, CovisibilityGraph, solve_pose_graph
from local_mapping import LocalMappingConfig
from dso_advanced import InverseDepthActiveWindow, PhotometricCalibrator


# =============================================================================
# Keyframe for DSO (stores photometric data + selected pixels)
# =============================================================================

@dataclass
class DSOKeyFrame:
    id: int
    timestamp: float
    pose: np.ndarray  # Tcw 4x4
    gray: np.ndarray
    depth: np.ndarray
    active_uvs: np.ndarray  # selected high-gradient pixel coords
    active_points_3d: np.ndarray
    active_inv_depths: np.ndarray
    active_intensities: np.ndarray
    affine_gain: float = 1.0
    affine_bias: float = 0.0
    depth_filter: str = "positive_disparity_depth_range"
    tracking_only: bool = False
    bow_vector: Optional[np.ndarray] = None
    observations: list = None

    def __post_init__(self):
        if self.observations is None:
            self.observations = []

    @staticmethod
    def next_id() -> int:
        DSOKeyFrame._counter += 1
        return DSOKeyFrame._counter

DSOKeyFrame._counter = 0


@dataclass
class DSOTrackingConfig:
    max_translation_m: float = 5.0
    max_rotation_deg: float = 15.0
    min_valid_projected_points: int = 30
    min_inlier_ratio: float = 0.15
    residual_inlier_threshold: float = 25.0
    enable_motion_gate: bool = True
    enable_outlier_culling: bool = True
    enable_grid_selection: bool = True
    enable_photometric_refinement: bool = True
    enable_affine_brightness: bool = True
    enable_strict_loop_verification: bool = True
    enable_loop_correction: bool = True
    enable_left_right_depth_check: bool = True
    lr_disparity_tolerance_px: float = 1.5
    min_gradient_magnitude: float = 8.0
    min_valid_projected_ratio: float = 0.35
    enable_residual_p95_gate: bool = True
    max_residual_p95: float = 140.0
    enable_cost_jump_gate: bool = True
    max_cost_jump_ratio: float = 3.0
    enable_lk_consistency_gate: bool = True
    max_lk_translation_delta_m: float = 2.5
    max_lk_rotation_delta_deg: float = 8.0
    enable_joint_window_ba: bool = True
    ba_max_keyframes: int = 5
    ba_max_points_per_keyframe: int = 160
    ba_max_nfev: int = 15
    force_keyframe_after_fallback: bool = True
    max_consecutive_failures: int = 3
    enable_clahe: bool = False
    enable_gradient_normalized_residual: bool = False
    fallback_mode: str = "lk_pnp_then_constant_velocity"


# =============================================================================
# Photometric energy functional (DSO-style cost function)
# =============================================================================

def build_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def select_high_gradient_pixels(
    gray: np.ndarray,
    depth: np.ndarray,
    num_points: int = 1500,
    min_gradient: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select high-gradient pixels for direct method."""
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Mask borders and invalid depth
    border = 10
    magnitude[:border, :] = 0
    magnitude[-border:, :] = 0
    magnitude[:, :border] = 0
    magnitude[:, -border:] = 0

    valid_depth = (depth > 0.5) & (depth < 100.0)
    magnitude[~valid_depth] = 0

    if min_gradient > 0:
        magnitude[magnitude < min_gradient] = 0

    flat = magnitude.ravel()
    valid_indices = np.flatnonzero(flat > 0)
    if len(valid_indices) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, np.empty(0, dtype=int), np.empty(0, dtype=int)
    order = np.argsort(flat[valid_indices])[::-1]
    indices = valid_indices[order[:num_points]]
    rows, cols = np.unravel_index(indices, gray.shape)
    uvs = np.column_stack([cols.astype(np.float32), rows.astype(np.float32)])
    return uvs, cols, rows


def select_grid_high_gradient_pixels(
    gray: np.ndarray,
    depth: np.ndarray,
    num_points: int = 1500,
    grid_rows: int = 12,
    grid_cols: int = 16,
    border: int = 10,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
    min_gradient: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select high-gradient pixels while spreading them over the image."""
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    magnitude[:border, :] = 0
    magnitude[-border:, :] = 0
    magnitude[:, :border] = 0
    magnitude[:, -border:] = 0
    # The current stereo filter uses positive disparity plus a depth range.
    # It does not yet implement a full left-right consistency check.
    valid_depth = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    magnitude[~valid_depth] = 0
    if min_gradient > 0:
        magnitude[magnitude < min_gradient] = 0

    h, w = gray.shape
    points_per_cell = max(1, int(np.ceil(num_points / max(grid_rows * grid_cols, 1))))
    selected: list[np.ndarray] = []

    for r in range(grid_rows):
        y0 = int(round(r * h / grid_rows))
        y1 = int(round((r + 1) * h / grid_rows))
        for c in range(grid_cols):
            x0 = int(round(c * w / grid_cols))
            x1 = int(round((c + 1) * w / grid_cols))
            cell = magnitude[y0:y1, x0:x1]
            valid = np.flatnonzero(cell.ravel() > 0)
            if len(valid) == 0:
                continue
            order = np.argsort(cell.ravel()[valid])[::-1][:points_per_cell]
            cell_indices = valid[order]
            ys, xs = np.unravel_index(cell_indices, cell.shape)
            selected.append(np.column_stack([xs + x0, ys + y0]))

    if not selected:
        return select_high_gradient_pixels(gray, depth, num_points, min_gradient=min_gradient)

    coords = np.vstack(selected)
    if len(coords) > num_points:
        scores = magnitude[coords[:, 1], coords[:, 0]]
        keep = np.argsort(scores)[::-1][:num_points]
        coords = coords[keep]

    cols = coords[:, 0].astype(int)
    rows = coords[:, 1].astype(int)
    uvs = np.column_stack([cols.astype(np.float32), rows.astype(np.float32)])
    return uvs, cols, rows


def select_grid_uniform_high_gradient_pixels(
    gray: np.ndarray,
    depth: np.ndarray,
    num_points: int = 1500,
    grid_rows: int = 8,
    grid_cols: int = 12,
    border: int = 10,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
    min_gradient: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Goal.md-compatible wrapper for grid-uniform DSO point selection."""
    return select_grid_high_gradient_pixels(
        gray,
        depth,
        num_points=num_points,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        border=border,
        min_depth=min_depth,
        max_depth=max_depth,
        min_gradient=min_gradient,
    )


def _create_sgbm(min_disparity: int = 0, num_disparities: int = 128) -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(
        minDisparity=min_disparity, numDisparities=num_disparities, blockSize=7,
        P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_stereo_depth(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_right_check: bool = False,
    lr_tolerance_px: float = 1.5,
) -> np.ndarray:
    stereo = _create_sgbm(0, 128)
    disparity = stereo.compute(left, right).astype(np.float32) / 16.0
    if left_right_check:
        right_stereo = _create_sgbm(-128, 128)
        disparity_right = right_stereo.compute(right, left).astype(np.float32) / 16.0
        h, w = disparity.shape
        xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        ys = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        xr = xs - disparity
        valid = (disparity > 0.1) & (xr >= 0) & (xr < w - 1)
        sampled_right = bilinear_interpolate(disparity_right, xr.astype(np.float32), ys.astype(np.float32))
        consistent = np.abs(disparity + sampled_right) <= lr_tolerance_px
        disparity = disparity.copy()
        disparity[~(valid & consistent)] = 0.0
    return disparity


def bilinear_interpolate(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1

    u0 = np.clip(u0, 0, w - 2)
    u1 = np.clip(u1, 0, w - 1)
    v0 = np.clip(v0, 0, h - 2)
    v1 = np.clip(v1, 0, h - 1)

    du = u - u0
    dv = v - v0
    img_f = image.astype(np.float32)
    return (
        img_f[v0, u0] * (1 - du) * (1 - dv)
        + img_f[v0, u1] * du * (1 - dv)
        + img_f[v1, u0] * (1 - du) * dv
        + img_f[v1, u1] * du * dv
    )


def pose_to_matrix(xi: np.ndarray) -> np.ndarray:
    tx, ty, tz, rx, ry, rz = xi[:6]
    angle = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    if angle < 1e-8:
        rotation = np.eye(3)
    else:
        axis = np.array([rx, ry, rz]) / angle
        K_mat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * K_mat @ K_mat
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = [tx, ty, tz]
    return T


def compute_photometric_residuals(
    xi: np.ndarray,
    points_3d: np.ndarray,
    intensities_ref: np.ndarray,
    curr_pyramid: list[np.ndarray],
    calib: CameraCalibration,
    level: int,
    gradient_normalized: bool = False,
) -> np.ndarray:
    scale = 0.5 ** level
    sf = calib.focal_length * scale
    scx = calib.cx * scale
    scy = calib.cy * scale

    T = pose_to_matrix(xi)
    if len(xi) >= 8:
        affine_gain = float(np.exp(np.clip(xi[6], -1.0, 1.0)))
        affine_bias = float(np.clip(xi[7], -50.0, 50.0))
    else:
        affine_gain = 1.0
        affine_bias = 0.0
    pts_cam = (T[:3, :3] @ points_3d.T).T + T[:3, 3]
    depth = pts_cam[:, 2]
    valid = depth > 0.1

    u_proj = np.where(valid, sf * pts_cam[:, 0] / depth + scx, 0.0)
    v_proj = np.where(valid, sf * pts_cam[:, 1] / depth + scy, 0.0)

    curr_gray = curr_pyramid[level].astype(np.float32)
    h, w = curr_gray.shape
    in_bounds = (u_proj > 0) & (u_proj < w - 1) & (v_proj > 0) & (v_proj < h - 1)
    mask = valid & in_bounds

    residuals = np.full(len(points_3d), 50.0, dtype=np.float32)
    if mask.any():
        interp = bilinear_interpolate(curr_gray, u_proj[mask], v_proj[mask])
        valid_residuals = intensities_ref[mask] - (affine_gain * interp + affine_bias)
        if gradient_normalized:
            grad_x = cv2.Sobel(curr_gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(curr_gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_values = bilinear_interpolate(grad_mag, u_proj[mask], v_proj[mask])
            valid_residuals = valid_residuals / np.sqrt(1.0 + grad_values / 25.0)
        residuals[mask] = valid_residuals

    return residuals


def rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def relative_motion_stats(transform: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(transform[:3, 3]))
    rotation = rotation_angle_deg(transform[:3, :3])
    return translation, rotation


def finite_float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def compute_photometric_residual_stats(
    xi: np.ndarray,
    points_3d: np.ndarray,
    intensities_ref: np.ndarray,
    curr_pyramid: list[np.ndarray],
    calib: CameraCalibration,
    level: int,
    inlier_threshold: float = 25.0,
    gradient_normalized: bool = False,
) -> dict[str, float | int]:
    if len(points_3d) == 0:
        return {
            "valid_projected_points": 0,
            "valid_projected_ratio": 0.0,
            "residual_mean": float("nan"),
            "residual_median": float("nan"),
            "residual_p95": float("nan"),
            "inlier_ratio": 0.0,
            "affine_gain": 1.0,
            "affine_bias": 0.0,
        }

    scale = 0.5 ** level
    sf = calib.focal_length * scale
    scx = calib.cx * scale
    scy = calib.cy * scale

    transform = pose_to_matrix(xi)
    if len(xi) >= 8:
        affine_gain = float(np.exp(np.clip(xi[6], -1.0, 1.0)))
        affine_bias = float(np.clip(xi[7], -50.0, 50.0))
    else:
        affine_gain = 1.0
        affine_bias = 0.0

    pts_cam = (transform[:3, :3] @ points_3d.T).T + transform[:3, 3]
    depth = pts_cam[:, 2]
    positive_depth = depth > 0.1
    u_proj = np.where(positive_depth, sf * pts_cam[:, 0] / np.maximum(depth, 1e-8) + scx, 0.0)
    v_proj = np.where(positive_depth, sf * pts_cam[:, 1] / np.maximum(depth, 1e-8) + scy, 0.0)

    curr_gray = curr_pyramid[level].astype(np.float32)
    h, w = curr_gray.shape
    in_bounds = (u_proj > 0) & (u_proj < w - 1) & (v_proj > 0) & (v_proj < h - 1)
    mask = positive_depth & in_bounds
    valid_count = int(mask.sum())
    if valid_count == 0:
        return {
            "valid_projected_points": 0,
            "valid_projected_ratio": 0.0,
            "residual_mean": float("nan"),
            "residual_median": float("nan"),
            "residual_p95": float("nan"),
            "inlier_ratio": 0.0,
            "affine_gain": affine_gain,
            "affine_bias": affine_bias,
        }

    interp = bilinear_interpolate(curr_gray, u_proj[mask], v_proj[mask])
    residuals = intensities_ref[mask] - (affine_gain * interp + affine_bias)
    if gradient_normalized:
        grad_x = cv2.Sobel(curr_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(curr_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_values = bilinear_interpolate(grad_mag, u_proj[mask], v_proj[mask])
        residuals = residuals / np.sqrt(1.0 + grad_values / 25.0)
    abs_residuals = np.abs(residuals)
    return {
        "valid_projected_points": valid_count,
        "valid_projected_ratio": float(valid_count / max(len(points_3d), 1)),
        "residual_mean": float(np.mean(abs_residuals)),
        "residual_median": float(np.median(abs_residuals)),
        "residual_p95": float(np.percentile(abs_residuals, 95)),
        "inlier_ratio": float(np.mean(abs_residuals <= inlier_threshold)),
        "affine_gain": affine_gain,
        "affine_bias": affine_bias,
    }


def matrix_to_pose_xi(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 SE(3) to 6-dof se(3) parameterization."""
    t = T[:3, 3]
    R = T[:3, :3]
    trace = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(trace)
    if angle < 1e-8:
        omega = np.zeros(3)
    else:
        log_R = angle / (2 * np.sin(angle)) * (R - R.T)
        omega = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])
    return np.concatenate([t, omega])


# =============================================================================
# DSO Tracking (frontend)
# =============================================================================

class DSOTacker:
    """Direct sparse tracking for DSO-SLAM."""

    def __init__(
        self,
        calib: CameraCalibration,
        num_active_points: int = 1500,
        pyramid_levels: int = 4,
        photometric_calibrator: Optional[PhotometricCalibrator] = None,
        config: Optional[DSOTrackingConfig] = None,
    ):
        self.calib = calib
        self.num_active_points = num_active_points
        self.pyramid_levels = pyramid_levels
        self.config = config or DSOTrackingConfig()
        self.keyframe_flow_threshold = 20.0
        self.photometric_calibrator = photometric_calibrator or PhotometricCalibrator()
        self.last_tracking_success = False
        self.last_tracking_cost = float("nan")
        self.last_tracking_report = self._empty_report("not_run")
        self.last_lk_pose = np.eye(4, dtype=np.float64)
        self.last_lk_valid = False
        self.last_lk_inliers = 0
        self.last_good_cost_per_point: float | None = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        corrected = self.photometric_calibrator.correct(image)
        if self.config.enable_clahe:
            corrected = self._clahe.apply(corrected)
        return corrected

    def _empty_report(self, failure_reason: str | None = None) -> dict[str, float | int | str | bool | None]:
        return {
            "tracking_cost": None,
            "tracking_cost_per_point": None,
            "valid_projected_points": 0,
            "valid_projected_ratio": 0.0,
            "residual_mean": None,
            "residual_median": None,
            "residual_p95": None,
            "inlier_ratio": None,
            "relative_translation_m": None,
            "relative_rotation_deg": None,
            "affine_gain": 1.0,
            "affine_bias": 0.0,
            "brightness_mode": "affine" if self.config.enable_affine_brightness else "fixed",
            "depth_filter": "left_right_consistency" if self.config.enable_left_right_depth_check else "positive_disparity_depth_range",
            "success": False,
            "failure_reason": failure_reason,
            "lk_inliers": 0,
            "lk_photometric_translation_delta_m": None,
            "lk_photometric_rotation_delta_deg": None,
        }

    def make_keyframe(
        self,
        left: np.ndarray,
        right: np.ndarray,
        pose: np.ndarray,
        timestamp: float | None = 0.0,
        tracking_only: bool = False,
    ) -> DSOKeyFrame:
        left_corr = self._preprocess(left)
        right_corr = self._preprocess(right)
        disparity = compute_stereo_depth(
            left_corr,
            right_corr,
            left_right_check=self.config.enable_left_right_depth_check,
            lr_tolerance_px=self.config.lr_disparity_tolerance_px,
        )
        depth = disparity_to_depth(disparity, self.calib)
        depth_filter = "left_right_consistency" if self.config.enable_left_right_depth_check else "positive_disparity_depth_range"
        if self.config.enable_grid_selection:
            uvs, cols, rows = select_grid_uniform_high_gradient_pixels(
                left_corr,
                depth,
                self.num_active_points,
                min_gradient=self.config.min_gradient_magnitude,
            )
        else:
            uvs, cols, rows = select_high_gradient_pixels(
                left_corr,
                depth,
                self.num_active_points,
                min_gradient=self.config.min_gradient_magnitude,
            )

        u = uvs[:, 0].astype(np.float32)
        v = uvs[:, 1].astype(np.float32)
        z = bilinear_interpolate(depth, u, v)
        valid = np.isfinite(z) & (z > 0.5) & (z < 80.0)
        uvs = uvs[valid]
        z = z[valid]

        x = (uvs[:, 0] - self.calib.cx) * z / self.calib.focal_length
        y = (uvs[:, 1] - self.calib.cy) * z / self.calib.focal_length
        pts_3d = np.column_stack([x, y, z])
        inv_depths = 1.0 / np.maximum(z, 1e-8)

        intensities = bilinear_interpolate(left_corr.astype(np.float32), uvs[:, 0], uvs[:, 1])

        kf = DSOKeyFrame(
            id=DSOKeyFrame.next_id(),
            timestamp=float(timestamp or 0.0),
            pose=pose,
            gray=left_corr.copy(),
            depth=depth.copy(),
            active_uvs=uvs,
            active_points_3d=pts_3d,
            active_inv_depths=inv_depths,
            active_intensities=intensities,
            depth_filter=depth_filter,
            tracking_only=tracking_only,
        )
        return kf

    def _initial_pose_lk_pnp(self, ref_kf: DSOKeyFrame, curr_gray: np.ndarray) -> np.ndarray:
        """Initialize direct alignment with sparse optical-flow correspondences."""
        self.last_lk_pose = np.eye(4, dtype=np.float64)
        self.last_lk_valid = False
        self.last_lk_inliers = 0
        if len(ref_kf.active_uvs) < 20:
            return np.eye(4, dtype=np.float64)

        max_points = min(600, len(ref_kf.active_uvs))
        step = max(1, len(ref_kf.active_uvs) // max_points)
        indices = np.arange(0, len(ref_kf.active_uvs), step)[:max_points]
        prev_pts = ref_kf.active_uvs[indices].reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(ref_kf.gray, curr_gray, prev_pts, None)
        if next_pts is None or status is None:
            return np.eye(4, dtype=np.float64)

        mask = status.ravel() > 0
        object_pts = ref_kf.active_points_3d[indices][mask].astype(np.float32)
        image_pts = next_pts.reshape(-1, 2)[mask].astype(np.float32)
        if len(object_pts) < 6:
            return np.eye(4, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_pts,
            image_pts,
            self.calib.k_left,
            None,
            reprojectionError=4.0,
            iterationsCount=100,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None or len(inliers) < 12:
            return np.eye(4, dtype=np.float64)

        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec.reshape(3)
        self.last_lk_pose = transform.copy()
        self.last_lk_valid = True
        self.last_lk_inliers = int(len(inliers))
        return transform

    def estimate_pose(self, ref_kf: DSOKeyFrame, curr_gray: np.ndarray) -> np.ndarray:
        """Estimate relative pose via photometric alignment with multi-level LM."""
        self.last_tracking_success = False
        self.last_tracking_cost = float("nan")
        self.last_tracking_report = self._empty_report("not_run")
        curr_gray = self._preprocess(curr_gray)
        if len(ref_kf.active_points_3d) < 20:
            self.last_tracking_report = self._empty_report("insufficient_points")
            return np.eye(4, dtype=np.float64)

        curr_pyramid = build_pyramid(curr_gray, self.pyramid_levels)
        ref_pyramid = build_pyramid(ref_kf.gray, self.pyramid_levels)

        initial_pose = self._initial_pose_lk_pnp(ref_kf, curr_gray)
        if self.config.enable_affine_brightness:
            xi = np.concatenate([matrix_to_pose_xi(initial_pose), np.zeros(2)])
        else:
            xi = matrix_to_pose_xi(initial_pose)
        optimization_success = self.last_lk_valid

        if not self.config.enable_photometric_refinement and not self.last_lk_valid:
            self.last_tracking_report = self._empty_report("lk_pnp_failed")
            return initial_pose

        if self.config.enable_photometric_refinement:
            for level in range(self.pyramid_levels - 1, -1, -1):
                scale = 0.5 ** level
                ref_intensities = bilinear_interpolate(
                    ref_pyramid[level].astype(np.float32),
                    ref_kf.active_uvs[:, 0] * scale,
                    ref_kf.active_uvs[:, 1] * scale,
                )

                def residual_fn(x_opt):
                    residuals = compute_photometric_residuals(
                        x_opt, ref_kf.active_points_3d, ref_intensities,
                        curr_pyramid, self.calib, level,
                        gradient_normalized=self.config.enable_gradient_normalized_residual,
                    )
                    if self.config.enable_outlier_culling:
                        threshold = self.config.residual_inlier_threshold * 2.0
                        residuals = np.clip(residuals, -threshold, threshold)
                    return residuals

                result = least_squares(
                    residual_fn,
                    xi,
                    method="trf",
                    loss="huber",
                    f_scale=10.0,
                    max_nfev=30,
                    ftol=1e-4,
                )
                if not result.success or not np.all(np.isfinite(result.x)):
                    optimization_success = False
                    break
                xi = result.x
                optimization_success = True
                self.last_tracking_cost = float(result.cost)

        relative_pose = pose_to_matrix(xi)
        translation_m, rotation_deg = relative_motion_stats(relative_pose)
        residual_stats = compute_photometric_residual_stats(
            xi,
            ref_kf.active_points_3d,
            ref_kf.active_intensities,
            curr_pyramid,
            self.calib,
            level=0,
            inlier_threshold=self.config.residual_inlier_threshold,
            gradient_normalized=self.config.enable_gradient_normalized_residual,
        )

        valid_count = int(residual_stats["valid_projected_points"])
        valid_ratio = float(residual_stats["valid_projected_ratio"])
        cost_per_point = None
        if np.isfinite(self.last_tracking_cost) and valid_count > 0:
            cost_per_point = float(self.last_tracking_cost / valid_count)

        lk_translation_delta = None
        lk_rotation_delta = None
        if self.last_lk_valid:
            try:
                lk_delta = np.linalg.inv(self.last_lk_pose) @ relative_pose
                lk_translation_delta, lk_rotation_delta = relative_motion_stats(lk_delta)
            except np.linalg.LinAlgError:
                lk_translation_delta, lk_rotation_delta = float("inf"), float("inf")

        failure_reason = None
        if not optimization_success:
            failure_reason = "photometric_optimization_failed"
        if failure_reason is None and self.config.enable_motion_gate and (
            translation_m > self.config.max_translation_m
            or rotation_deg > self.config.max_rotation_deg
        ):
            failure_reason = "motion_gate"
        elif failure_reason is None and (
            valid_count < self.config.min_valid_projected_points
            or valid_ratio < self.config.min_valid_projected_ratio
        ):
            failure_reason = "low_projection"
        elif (
            failure_reason is None
            and self.config.enable_residual_p95_gate
            and finite_float_or_none(residual_stats["residual_p95"]) is not None
            and float(residual_stats["residual_p95"]) > self.config.max_residual_p95
        ):
            failure_reason = "high_residual_p95"
        elif failure_reason is None and residual_stats["inlier_ratio"] < self.config.min_inlier_ratio:
            failure_reason = "low_inlier_ratio"
        elif (
            failure_reason is None
            and self.config.enable_cost_jump_gate
            and self.last_good_cost_per_point is not None
            and cost_per_point is not None
            and cost_per_point > self.config.max_cost_jump_ratio * self.last_good_cost_per_point
        ):
            failure_reason = "cost_jump"
        elif (
            failure_reason is None
            and self.config.enable_lk_consistency_gate
            and self.last_lk_valid
            and (
                lk_translation_delta is not None
                and (
                    lk_translation_delta > self.config.max_lk_translation_delta_m
                    or lk_rotation_delta > self.config.max_lk_rotation_delta_deg
                )
            )
        ):
            failure_reason = "lk_photometric_disagreement"

        self.last_tracking_success = failure_reason is None
        if self.last_tracking_success and cost_per_point is not None:
            self.last_good_cost_per_point = cost_per_point
        self.last_tracking_report = {
            "tracking_cost": finite_float_or_none(self.last_tracking_cost),
            "tracking_cost_per_point": finite_float_or_none(cost_per_point),
            "valid_projected_points": valid_count,
            "valid_projected_ratio": finite_float_or_none(valid_ratio),
            "residual_mean": finite_float_or_none(residual_stats["residual_mean"]),
            "residual_median": finite_float_or_none(residual_stats["residual_median"]),
            "residual_p95": finite_float_or_none(residual_stats["residual_p95"]),
            "inlier_ratio": finite_float_or_none(residual_stats["inlier_ratio"]),
            "relative_translation_m": finite_float_or_none(translation_m),
            "relative_rotation_deg": finite_float_or_none(rotation_deg),
            "affine_gain": finite_float_or_none(residual_stats["affine_gain"]),
            "affine_bias": finite_float_or_none(residual_stats["affine_bias"]),
            "brightness_mode": "affine" if self.config.enable_affine_brightness else "fixed",
            "depth_filter": ref_kf.depth_filter,
            "success": self.last_tracking_success,
            "failure_reason": failure_reason,
            "lk_inliers": self.last_lk_inliers,
            "lk_photometric_translation_delta_m": finite_float_or_none(lk_translation_delta),
            "lk_photometric_rotation_delta_deg": finite_float_or_none(lk_rotation_delta),
        }
        return relative_pose

    def should_update_keyframe(self, ref_kf: DSOKeyFrame, curr_gray: np.ndarray) -> bool:
        """Use ORB optical flow to measure frame-to-frame motion."""
        curr_gray = self._preprocess(curr_gray)
        orb = cv2.ORB_create(nfeatures=200)
        kp1, desc1 = orb.detectAndCompute(ref_kf.gray, None)
        kp2, desc2 = orb.detectAndCompute(curr_gray, None)
        if desc1 is None or desc2 is None or not kp1 or not kp2:
            return True

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = sorted(matcher.match(desc1, desc2), key=lambda m: m.distance)
        if not matches:
            return True

        flows = []
        for m in matches[:50]:
            dx = kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]
            dy = kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]
            flows.append(np.sqrt(dx ** 2 + dy ** 2))

        return float(np.mean(flows)) > self.keyframe_flow_threshold


# =============================================================================
# DSO Local Mapping (backend — photometric BA)
# =============================================================================

class DSOLocalMapping:
    """Bounded DSO-style local mapping with a sparse active photometric window."""

    def __init__(
        self,
        calib: CameraCalibration,
        config: Optional[LocalMappingConfig] = None,
        tracking_config: Optional[DSOTrackingConfig] = None,
    ):
        self.calib = calib
        self.config = config or LocalMappingConfig()
        self.tracking_config = tracking_config or DSOTrackingConfig()
        self.active_keyframes: list[DSOKeyFrame] = []
        self.new_kf_queue: list[DSOKeyFrame] = []
        self.covisibility: Optional[CovisibilityGraph] = None
        self.active_window = InverseDepthActiveWindow(
            calib,
            max_keyframes=self.tracking_config.ba_max_keyframes,
        )
        self.marginalizations = 0
        self.photometric_ba_runs = 0
        self.ba_runs = 0
        self.ba_accepted = 0
        self.ba_rejected = 0
        self.active_points_culled = 0
        self._keyframes_since_ba = 0

    def set_covisibility(self, cov: CovisibilityGraph) -> None:
        self.covisibility = cov

    def insert_keyframe(self, kf: DSOKeyFrame) -> None:
        if not kf.tracking_only:
            self.new_kf_queue.append(kf)

    def reset(self) -> None:
        self.active_keyframes = []
        self.new_kf_queue = []
        self.active_window.reset()
        self._keyframes_since_ba = 0

    def run(self) -> list[DSOKeyFrame]:
        """Main loop: process new keyframes with photometric BA."""
        if not self.new_kf_queue:
            return []

        processed = []
        while self.new_kf_queue:
            kf = self.new_kf_queue.pop(0)
            self._process_keyframe(kf)
            processed.append(kf)

        return processed

    def _process_keyframe(self, kf: DSOKeyFrame) -> None:
        """Add keyframe to active pool, marginalize old, run photometric BA."""
        self.active_keyframes.append(kf)
        marginalized = self.active_window.add_keyframe(kf)
        if marginalized is not None:
            self.marginalizations += 1

        if len(self.active_keyframes) > self.tracking_config.ba_max_keyframes:
            self.active_keyframes.pop(0)

        self._keyframes_since_ba += 1
        if self._keyframes_since_ba >= 2:
            self._run_photometric_ba(kf)
            self._keyframes_since_ba = 0

    def _run_photometric_ba(self, kf: DSOKeyFrame) -> None:
        """Photometric Bundle Adjustment across active keyframes."""
        if len(self.active_keyframes) < 2:
            return
        if not self.tracking_config.enable_joint_window_ba:
            self._run_pairwise_photometric_ba(kf)
            return
        self._run_joint_window_ba()

    def _run_pairwise_photometric_ba(self, kf: DSOKeyFrame) -> None:
        """Fallback bounded pairwise BA used by ablations."""
        self.ba_runs += 1
        ref_kf = self.active_keyframes[0]
        xi_pose = matrix_to_pose_xi(np.linalg.inv(kf.pose) @ ref_kf.pose)
        xi_init = np.concatenate([xi_pose, np.zeros(2)]) if self.tracking_config.enable_affine_brightness else xi_pose
        curr_pyramid = build_pyramid(kf.gray, 4)
        ref_points = self.active_window.points_3d_for(ref_kf)
        ref_intensities = self.active_window.intensities_for(ref_kf)

        def residual_with_prior(xi: np.ndarray) -> np.ndarray:
            photometric = compute_photometric_residuals(
                xi, ref_points, ref_intensities,
                curr_pyramid, self.calib, level=0,
                gradient_normalized=self.tracking_config.enable_gradient_normalized_residual,
            )
            prior = self.active_window.prior.residual(kf.id, xi, xi_init)
            return np.concatenate([photometric, prior])

        initial = residual_with_prior(xi_init)
        result = least_squares(
            residual_with_prior,
            xi_init, method="trf", loss="huber", f_scale=10.0,
            max_nfev=self.tracking_config.ba_max_nfev, ftol=1e-6
        )

        final = residual_with_prior(result.x) if np.all(np.isfinite(result.x)) else initial
        if np.all(np.isfinite(result.x)) and 0.5 * float(np.sum(final ** 2)) < 0.5 * float(np.sum(initial ** 2)):
            T_cur_ref = pose_to_matrix(result.x)
            kf.pose = ref_kf.pose @ np.linalg.inv(T_cur_ref)
            if len(result.x) >= 8:
                kf.affine_gain = float(np.exp(np.clip(result.x[6], -1.0, 1.0)))
                kf.affine_bias = float(np.clip(result.x[7], -50.0, 50.0))
            self.active_window.refine_inverse_depths(ref_kf, kf, T_cur_ref)
            self.active_points_culled += self.active_window.cull_bad_points()
            self.ba_accepted += 1
            self.photometric_ba_runs += 1
        else:
            self.ba_rejected += 1

    def _run_joint_window_ba(self) -> None:
        window = [kf for kf in self.active_keyframes[-self.tracking_config.ba_max_keyframes:] if not kf.tracking_only]
        if len(window) < 2:
            return

        residual_specs: list[tuple[DSOKeyFrame, DSOKeyFrame, object, int]] = []
        x0: list[float] = []
        pose_slices: dict[int, slice] = {}
        affine_slices: dict[int, slice] = {}
        base_poses = {kf.id: kf.pose.copy() for kf in window}
        anchor_id = window[0].id

        for pose_kf in window[1:]:
            start = len(x0)
            x0.extend([0.0] * 6)
            pose_slices[pose_kf.id] = slice(start, start + 6)

        if self.tracking_config.enable_affine_brightness:
            for pose_kf in window[1:]:
                start = len(x0)
                x0.extend([float(np.log(max(pose_kf.affine_gain, 1e-6))), float(pose_kf.affine_bias)])
                affine_slices[pose_kf.id] = slice(start, start + 2)

        for ref_kf, target_kf in zip(window[:-1], window[1:]):
            # Python finite-difference BA needs an additional runtime guard beyond
            # the public budget, otherwise 100/300-frame runs become impractical.
            effective_point_budget = min(self.tracking_config.ba_max_points_per_keyframe, 40)
            points = self.active_window.sample_points(ref_kf, effective_point_budget)
            for point in points:
                residual_specs.append((ref_kf, target_kf, point, -1))

        if not residual_specs:
            return

        x0_array = np.asarray(x0, dtype=np.float64)
        gradients: dict[int, np.ndarray] = {}
        if self.tracking_config.enable_gradient_normalized_residual:
            for kf in window:
                grad_x = cv2.Sobel(kf.gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(kf.gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
                gradients[kf.id] = np.sqrt(grad_x ** 2 + grad_y ** 2)

        def pose_for(kf: DSOKeyFrame, params: np.ndarray) -> np.ndarray:
            if kf.id == anchor_id:
                return base_poses[kf.id]
            return pose_to_matrix(params[pose_slices[kf.id]]) @ base_poses[kf.id]

        def affine_for(kf: DSOKeyFrame, params: np.ndarray) -> tuple[float, float]:
            slc = affine_slices.get(kf.id)
            if slc is None:
                return 1.0, 0.0
            return (
                float(np.exp(np.clip(params[slc.start], -1.0, 1.0))),
                float(np.clip(params[slc.start + 1], -50.0, 50.0)),
            )

        def residual_fn(params: np.ndarray) -> np.ndarray:
            poses = {kf.id: pose_for(kf, params) for kf in window}
            pair_transforms: dict[tuple[int, int], np.ndarray] = {}
            residuals = np.empty(len(residual_specs), dtype=np.float64)
            for row, (ref_kf, target_kf, point, depth_index) in enumerate(residual_specs):
                key = (ref_kf.id, target_kf.id)
                if key not in pair_transforms:
                    try:
                        pair_transforms[key] = np.linalg.inv(poses[target_kf.id]) @ poses[ref_kf.id]
                    except np.linalg.LinAlgError:
                        pair_transforms[key] = np.eye(4, dtype=np.float64)
                inv_depth = float(np.clip(point.inverse_depth, 1e-4, 2.0))
                z = 1.0 / max(inv_depth, 1e-8)
                xyz = np.array([
                    (point.uv[0] - self.calib.cx) * z / self.calib.focal_length,
                    (point.uv[1] - self.calib.cy) * z / self.calib.focal_length,
                    z,
                ])
                pt = pair_transforms[key][:3, :3] @ xyz + pair_transforms[key][:3, 3]
                if pt[2] <= 0.1:
                    residuals[row] = 50.0 * point.weight
                    continue
                u = self.calib.focal_length * pt[0] / pt[2] + self.calib.cx
                v = self.calib.focal_length * pt[1] / pt[2] + self.calib.cy
                h, w = target_kf.gray.shape
                if not (1 <= u < w - 2 and 1 <= v < h - 2):
                    residuals[row] = 50.0 * point.weight
                    continue
                gain, bias = affine_for(target_kf, params)
                intensity = bilinear_interpolate(target_kf.gray.astype(np.float32), np.array([u]), np.array([v]))[0]
                residual = point.intensity - (gain * intensity + bias)
                if self.tracking_config.enable_gradient_normalized_residual:
                    grad = bilinear_interpolate(gradients[target_kf.id], np.array([u]), np.array([v]))[0]
                    residual = residual / np.sqrt(1.0 + grad / 25.0)
                residuals[row] = residual * point.weight
            if self.tracking_config.enable_outlier_culling:
                threshold = self.tracking_config.residual_inlier_threshold * 2.0
                residuals = np.clip(residuals, -threshold, threshold)
            return residuals

        jac_sparsity = lil_matrix((len(residual_specs), len(x0_array)), dtype=int)
        for row, (ref_kf, target_kf, _point, depth_index) in enumerate(residual_specs):
            ref_slice = pose_slices.get(ref_kf.id)
            if ref_slice is not None:
                jac_sparsity[row, ref_slice] = 1
            target_slice = pose_slices.get(target_kf.id)
            if target_slice is not None:
                jac_sparsity[row, target_slice] = 1
            affine_slice = affine_slices.get(target_kf.id)
            if affine_slice is not None:
                jac_sparsity[row, affine_slice] = 1
            if depth_index >= 0:
                jac_sparsity[row, depth_index] = 1

        self.ba_runs += 1
        initial = residual_fn(x0_array)
        initial_cost = 0.5 * float(np.sum(initial ** 2))
        result = least_squares(
            residual_fn,
            x0_array,
            method="trf",
            loss="huber",
            f_scale=10.0,
            max_nfev=min(self.tracking_config.ba_max_nfev, 5),
            ftol=1e-5,
            jac_sparsity=jac_sparsity.tocsr(),
        )
        final_residuals = residual_fn(result.x) if np.all(np.isfinite(result.x)) else initial
        final_cost = 0.5 * float(np.sum(final_residuals ** 2))
        if (
            not np.all(np.isfinite(result.x))
            or final_cost >= initial_cost
            or not self._joint_ba_motion_is_plausible(window, pose_for, result.x)
        ):
            self.ba_rejected += 1
            return

        for pose_kf in window[1:]:
            pose_kf.pose = pose_for(pose_kf, result.x)
            if pose_kf.id in affine_slices:
                pose_kf.affine_gain, pose_kf.affine_bias = affine_for(pose_kf, result.x)

        for ref_kf, target_kf in zip(window[:-1], window[1:]):
            try:
                target_ref = np.linalg.inv(target_kf.pose) @ ref_kf.pose
            except np.linalg.LinAlgError:
                continue
            self.active_window.refine_inverse_depths(ref_kf, target_kf, target_ref, max_step=0.01)

        for row, (_ref_kf, _target_kf, point, _depth_index) in enumerate(residual_specs):
            raw_residual = float(final_residuals[row] / max(point.weight, 1e-6))
            point.last_residual = raw_residual
            point.observations += 1
            if abs(raw_residual) > self.tracking_config.max_residual_p95:
                point.bad_count += 1
            else:
                point.bad_count = max(0, point.bad_count - 1)
            point.weight = float(np.clip(point.weight * 0.95 + 0.05 * np.exp(-abs(raw_residual) / 50.0), 0.01, 1.0))

        self.active_points_culled += self.active_window.cull_bad_points(max_residual=self.tracking_config.max_residual_p95)
        self.ba_accepted += 1
        self.photometric_ba_runs += 1

    def _joint_ba_motion_is_plausible(self, window, pose_for, params: np.ndarray) -> bool:
        poses = {kf.id: pose_for(kf, params) for kf in window}
        for ref_kf, target_kf in zip(window[:-1], window[1:]):
            try:
                relative = np.linalg.inv(poses[target_kf.id]) @ poses[ref_kf.id]
            except np.linalg.LinAlgError:
                return False
            translation_m, rotation_deg = relative_motion_stats(relative)
            if translation_m > self.tracking_config.max_translation_m * 2.0:
                return False
            if rotation_deg > self.tracking_config.max_rotation_deg * 2.0:
                return False
        return True

    def update_covisibility(self, kf: DSOKeyFrame) -> None:
        if self.covisibility:
            self.covisibility.add_keyframe(
                KeyFrame(id=kf.id, timestamp=kf.timestamp, pose=kf.pose,
                         left_image=kf.gray, descriptors=None)
            )


# =============================================================================
# DSO Loop Detector (uses ORB feature matching)
# =============================================================================

@dataclass
class DSOLoopCandidate:
    keyframe_id: int
    matches: int
    inliers: int
    inlier_ratio: float


class DSOLoopDetector:
    """Loop detection via ORB feature matching (DSO doesn't have native loop detection)."""

    def __init__(self, calib: CameraCalibration, min_matches: int = 30, ransac_threshold: float = 2.0):
        self.calib = calib
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.keyframes: list[DSOKeyFrame] = []
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bow_history: list[tuple] = []
        self.loop_edges: list[tuple[int, int, np.ndarray]] = []
        self._feature_cache: dict[int, tuple[list, np.ndarray | None]] = {}

    def add_keyframe(self, kf: DSOKeyFrame) -> Optional[DSOLoopCandidate]:
        """Add keyframe and look for loop candidates."""
        if kf.tracking_only:
            return None
        self.keyframes.append(kf)
        self.bow_history.append((kf.id, kf.timestamp, kf.pose))

        if len(self.bow_history) < 35:
            return None

        return self._find_loop_candidate(kf)

    def _features(self, kf: DSOKeyFrame) -> tuple[list, np.ndarray | None]:
        cached = self._feature_cache.get(kf.id)
        if cached is not None:
            return cached
        features = self.orb.detectAndCompute(kf.gray, None)
        self._feature_cache[kf.id] = features
        return features

    def _find_loop_candidate(self, kf: DSOKeyFrame) -> Optional[DSOLoopCandidate]:
        """Use ORB feature matching to find loop candidates."""
        best_candidate: DSOLoopCandidate | None = None
        kp1, desc1 = self._features(kf)
        if desc1 is None or not kp1:
            return None

        skip_recent = min(30, len(self.keyframes) - 1)

        for prev_kf in self.keyframes[:-skip_recent]:
            if prev_kf.id == kf.id:
                continue

            kp2, desc2 = self._features(prev_kf)
            if desc2 is None or not kp2:
                continue

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            raw_matches = matcher.knnMatch(desc1, desc2, k=2)
            matches = []
            for candidates in raw_matches:
                if len(candidates) < 2:
                    continue
                best, second = candidates
                if best.distance < 0.75 * second.distance:
                    matches.append(best)

            if len(matches) < self.min_matches:
                continue

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            _, inlier_mask = cv2.findEssentialMat(
                pts1,
                pts2,
                self.calib.k_left,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self.ransac_threshold,
            )
            if inlier_mask is None:
                continue
            inliers = int(inlier_mask.ravel().sum())
            inlier_ratio = float(inliers / max(len(matches), 1))

            if inliers < self.min_matches or inlier_ratio < 0.35:
                continue

            if best_candidate is None or inliers > best_candidate.inliers:
                best_candidate = DSOLoopCandidate(
                    keyframe_id=prev_kf.id,
                    matches=len(matches),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                )

        return best_candidate


# =============================================================================
# DSO-SLAM Complete System
# =============================================================================

class DSOSLAM:
    """
    Complete DSO-SLAM system:
    - Tracking: direct sparse photometric alignment
    - Backend: photometric BA on active keyframes
    - Loop: ORB feature matching + pose graph optimization
    """

    def __init__(
        self,
        loader: KITTIOdometryLoader,
        num_active_points: int = 1500,
        pyramid_levels: int = 4,
        tracking_config: Optional[DSOTrackingConfig] = None,
    ):
        self.loader = loader
        self.calib = loader.calibration

        self.tracking_config = tracking_config or DSOTrackingConfig()
        self.photometric_calibrator = PhotometricCalibrator(gamma=1.0, vignette_strength=0.08)
        self.tracker = DSOTacker(
            self.calib,
            num_active_points,
            pyramid_levels,
            self.photometric_calibrator,
            self.tracking_config,
        )

        lm_config = LocalMappingConfig(max_keyframes=self.tracking_config.ba_max_keyframes)
        self.local_mapping = DSOLocalMapping(self.calib, lm_config, self.tracking_config)
        self.covisibility = CovisibilityGraph()
        self.local_mapping.set_covisibility(self.covisibility)

        self.loop_detector = DSOLoopDetector(self.calib)

        self.keyframes: list[DSOKeyFrame] = []
        self.current_kf: Optional[DSOKeyFrame] = None
        self.trajectory: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
        self.frames_processed = 0
        self.tracking_failures = 0
        self.fallbacks_used = 0
        self.motion_gate_rejections = 0
        self.low_projection_rejections = 0
        self.residual_p95_gate_rejections = 0
        self.cost_jump_rejections = 0
        self.lk_consistency_rejections = 0
        self.consecutive_failures = 0
        self.reinitializations = 0
        self.loop_candidates = 0
        self.loop_verified = 0
        self.loop_corrections_applied = 0
        self.loop_closures = 0
        self.diagnostics: list[dict[str, object]] = []
        self.last_safe_pose = np.eye(4, dtype=np.float64)
        self.last_velocity = np.eye(4, dtype=np.float64)

    def _active_inverse_depth_count(self) -> int:
        return int(sum(len(points) for points in self.local_mapping.active_window.points_by_kf.values()))

    def _motion_passes_gate(self, relative_pose: np.ndarray) -> bool:
        if not self.tracking_config.enable_motion_gate:
            return True
        translation_m, rotation_deg = relative_motion_stats(relative_pose)
        return (
            translation_m <= self.tracking_config.max_translation_m
            and rotation_deg <= self.tracking_config.max_rotation_deg
        )

    def _fallback_pose(self, previous_pose: np.ndarray) -> tuple[np.ndarray, str]:
        mode = self.tracking_config.fallback_mode
        if mode in {"lk_pnp_then_constant_velocity", "lk_pnp"}:
            if self.tracker.last_lk_valid and self._motion_passes_gate(self.tracker.last_lk_pose):
                return self.current_kf.pose @ np.linalg.inv(self.tracker.last_lk_pose), "lk_pnp"

        if mode in {"lk_pnp_then_constant_velocity", "constant_velocity"}:
            if self._motion_passes_gate(self.last_velocity):
                return previous_pose @ self.last_velocity, "constant_velocity"

        return self.last_safe_pose.copy(), "last_safe_pose"

    def _count_rejection(self, reason: object) -> None:
        if reason == "motion_gate":
            self.motion_gate_rejections += 1
        elif reason == "low_projection":
            self.low_projection_rejections += 1
        elif reason == "high_residual_p95":
            self.residual_p95_gate_rejections += 1
        elif reason == "cost_jump":
            self.cost_jump_rejections += 1
        elif reason == "lk_photometric_disagreement":
            self.lk_consistency_rejections += 1

    def _refresh_tracking_keyframe(self, frame, pose: np.ndarray, *, tracking_only: bool) -> DSOKeyFrame:
        new_kf = self.tracker.make_keyframe(
            frame.left,
            frame.right,
            pose,
            frame.timestamp,
            tracking_only=tracking_only,
        )
        self.keyframes.append(new_kf)
        self.current_kf = new_kf
        return new_kf

    def _reinitialize_from_frame(self, frame, pose: np.ndarray) -> DSOKeyFrame:
        self.local_mapping.reset()
        self.covisibility = CovisibilityGraph()
        self.local_mapping.set_covisibility(self.covisibility)
        new_kf = self._refresh_tracking_keyframe(frame, pose, tracking_only=False)
        self.local_mapping.insert_keyframe(new_kf)
        self.local_mapping.run()
        self.local_mapping.update_covisibility(new_kf)
        self.consecutive_failures = 0
        self.reinitializations += 1
        return new_kf

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
                "tracking_cost": finite_float_or_none(report.get("tracking_cost")),
                "tracking_cost_per_point": finite_float_or_none(report.get("tracking_cost_per_point")),
                "valid_projected_points": int(report.get("valid_projected_points", 0) or 0),
                "valid_projected_ratio": finite_float_or_none(report.get("valid_projected_ratio")),
                "residual_mean": finite_float_or_none(report.get("residual_mean")),
                "residual_median": finite_float_or_none(report.get("residual_median")),
                "residual_p95": finite_float_or_none(report.get("residual_p95")),
                "inlier_ratio": finite_float_or_none(report.get("inlier_ratio")),
                "relative_translation_m": finite_float_or_none(report.get("relative_translation_m")),
                "relative_rotation_deg": finite_float_or_none(report.get("relative_rotation_deg")),
                "affine_gain": finite_float_or_none(report.get("affine_gain")),
                "affine_bias": finite_float_or_none(report.get("affine_bias")),
                "brightness_mode": report.get("brightness_mode"),
                "depth_filter": report.get("depth_filter"),
                "lk_inliers": int(report.get("lk_inliers", 0) or 0),
                "lk_photometric_translation_delta_m": finite_float_or_none(report.get("lk_photometric_translation_delta_m")),
                "lk_photometric_rotation_delta_deg": finite_float_or_none(report.get("lk_photometric_rotation_delta_deg")),
                "tracking_failure": bool(tracking_failure),
                "fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
                "failure_reason": report.get("failure_reason"),
                "keyframe_inserted": bool(keyframe_inserted),
                "consecutive_failures": self.consecutive_failures,
                "reinitializations": self.reinitializations,
                "active_window_keyframes": len(self.local_mapping.active_window.keyframes),
                "active_inverse_depth_points": self._active_inverse_depth_count(),
                "active_points_culled": self.local_mapping.active_points_culled,
                "residual_p95_gate_rejections": self.residual_p95_gate_rejections,
                "cost_jump_rejections": self.cost_jump_rejections,
                "lk_consistency_rejections": self.lk_consistency_rejections,
                "loop_candidates": self.loop_candidates,
                "loop_verified": self.loop_verified,
                "loop_corrections_applied": self.loop_corrections_applied,
                "ba_runs": self.local_mapping.ba_runs,
                "ba_accepted": self.local_mapping.ba_accepted,
                "ba_rejected": self.local_mapping.ba_rejected,
            }
        )

    def run(self) -> tuple[list[np.ndarray], list[float]]:
        trajectory = [np.eye(4, dtype=np.float64)]
        runtimes = []

        frames = iter(self.loader.iter_frames())
        try:
            first_frame = next(frames)
        except StopIteration:
            return trajectory, runtimes

        current_pose = np.eye(4, dtype=np.float64)
        previous_pose = current_pose.copy()
        self.last_safe_pose = current_pose.copy()
        self.last_velocity = np.eye(4, dtype=np.float64)
        self.diagnostics = []
        self.consecutive_failures = 0

        # Create first keyframe
        first_kf = self.tracker.make_keyframe(first_frame.left, first_frame.right, current_pose, first_frame.timestamp)
        self.keyframes.append(first_kf)
        self.current_kf = first_kf
        self.local_mapping.insert_keyframe(first_kf)
        self.local_mapping.run()
        self.local_mapping.update_covisibility(first_kf)
        self.loop_detector.add_keyframe(first_kf)
        self.frames_processed = 1
        self._record_diagnostic(
            first_frame,
            self.tracker._empty_report(None),
            tracking_failure=False,
            fallback_used=False,
            fallback_reason=None,
            keyframe_inserted=True,
        )

        for frame in frames:
            start = time.perf_counter()

            # === TRACKING (DIRECT PHOTOMETRIC ALIGNMENT) ===
            relative_pose = self.tracker.estimate_pose(self.current_kf, frame.left)
            report = dict(self.tracker.last_tracking_report)
            tracking_failure = not self.tracker.last_tracking_success
            fallback_used = False
            fallback_reason = None

            if tracking_failure:
                self.tracking_failures += 1
                self.consecutive_failures += 1
                self._count_rejection(report.get("failure_reason"))
                current_pose, fallback_reason = self._fallback_pose(previous_pose)
                fallback_used = True
                self.fallbacks_used += 1
            else:
                self.consecutive_failures = 0
                current_pose = self.current_kf.pose @ np.linalg.inv(relative_pose)

            # === KEYFRAME MANAGEMENT ===
            keyframe_inserted = False
            if tracking_failure:
                if self.consecutive_failures > self.tracking_config.max_consecutive_failures:
                    self._reinitialize_from_frame(frame, current_pose)
                    keyframe_inserted = True
                elif self.tracking_config.force_keyframe_after_fallback:
                    self._refresh_tracking_keyframe(frame, current_pose, tracking_only=True)
                    keyframe_inserted = True
            elif self.tracker.should_update_keyframe(self.current_kf, frame.left):
                new_kf = self.tracker.make_keyframe(frame.left, frame.right, current_pose, frame.timestamp)
                self.keyframes.append(new_kf)
                self.current_kf = new_kf
                keyframe_inserted = True

                # Local mapping
                self.local_mapping.insert_keyframe(new_kf)
                self.local_mapping.run()
                current_pose = new_kf.pose.copy()
                self.local_mapping.update_covisibility(new_kf)

                # Loop detection
                loop_candidate = self.loop_detector.add_keyframe(new_kf)
                if loop_candidate is not None:
                    self.loop_candidates += 1
                    self._correct_loop(loop_candidate, new_kf.id)

            trajectory.append(current_pose.copy())
            if not tracking_failure or fallback_reason in {"lk_pnp", "constant_velocity"}:
                try:
                    self.last_velocity = np.linalg.inv(previous_pose) @ current_pose
                except np.linalg.LinAlgError:
                    self.last_velocity = np.eye(4, dtype=np.float64)
                self.last_safe_pose = current_pose.copy()
            previous_pose = current_pose.copy()

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

    def _strict_loop_candidate_ok(self, candidate: DSOLoopCandidate, current_kf_id: int) -> bool:
        if not self.tracking_config.enable_strict_loop_verification:
            return True
        if candidate.inliers < 45 or candidate.inlier_ratio < 0.45:
            return False
        keyframe_ids = [kf.id for kf in self.keyframes if not kf.tracking_only]
        try:
            current_index = keyframe_ids.index(current_kf_id)
            loop_index = keyframe_ids.index(candidate.keyframe_id)
        except ValueError:
            return False
        if abs(current_index - loop_index) < 30:
            return False
        kf_poses = {kf.id: kf.pose for kf in self.keyframes if not kf.tracking_only}
        if candidate.keyframe_id not in kf_poses or current_kf_id not in kf_poses:
            return False
        relative = np.linalg.inv(kf_poses[candidate.keyframe_id]) @ kf_poses[current_kf_id]
        translation_m, rotation_deg = relative_motion_stats(relative)
        return translation_m < 10.0 and rotation_deg < 45.0

    def _correct_loop(self, candidate: DSOLoopCandidate, current_kf_id: int) -> None:
        """Correct a verified loop only when the sparse pose graph is meaningful."""
        if not self._strict_loop_candidate_ok(candidate, current_kf_id):
            return
        self.loop_verified += 1
        if not self.tracking_config.enable_loop_correction:
            return

        kf_poses = {kf.id: kf.pose for kf in self.keyframes if not kf.tracking_only}
        if candidate.keyframe_id not in kf_poses or current_kf_id not in kf_poses:
            return

        edges: list[tuple[int, int, np.ndarray]] = []
        graph_kfs = [kf for kf in self.keyframes if not kf.tracking_only and kf.id in kf_poses]
        for prev_kf, next_kf in zip(graph_kfs[:-1], graph_kfs[1:]):
            edges.append((prev_kf.id, next_kf.id, np.linalg.inv(prev_kf.pose) @ next_kf.pose))

        T_loop_cur = np.linalg.inv(kf_poses[candidate.keyframe_id]) @ kf_poses[current_kf_id]
        edges.append((candidate.keyframe_id, current_kf_id, T_loop_cur))
        if len(edges) * 2 + 6 < len(kf_poses) * 6:
            return

        optimized = solve_pose_graph(edges, kf_poses, iterations=100)
        total_correction = 0.0
        for kf in self.keyframes:
            if kf.id in optimized:
                total_correction += float(np.linalg.norm(optimized[kf.id][:3, 3] - kf.pose[:3, 3]))
        if total_correction <= 1e-6:
            return

        self.loop_detector.loop_edges.append((candidate.keyframe_id, current_kf_id, T_loop_cur))
        for kf in self.keyframes:
            if kf.id in optimized:
                kf.pose = optimized[kf.id]
        if current_kf_id in optimized and self.current_kf is not None:
            self.current_kf.pose = optimized[current_kf_id]
        self.loop_corrections_applied += 1
        self.loop_closures = self.loop_corrections_applied

    def get_stats(self) -> dict[str, object]:
        active_points = self._active_inverse_depth_count()
        robustness = self.get_robustness_summary()
        return {
            "frames_processed": self.frames_processed,
            "keyframes": len(self.keyframes),
            "tracking_failures": self.tracking_failures,
            "fallbacks_used": self.fallbacks_used,
            "motion_gate_rejections": self.motion_gate_rejections,
            "low_projection_rejections": self.low_projection_rejections,
            "residual_p95_gate_rejections": self.residual_p95_gate_rejections,
            "cost_jump_rejections": self.cost_jump_rejections,
            "lk_consistency_rejections": self.lk_consistency_rejections,
            "consecutive_failures": self.consecutive_failures,
            "reinitializations": self.reinitializations,
            "relocalization_attempts": 0,
            "relocalization_successes": 0,
            "loop_candidates": self.loop_candidates,
            "loop_verified": self.loop_verified,
            "loop_corrections_applied": self.loop_corrections_applied,
            "loop_closures": self.loop_closures,
            "active_window_keyframes": len(self.local_mapping.active_window.keyframes),
            "active_inverse_depth_points": active_points,
            "active_points_culled": self.local_mapping.active_points_culled,
            "mean_valid_projected_points": robustness["mean_valid_projected_points"],
            "median_valid_projected_points": robustness["median_valid_projected_points"],
            "mean_inlier_ratio": robustness["mean_inlier_ratio"],
            "marginalizations": self.local_mapping.marginalizations,
            "photometric_ba_runs": self.local_mapping.photometric_ba_runs,
            "ba_runs": self.local_mapping.ba_runs,
            "ba_accepted": self.local_mapping.ba_accepted,
            "ba_rejected": self.local_mapping.ba_rejected,
            "last_tracking_cost": finite_float_or_none(self.tracker.last_tracking_cost),
            "stereo_depth_filter": (
                "left_right_consistency"
                if self.tracking_config.enable_left_right_depth_check
                else "positive_disparity_depth_range"
            ),
        }

    def get_diagnostics(self) -> list[dict[str, object]]:
        return [dict(item) for item in self.diagnostics]

    def get_robustness_summary(self) -> dict[str, float | int]:
        valid_counts = np.asarray(
            [item["valid_projected_points"] for item in self.diagnostics if item["valid_projected_points"] > 0],
            dtype=np.float64,
        )
        inlier_values = [
            finite_float_or_none(item.get("inlier_ratio"))
            for item in self.diagnostics
        ]
        inlier_ratios = np.asarray([value for value in inlier_values if value is not None], dtype=np.float64)
        return {
            "frames_processed": self.frames_processed,
            "tracking_failures": self.tracking_failures,
            "fallbacks_used": self.fallbacks_used,
            "motion_gate_rejections": self.motion_gate_rejections,
            "low_projection_rejections": self.low_projection_rejections,
            "residual_p95_gate_rejections": self.residual_p95_gate_rejections,
            "cost_jump_rejections": self.cost_jump_rejections,
            "lk_consistency_rejections": self.lk_consistency_rejections,
            "reinitializations": self.reinitializations,
            "mean_valid_projected_points": float(np.mean(valid_counts)) if len(valid_counts) else 0.0,
            "median_valid_projected_points": float(np.median(valid_counts)) if len(valid_counts) else 0.0,
            "mean_inlier_ratio": float(np.mean(inlier_ratios)) if len(inlier_ratios) else 0.0,
            "active_inverse_depth_points": self._active_inverse_depth_count(),
            "loop_candidates": self.loop_candidates,
            "loop_verified": self.loop_verified,
            "loop_corrections_applied": self.loop_corrections_applied,
            "loop_closures": self.loop_closures,
        }


def run_dso_slam(
    data_dir: str,
    seq: str,
    output_path: str | None = None,
    num_active_points: int = 1500,
) -> tuple[list[np.ndarray], list[float]]:
    loader = KITTIOdometryLoader(data_dir, seq)
    slam = DSOSLAM(loader, num_active_points=num_active_points)
    trajectory, runtimes = slam.run()

    out_path = output_path or f"results/dso_slam_seq{int(seq):02d}.txt"
    save_trajectory_kitti(trajectory, out_path)
    return trajectory, runtimes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DSO-SLAM on KITTI")
    parser.add_argument("--data-dir", default="data/kitti_odometry")
    parser.add_argument("--seq", default="00")
    parser.add_argument("--output", default=None)
    parser.add_argument("--points", type=int, default=1500)
    args = parser.parse_args()

    trajectory, runtimes = run_dso_slam(args.data_dir, args.seq, args.output, args.points)
    print(f"DSO-SLAM: {len(trajectory)} frames, avg {np.mean(runtimes):.2f} ms/frame")
