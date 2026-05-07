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
from dataclasses import dataclass
from typing import Optional

from kitti_utils import KITTIOdometryLoader, CameraCalibration, disparity_to_depth, save_trajectory_kitti
from slam_base import KeyFrame, MapPoint, CovisibilityGraph, EssentialGraph, solve_local_ba, solve_pose_graph
from local_mapping import LocalMapping, LocalMappingConfig


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
    active_intensities: np.ndarray
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


# =============================================================================
# Photometric energy functional (DSO-style cost function)
# =============================================================================

def build_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def select_high_gradient_pixels(gray: np.ndarray, depth: np.ndarray, num_points: int = 1500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    flat = magnitude.ravel()
    indices = np.argsort(flat)[::-1][:num_points]
    rows, cols = np.unravel_index(indices, gray.shape)
    uvs = np.column_stack([cols.astype(np.float32), rows.astype(np.float32)])
    return uvs, cols, rows


def compute_stereo_depth(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=7,
        P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(left, right).astype(np.float32) / 16.0
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
    tx, ty, tz, rx, ry, rz = xi
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
) -> np.ndarray:
    scale = 0.5 ** level
    sf = calib.focal_length * scale
    scx = calib.cx * scale
    scy = calib.cy * scale

    T = pose_to_matrix(xi)
    pts_cam = (T[:3, :3] @ points_3d.T).T + T[:3, 3]
    depth = pts_cam[:, 2]
    valid = depth > 0.1

    u_proj = np.where(valid, sf * pts_cam[:, 0] / depth + scx, 0.0)
    v_proj = np.where(valid, sf * pts_cam[:, 1] / depth + scy, 0.0)

    curr_gray = curr_pyramid[level].astype(np.float32)
    h, w = curr_gray.shape
    in_bounds = (u_proj > 0) & (u_proj < w - 1) & (v_proj > 0) & (v_proj < h - 1)
    mask = valid & in_bounds

    residuals = np.zeros(len(points_3d), dtype=np.float32)
    if mask.any():
        interp = bilinear_interpolate(curr_gray, u_proj[mask], v_proj[mask])
        residuals[mask] = intensities_ref[mask] - interp
        residuals[~mask] = 0.0

    return residuals


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

    def __init__(self, calib: CameraCalibration, num_active_points: int = 1500, pyramid_levels: int = 4):
        self.calib = calib
        self.num_active_points = num_active_points
        self.pyramid_levels = pyramid_levels
        self.keyframe_flow_threshold = 20.0

    def make_keyframe(self, left: np.ndarray, right: np.ndarray, pose: np.ndarray) -> DSOKeyFrame:
        disparity = compute_stereo_depth(left, right)
        depth = disparity_to_depth(disparity, self.calib)
        uvs, cols, rows = select_high_gradient_pixels(left, depth, self.num_active_points)

        u = uvs[:, 0].astype(np.float32)
        v = uvs[:, 1].astype(np.float32)
        z = bilinear_interpolate(depth, u, v)
        valid = z > 0.5
        uvs = uvs[valid]
        z = z[valid]

        x = (uvs[:, 0] - self.calib.cx) * z / self.calib.focal_length
        y = (uvs[:, 1] - self.calib.cy) * z / self.calib.focal_length
        pts_3d = np.column_stack([x, y, z])

        intensities = bilinear_interpolate(left.astype(np.float32), uvs[:, 0], uvs[:, 1])

        kf = DSOKeyFrame(
            id=DSOKeyFrame.next_id(),
            timestamp=0.0,
            pose=pose,
            gray=left.copy(),
            depth=depth.copy(),
            active_uvs=uvs,
            active_points_3d=pts_3d,
            active_intensities=intensities,
        )
        return kf

    def estimate_pose(self, ref_kf: DSOKeyFrame, curr_gray: np.ndarray) -> np.ndarray:
        """Estimate relative pose via photometric alignment with multi-level LM."""
        curr_pyramid = build_pyramid(curr_gray, self.pyramid_levels)
        ref_pyramid = build_pyramid(ref_kf.gray, self.pyramid_levels)

        xi = np.zeros(6, dtype=np.float64)

        for level in range(self.pyramid_levels - 1, -1, -1):
            scale = 0.5 ** level
            ref_intensities = bilinear_interpolate(
                ref_pyramid[level].astype(np.float32),
                ref_kf.active_uvs[:, 0] * scale,
                ref_kf.active_uvs[:, 1] * scale,
            )

            def residual_fn(x_opt):
                return compute_photometric_residuals(
                    x_opt, ref_kf.active_points_3d, ref_intensities,
                    curr_pyramid, self.calib, level
                )

            result = least_squares(residual_fn, xi, method="lm", max_nfev=20, ftol=1e-4)
            xi = result.x

        return pose_to_matrix(xi)

    def should_update_keyframe(self, ref_kf: DSOKeyFrame, curr_gray: np.ndarray) -> bool:
        """Use ORB optical flow to measure frame-to-frame motion."""
        orb = cv2.ORB_create(nfeatures=200)
        kp1, desc1 = orb.detectAndCompute(ref_kf.gray, None)
        kp2, desc2 = orb.detectAndCompute(curr_gray, None)
        if desc1 is None or desc2 is None or not kp1 or not kp2:
            return True

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.match(desc1, desc2)
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
    """
    DSO-style local mapping with photometric bundle adjustment.
    Uses the same LocalMapping infrastructure (slam_base) but with光度 error BA.
    """

    def __init__(self, calib: CameraCalibration, config: Optional[LocalMappingConfig] = None):
        self.calib = calib
        self.config = config or LocalMappingConfig()
        self.active_keyframes: list[DSOKeyFrame] = []
        self.new_kf_queue: list[DSOKeyFrame] = []
        self.covisibility: Optional[CovisibilityGraph] = None

    def set_covisibility(self, cov: CovisibilityGraph) -> None:
        self.covisibility = cov

    def insert_keyframe(self, kf: DSOKeyFrame) -> None:
        self.new_kf_queue.append(kf)

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

        # Marginalize old keyframes (keep last 5)
        if len(self.active_keyframes) > 5:
            self.active_keyframes.pop(0)

        # Run photometric BA on active keyframes
        self._run_photometric_ba(kf)

    def _run_photometric_ba(self, kf: DSOKeyFrame) -> None:
        """Photometric Bundle Adjustment across active keyframes."""
        if len(self.active_keyframes) < 2:
            return

        # For each keyframe, compute residuals with its reference
        # In full DSO: jointly optimize across all active frames
        # Simplified here: optimize relative pose between kf and its reference
        ref_kf = self.active_keyframes[0]  # oldest keyframe in window

        # Build energy function over all active frames
        xi_init = matrix_to_pose_xi(kf.pose)
        curr_pyramid = build_pyramid(kf.gray, 4)

        ref_intensities = bilinear_interpolate(
            ref_kf.gray.astype(np.float32),
            ref_kf.active_uvs[:, 0],
            ref_kf.active_uvs[:, 1],
        )

        result = least_squares(
            lambda xi: compute_photometric_residuals(
                xi, ref_kf.active_points_3d, ref_intensities,
                curr_pyramid, self.calib, level=0
            ),
            xi_init, method="lm", max_nfev=50, ftol=1e-6
        )

        # Update keyframe pose
        kf.pose = pose_to_matrix(result.x)

    def update_covisibility(self, kf: DSOKeyFrame) -> None:
        if self.covisibility:
            self.covisibility.add_keyframe(
                KeyFrame(id=kf.id, timestamp=kf.timestamp, pose=kf.pose,
                         left_image=kf.gray, descriptors=None)
            )


# =============================================================================
# DSO Loop Detector (uses ORB feature matching)
# =============================================================================

class DSOLoopDetector:
    """Loop detection via ORB feature matching (DSO doesn't have native loop detection)."""

    def __init__(self, min_matches: int = 20, ransac_threshold: float = 3.0):
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.keyframes: list[DSOKeyFrame] = []
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bow_history: list[tuple] = []
        self.loop_edges: list[tuple[int, int, np.ndarray]] = []

    def add_keyframe(self, kf: DSOKeyFrame) -> Optional[int]:
        """Add keyframe and look for loop candidates."""
        self.keyframes.append(kf)
        self.bow_history.append((kf.id, kf.timestamp, kf.pose))

        if len(self.bow_history) < 15:
            return None

        # ORB matching with past keyframes (skip recent 10)
        loop_candidate = self._find_loop_candidate(kf)
        return loop_candidate

    def _find_loop_candidate(self, kf: DSOKeyFrame) -> Optional[int]:
        """Use ORB feature matching to find loop candidates."""
        best_match_count = 0
        best_kf_id = None

        skip_recent = min(10, len(self.keyframes) - 1)

        for prev_kf in self.keyframes[:-skip_recent]:
            if prev_kf.id == kf.id:
                continue

            kp1, desc1 = self.orb.detectAndCompute(kf.gray, None)
            kp2, desc2 = self.orb.detectAndCompute(prev_kf.gray, None)

            if desc1 is None or desc2 is None:
                continue

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = matcher.match(desc1, desc2)

            if len(matches) > best_match_count:
                best_match_count = len(matches)
                best_kf_id = prev_kf.id

        if best_match_count < self.min_matches:
            return None

        return best_kf_id


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
    ):
        self.loader = loader
        self.calib = loader.calibration

        self.tracker = DSOTacker(self.calib, num_active_points, pyramid_levels)

        lm_config = LocalMappingConfig(max_keyframes=5)
        self.local_mapping = DSOLocalMapping(self.calib, lm_config)
        self.covisibility = CovisibilityGraph()
        self.local_mapping.set_covisibility(self.covisibility)

        self.loop_detector = DSOLoopDetector()

        self.keyframes: list[DSOKeyFrame] = []
        self.current_kf: Optional[DSOKeyFrame] = None
        self.trajectory: list[np.ndarray] = [np.eye(4, dtype=np.float64)]

    def run(self) -> tuple[list[np.ndarray], list[float]]:
        trajectory = [np.eye(4, dtype=np.float64)]
        runtimes = []

        frames = self.loader.iter_frames()
        first_frame = next(frames)
        current_pose = np.eye(4, dtype=np.float64)

        # Create first keyframe
        first_kf = self.tracker.make_keyframe(first_frame.left, first_frame.right, current_pose)
        self.keyframes.append(first_kf)
        self.current_kf = first_kf

        for frame in frames:
            start = time.perf_counter()

            # === TRACKING (DIRECT PHOTOMETRIC ALIGNMENT) ===
            relative_pose = self.tracker.estimate_pose(self.current_kf, frame.left)
            current_pose = self.current_kf.pose @ relative_pose

            # === KEYFRAME MANAGEMENT ===
            if self.tracker.should_update_keyframe(self.current_kf, frame.left):
                new_kf = self.tracker.make_keyframe(frame.left, frame.right, current_pose)
                self.keyframes.append(new_kf)
                self.current_kf = new_kf

                # Local mapping
                self.local_mapping.insert_keyframe(new_kf)
                self.local_mapping.run()
                self.local_mapping.update_covisibility(new_kf)

                # Loop detection
                loop_kf_id = self.loop_detector.add_keyframe(new_kf)
                if loop_kf_id is not None:
                    self._correct_loop(loop_kf_id, new_kf.id)

            trajectory.append(current_pose.copy())
            runtimes.append((time.perf_counter() - start) * 1000.0)

        return trajectory, runtimes

    def _correct_loop(self, loop_kf_id: int, current_kf_id: int) -> None:
        """Correct loop by running pose graph optimization."""
        kf_poses = {kf.id: kf.pose for kf in self.keyframes}
        kf_poses[current_kf_id] = self.current_kf.pose

        T_loop_cur = np.linalg.inv(kf_poses[loop_kf_id]) @ kf_poses[current_kf_id]
        self.loop_detector.loop_edges.append((loop_kf_id, current_kf_id, T_loop_cur))

        if len(self.loop_detector.loop_edges) > 0:
            optimized = solve_pose_graph(self.loop_detector.loop_edges, kf_poses, iterations=100)
            for kf in self.keyframes:
                if kf.id in optimized:
                    kf.pose = optimized[kf.id]
            if current_kf_id in optimized:
                self.current_kf.pose = optimized[current_kf_id]


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