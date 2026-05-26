"""
SLAM base infrastructure: KeyFrame, MapPoint, CovisibilityGraph, EssentialGraph
Shared by both ORB-SLAM2 and DSO-SLAM systems.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# -----------------------------------------------------------------------------
# KeyFrame — a single camera frame used in SLAM
# -----------------------------------------------------------------------------

@dataclass
class KeyFrame:
    id: int
    timestamp: float
    pose: np.ndarray  # Twc, 4x4 SE(3), camera pose in world coordinates
    left_image: Optional[np.ndarray] = None
    right_image: Optional[np.ndarray] = None
    keypoints: list = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    points_3d: Optional[np.ndarray] = None  # stereo points in the keyframe camera coordinates
    valid_mask: Optional[np.ndarray] = None
    map_points: list = field(default_factory=list)  # observed MapPoints
    feature_map_points: dict = field(default_factory=dict)  # feature index -> MapPoint
    bow_vector: Optional[np.ndarray] = None  # bag-of-words vector
    observations: list = field(default_factory=list)  # (map_point, u, v)
    connections: dict = field(default_factory=dict)  # kf_id -> shared_map_point_count

    @staticmethod
    def next_id() -> int:
        KeyFrame._id_counter += 1
        return KeyFrame._id_counter

KeyFrame._id_counter = 0


# -----------------------------------------------------------------------------
# MapPoint — 3D point in the world
# -----------------------------------------------------------------------------

@dataclass
class MapPoint:
    id: int
    position: np.ndarray  # 3D world coordinates
    observations: list  # [(keyframe_id, u, v), ...]
    descriptor: Optional[np.ndarray] = None
    valid: bool = True
    found_count: int = 0
    matched_count: int = 0

    @staticmethod
    def next_id() -> int:
        MapPoint._id_counter += 1
        return MapPoint._id_counter

MapPoint._id_counter = 0


# -----------------------------------------------------------------------------
# CovisibilityGraph — edges weighted by shared map point count
# -----------------------------------------------------------------------------

class CovisibilityGraph:
    def __init__(self, min_edges: int = 15):
        self.kfs: dict[int, KeyFrame] = {}
        self.edges: dict[tuple[int, int], int] = {}  # (kf_a, kf_b) -> shared_count
        self.min_edges = min_edges

    def add_keyframe(self, kf: KeyFrame) -> None:
        self.kfs[kf.id] = kf

    def update_edge(self, kf_a: int, kf_b: int, shared: int) -> None:
        if shared > 0:
            key = (min(kf_a, kf_b), max(kf_a, kf_b))
            self.edges[key] = shared

    def get_neighbors(self, kf_id: int, min_shared: int = 15) -> list[int]:
        neighbors = []
        for kf in self.kfs.values():
            if kf.id == kf_id:
                continue
            key = (min(kf_id, kf.id), max(kf_id, kf.id))
            if self.edges.get(key, 0) >= min_shared:
                neighbors.append(kf.id)
        return neighbors

    def get_local_window(self, kf_id: int, radius: int = 10) -> list[int]:
        """Get kf_id plus its neighbors up to 2 levels deep."""
        visited = {kf_id}
        current = {kf_id}
        for _ in range(radius):
            next_level = []
            for cid in current:
                neighbors = self.get_neighbors(cid, min_shared=1)
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        next_level.append(n)
            current = next_level
        return list(visited)


# -----------------------------------------------------------------------------
# EssentialGraph — sparse subset of covisibility for global optimization
# -----------------------------------------------------------------------------

class EssentialGraph:
    def __init__(self):
        self.edges: dict[tuple[int, int], np.ndarray] = {}  # (kf_a, kf_b) -> T_ab

    def add_edge(self, kf_a: int, kf_b: int, T_ab: np.ndarray) -> None:
        key = (min(kf_a, kf_b), max(kf_a, kf_b))
        self.edges[key] = T_ab


# -----------------------------------------------------------------------------
# LocalBundleAdjustor — shared BA solver using scipy
# -----------------------------------------------------------------------------

def solve_local_ba(
    keyframes: list[KeyFrame],
    map_points: list[MapPoint],
    max_iter: int = 50,
    camera_matrix: Optional[np.ndarray] = None,
) -> bool:
    """
    Local Bundle Adjustment: optimize keyframe poses and map point positions
    to minimize reprojection error.

    Args:
        keyframes: list of KeyFrame objects
        map_points: list of MapPoint objects (subset observed by keyframes)
        max_iter: max LM iterations

    Returns:
        True if optimization converged
    """
    if not keyframes or not map_points:
        return False

    # Build mappings: kf_id -> index, mp_id -> index
    kf_ids = [kf.id for kf in keyframes]
    kf_idx = {kf_id: i for i, kf_id in enumerate(kf_ids)}
    mp_ids = [mp.id for mp in map_points]
    mp_idx = {mp_id: i for i, mp_id in enumerate(mp_ids)}

    K = np.asarray(camera_matrix, dtype=np.float64) if camera_matrix is not None else np.array(
        [[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]],
        dtype=np.float64,
    )

    n_kf = len(keyframes)
    n_mp = len(map_points)

    # Flatten initial parameters: [p0_xyz(p_mp), q0(p_kf), ...]
    # Map points: 3 params each
    # Keyframe poses: 6 params each (tx, ty, tz, rx, ry, rz -> se3)
    params = np.zeros(n_mp * 3 + n_kf * 6)
    for i, mp in enumerate(map_points):
        params[i * 3:(i + 1) * 3] = mp.position

    for i, kf in enumerate(keyframes):
        pose = kf.pose
        t = pose[:3, 3]
        R = pose[:3, :3]
        # se3 logarithmic map
        trace = np.clip(np.trace(R), -1.0, 3.0)
        ang = np.arccos((trace - 1.0) / 2.0)
        if ang < 1e-8:
            omega = np.zeros(3)
        else:
            log_R = ang / (2 * np.sin(ang)) * (R - R.T)
            omega = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])
        base = n_mp * 3 + i * 6
        params[base:base + 3] = t
        params[base + 3:base + 6] = omega

    observations = []  # (kf_idx, mp_idx, u, v)
    for kf_idx, kf in enumerate(keyframes):
        for mp in kf.map_points:
            if mp.id in mp_idx:
                for obs in mp.observations:
                    if obs[0] == kf.id:
                        u, v = obs[1], obs[2]
                        observations.append((kf_idx, mp_idx[mp.id], u, v))
                        break

    if not observations:
        return False

    first_pose_base = n_mp * 3
    first_pose_prior = params[first_pose_base:first_pose_base + 6].copy()

    def residuals(params_flat):
        errors = []
        for kf_i, mp_i, u, v in observations:
            # Map point
            mp_pos = params_flat[mp_i * 3:(mp_i + 1) * 3]
            # Keyframe pose
            base = n_mp * 3 + kf_i * 6
            t = params_flat[base:base + 3]
            omega = params_flat[base + 3:base + 6]
            angle = np.linalg.norm(omega)
            if angle < 1e-8:
                R_mat = np.eye(3)
            else:
                axis = omega / angle
                K_mat = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R_mat = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * K_mat @ K_mat
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = t
            Tcw = np.linalg.inv(T)
            cam_pos = Tcw[:3, :3] @ mp_pos + Tcw[:3, 3]
            if cam_pos[2] <= 1e-6:
                errors.append(1e3)
                errors.append(1e3)
                continue
            proj = K @ cam_pos
            proj /= proj[2]
            errors.append(proj[0] - u)
            errors.append(proj[1] - v)
        # Anchor the first keyframe to remove gauge freedom in the local window.
        errors.extend((params_flat[first_pose_base:first_pose_base + 6] - first_pose_prior) * 1e3)
        return np.array(errors)

    from scipy.optimize import least_squares
    result = least_squares(
        residuals,
        params,
        method="trf",
        loss="huber",
        f_scale=3.0,
        max_nfev=max_iter,
        ftol=1e-6,
        xtol=1e-6,
    )
    optimized = result.x

    # Write back optimized values
    for i, mp in enumerate(map_points):
        mp.position = optimized[i * 3:(i + 1) * 3]

    for i, kf in enumerate(keyframes):
        base = n_mp * 3 + i * 6
        t = optimized[base:base + 3]
        omega = optimized[base + 3:base + 6]
        angle = np.linalg.norm(omega)
        if angle < 1e-8:
            R_mat = np.eye(3)
        else:
            axis = omega / angle
            K_mat = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_mat = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * K_mat @ K_mat
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        kf.pose = T

    return result.success or result.nfev > 0


# -----------------------------------------------------------------------------
# PoseGraphOptimizer — for global pose graph optimization after loop closure
# -----------------------------------------------------------------------------

def solve_pose_graph(
    edges: list[tuple[int, int, np.ndarray]],
    initial_poses: dict[int, np.ndarray],
    iterations: int = 100,
) -> dict[int, np.ndarray]:
    """
    Pose graph optimization: minimize Σ ||T_i^{-1} T_j - T_ij||²

    edges: list of (kf_i, kf_j, T_relative_ij)
    initial_poses: kf_id -> 4x4 pose
    Returns: optimized poses
    """
    kf_ids = sorted(set(initial_poses.keys()) | set([e[0] for e in edges] + [e[1] for e in edges]))
    n = len(kf_ids)
    idx_map = {kid: i for i, kid in enumerate(kf_ids)}

    # Initialize with initial poses (flatten to se3)
    x0 = np.zeros(n * 6)
    for kid, pose in initial_poses.items():
        i = idx_map[kid]
        x0[i * 6:i * 6 + 3] = pose[:3, 3]
        R = pose[:3, :3]
        trace = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        ang = np.arccos(trace)
        if ang < 1e-8:
            omega = np.zeros(3)
        else:
            log_R = ang / (2 * np.sin(ang)) * (R - R.T)
            omega = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])
        x0[i * 6 + 3:i * 6 + 6] = omega

    def pose_to_mat(xi):
        tx, ty, tz, rx, ry, rz = xi
        angle = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        if angle < 1e-8:
            rotation = np.eye(3)
        else:
            axis = np.array([rx, ry, rz]) / angle
            K_mat = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * K_mat @ K_mat
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = [tx, ty, tz]
        return T

    def residuals(x):
        errs = []
        for i, j, T_ij in edges:
            xi = x[idx_map[i] * 6:idx_map[i] * 6 + 6]
            xj = x[idx_map[j] * 6:idx_map[j] * 6 + 6]
            Ti = pose_to_mat(xi)
            Tj = pose_to_mat(xj)
            T_pred = np.linalg.inv(Ti) @ Tj
            error_mat = np.linalg.inv(T_ij) @ T_pred
            tr = np.clip((np.trace(error_mat[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
            r_err = np.arccos(tr)
            t_err = np.linalg.norm(error_mat[:3, 3])
            errs.append(r_err * 100)
            errs.append(t_err * 10)
        # Gauge prior: keep the first pose fixed to its initial value.
        first = 0
        errs.extend((x[first:first + 6] - x0[first:first + 6]) * 1e3)
        return np.array(errs)

    from scipy.optimize import least_squares

    initial_residuals = residuals(x0)
    if initial_residuals.size < x0.size:
        # The simplified graph can be highly underconstrained when only one or
        # two loop edges exist. In that case a global optimization is not
        # meaningful, and dense trust-region iterations can dominate runtime.
        return {kid: pose_to_mat(x0[i * 6:i * 6 + 6]) for kid, i in idx_map.items()}

    result = least_squares(residuals, x0, method="lm", max_nfev=iterations)

    optimized = {}
    for kid, i in idx_map.items():
        xi = result.x[i * 6:i * 6 + 6]
        optimized[kid] = pose_to_mat(xi)

    return optimized
