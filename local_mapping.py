"""
LocalMapping: local mapping thread for both ORB-SLAM2 and DSO-SLAM.
Handles map point creation, local BA, and map point culling.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from slam_base import KeyFrame, MapPoint, CovisibilityGraph, solve_local_ba


@dataclass
class LocalMappingConfig:
    max_keyframes: int = 8
    min_tracked_points: int = 50
    point_culling_age: int = 2
    min_observations: int = 2
    max_reproj_error: float = 4.0
    min_distance: float = 0.5
    max_distance: float = 10.0
    max_ba_points: int = 80
    max_ba_iterations: int = 10


class LocalMapping:
    """Local mapping thread: processes new keyframes, creates map points, runs BA."""

    def __init__(self, config: Optional[LocalMappingConfig] = None):
        self.config = config or LocalMappingConfig()
        self.new_keyframes: list[KeyFrame] = []
        self.covisibility: Optional[CovisibilityGraph] = None
        self.processed_kf_count = 0
        self.fx, self.fy = 718.856, 718.856
        self.cx, self.cy = 607.1928, 185.2157
        self.baseline = 0.54

    def set_covisibility(self, cov: CovisibilityGraph) -> None:
        self.covisibility = cov

    def insert_keyframe(self, kf: KeyFrame) -> None:
        """Add a new keyframe to the processing queue."""
        self.new_keyframes.append(kf)

    def run(self) -> list[KeyFrame]:
        """
        Main local mapping loop. Call this periodically (e.g. per frame).
        Returns list of locally optimized keyframes.
        """
        if not self.new_keyframes:
            return []

        processed = []
        while self.new_keyframes:
            kf = self.new_keyframes.pop(0)
            self._process_keyframe(kf)
            processed.append(kf)
            self.processed_kf_count += 1

        return processed

    def _process_keyframe(self, kf: KeyFrame) -> None:
        """Process a single keyframe: create map points, update connections, run BA."""
        self._associate_existing_map_points(kf)

        # Create new map points from stereo triangulation
        new_points = self._create_stereo_map_points(kf)

        # Update covisibility graph connections
        self._update_connections(kf)

        # Run local BA on this keyframe and its neighbors
        if self.covisibility and len(self.new_keyframes) == 0:
            # Only run BA when queue is nearly empty (reduce latency)
            self._run_local_ba(kf)

        # Cull old map points
        self._cull_map_points(kf)

    def _create_stereo_map_points(self, kf: KeyFrame) -> list[MapPoint]:
        """Create MapPoints by triangulating stereo ORB matches."""
        new_points = []

        if kf.points_3d is not None and kf.valid_mask is not None and kf.keypoints:
            for feature_idx, is_valid in enumerate(kf.valid_mask):
                if not is_valid or feature_idx in kf.feature_map_points:
                    continue
                if len(kf.map_points) >= 2000:
                    break

                pt_cam = np.asarray(kf.points_3d[feature_idx], dtype=np.float64)
                if np.linalg.norm(pt_cam) < 1e-6:
                    continue
                pt_world = kf.pose[:3, :3] @ pt_cam + kf.pose[:3, 3]
                u, v = kf.keypoints[feature_idx].pt
                mp = MapPoint(
                    id=MapPoint.next_id(),
                    position=pt_world,
                    observations=[(kf.id, u, v)],
                    descriptor=kf.descriptors[feature_idx].copy() if kf.descriptors is not None else None,
                    found_count=1,
                    matched_count=1,
                )
                new_points.append(mp)
                kf.map_points.append(mp)
                kf.feature_map_points[feature_idx] = mp
            return new_points

        if kf.left_image is None or kf.right_image is None:
            return new_points

        # Use ORB to find stereo matches
        orb = cv2.ORB_create(nfeatures=500)
        kp1, desc1 = orb.detectAndCompute(kf.left_image, None)
        kp2, desc2 = orb.detectAndCompute(kf.right_image, None)

        if desc1 is None or desc2 is None:
            return new_points

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(desc1, desc2, k=2)

        for candidates in matches:
            if len(candidates) < 2:
                continue
            m, second = candidates
            if m.distance >= 0.75 * second.distance:
                continue
            if len(kf.map_points) >= 2000:
                break
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt

            disparity = u1 - u2
            if disparity < 1.0 or abs(v1 - v2) > 2.0:
                continue

            # Get calibration (should be passed to keyframe or stored globally)
            fx, fy, cx, cy, baseline = self._get_calibration(kf)

            depth = fx * baseline / disparity
            if depth < self.config.min_distance or depth > self.config.max_distance:
                continue

            x = (u1 - cx) * depth / fx
            y = (v1 - cy) * depth / fy
            z = depth

            pt_cam = np.array([x, y, z], dtype=np.float64)
            pt_world = kf.pose[:3, :3] @ pt_cam + kf.pose[:3, 3]
            mp = MapPoint(
                id=MapPoint.next_id(),
                position=pt_world,
                observations=[(kf.id, u1, v1)],
                descriptor=desc1[m.queryIdx].copy(),
                found_count=1,
                matched_count=1,
            )
            new_points.append(mp)
            kf.map_points.append(mp)
            kf.feature_map_points[m.queryIdx] = mp

        return new_points

    def _get_calibration(self, kf: KeyFrame):
        return self.fx, self.fy, self.cx, self.cy, self.baseline

    def _associate_existing_map_points(self, kf: KeyFrame) -> None:
        """Associate current keyframe features with existing map points."""
        if self.covisibility is None or kf.descriptors is None or not kf.keypoints:
            return

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        recent_keyframes = list(self.covisibility.kfs.values())[-5:]
        for prev_kf in recent_keyframes:
            if prev_kf.id == kf.id or prev_kf.descriptors is None or not prev_kf.feature_map_points:
                continue
            matches = matcher.knnMatch(prev_kf.descriptors, kf.descriptors, k=2)
            for candidates in matches:
                if len(candidates) < 2:
                    continue
                best, second = candidates
                if best.distance >= 0.75 * second.distance:
                    continue
                mp = prev_kf.feature_map_points.get(best.queryIdx)
                if mp is None or not mp.valid or best.trainIdx in kf.feature_map_points:
                    continue
                u, v = kf.keypoints[best.trainIdx].pt
                if not any(obs[0] == kf.id for obs in mp.observations):
                    mp.observations.append((kf.id, u, v))
                    mp.found_count += 1
                    mp.matched_count += 1
                    if mp.descriptor is None:
                        mp.descriptor = prev_kf.descriptors[best.queryIdx].copy()
                kf.feature_map_points[best.trainIdx] = mp
                if all(existing.id != mp.id for existing in kf.map_points):
                    kf.map_points.append(mp)

    def _update_connections(self, kf: KeyFrame) -> None:
        """Update covisibility graph edges for the new keyframe."""
        if self.covisibility is None:
            return

        for existing_kf in self.covisibility.kfs.values():
            shared = self._count_shared_map_points(kf, existing_kf)
            if shared > 0:
                self.covisibility.update_edge(kf.id, existing_kf.id, shared)

    def _count_shared_map_points(self, kf1: KeyFrame, kf2: KeyFrame) -> int:
        """Count map points observed by both keyframes."""
        ids1 = {mp.id for mp in kf1.map_points if mp.valid}
        ids2 = {mp.id for mp in kf2.map_points if mp.valid}
        return len(ids1 & ids2)

    def _run_local_ba(self, kf: KeyFrame) -> None:
        """Run local BA on keyframe and its neighbors."""
        if self.covisibility is None:
            return

        # Get local window (2-level neighbors)
        local_ids = self.covisibility.get_local_window(kf.id, radius=2)
        local_kfs = [self.covisibility.kfs[i] for i in local_ids if i in self.covisibility.kfs]

        if len(local_kfs) < 2:
            return

        # Collect all map points from local keyframes
        all_mps = {}
        for lkf in local_kfs:
            for mp in lkf.map_points:
                all_mps[mp.id] = mp

        mp_list = [
            mp for mp in all_mps.values()
            if mp.valid and len(mp.observations) >= self.config.min_observations
        ]
        if len(mp_list) < 10:
            return
        mp_list = sorted(mp_list, key=lambda mp: len(mp.observations), reverse=True)
        if len(mp_list) > self.config.max_ba_points:
            mp_list = mp_list[:self.config.max_ba_points]

        K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)
        solve_local_ba(local_kfs, mp_list, max_iter=self.config.max_ba_iterations, camera_matrix=K)

    def _cull_map_points(self, kf: KeyFrame) -> None:
        """Remove bad map points (too old, too few observations, outlier)."""
        surviving = []
        for mp in kf.map_points:
            if not mp.valid:
                continue
            first_obs_id = mp.observations[0][0] if mp.observations else kf.id
            age = kf.id - first_obs_id
            if len(mp.observations) < self.config.min_observations and age > self.config.point_culling_age:
                mp.valid = False
                continue
            if mp.found_count > 0 and mp.matched_count / mp.found_count < 0.25:
                mp.valid = False
                continue
            surviving.append(mp)

        # Limit total map points
        if len(surviving) > 2000:
            surviving = surviving[:2000]
        kf.map_points = surviving

    def set_calibration(self, fx: float, fy: float, cx: float, cy: float, baseline: float) -> None:
        """Set camera calibration parameters for map point triangulation."""
        self.fx, self.fy, self.cx, self.cy, self.baseline = fx, fy, cx, cy, baseline
