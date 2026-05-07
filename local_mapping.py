"""
LocalMapping: local mapping thread for both ORB-SLAM2 and DSO-SLAM.
Handles map point creation, local BA, and map point culling.
"""

from __future__ import annotations

import time
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


class LocalMapping:
    """Local mapping thread: processes new keyframes, creates map points, runs BA."""

    def __init__(self, config: Optional[LocalMappingConfig] = None):
        self.config = config or LocalMappingConfig()
        self.new_keyframes: list[KeyFrame] = []
        self.covisibility: Optional[CovisibilityGraph] = None
        self.processed_kf_count = 0

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
        if kf.left_image is None or kf.right_image is None:
            return new_points

        # Use ORB to find stereo matches
        orb = cv2.ORB_create(nfeatures=500)
        kp1, desc1 = orb.detectAndCompute(kf.left_image, None)
        kp2, desc2 = orb.detectAndCompute(kf.right_image, None)

        if desc1 is None or desc2 is None:
            return new_points

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.match(desc1, desc2)

        for m in matches:
            if len(kf.map_points) >= 2000:
                break
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt

            disparity = abs(u1 - u2)
            if disparity < 1.0:
                continue

            # Get calibration (should be passed to keyframe or stored globally)
            fx, fy, cx, cy, baseline = self._get_calibration(kf)

            depth = fx * baseline / disparity
            if depth < self.config.min_distance or depth > self.config.max_distance:
                continue

            x = (u1 - cx) * depth / fx
            y = (v1 - cy) * depth / fy
            z = depth

            mp = MapPoint(
                id=MapPoint.next_id(),
                position=np.array([x, y, z], dtype=np.float64),
                observations=[(kf.id, u1, v1)],
                found_count=1,
                matched_count=1,
            )
            new_points.append(mp)
            kf.map_points.append(mp)

        return new_points

    def _get_calibration(self, kf: KeyFrame):
        """Placeholder calibration. In real impl, pass calib as constructor arg."""
        fx, fy, cx, cy, baseline = 718.856, 718.856, 607.1928, 185.2157, 0.54
        return fx, fy, cx, cy, baseline

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
        ids1 = {id(mp) for mp in kf1.map_points}
        ids2 = {id(mp) for mp in kf2.map_points}
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

        mp_list = list(all_mps.values())
        if len(mp_list) < 10:
            return

        # Run BA
        solve_local_ba(local_kfs, mp_list, max_iter=30)

    def _cull_map_points(self, kf: KeyFrame) -> None:
        """Remove bad map points (too old, too few observations, outlier)."""
        surviving = []
        for mp in kf.map_points:
            if not mp.valid:
                continue
            if mp.found_count < self.config.min_observations:
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