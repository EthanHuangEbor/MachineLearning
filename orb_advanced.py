from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from loop_detector import solve_sim3
from slam_base import KeyFrame, MapPoint, CovisibilityGraph, solve_local_ba


def _as_descriptor_matrix(map_points: list[MapPoint]) -> tuple[np.ndarray, list[MapPoint]]:
    usable = [mp for mp in map_points if mp.valid and mp.descriptor is not None]
    if not usable:
        return np.empty((0, 32), dtype=np.uint8), []
    return np.asarray([mp.descriptor for mp in usable], dtype=np.uint8), usable


def _pose_from_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    min_inliers: int,
) -> tuple[Optional[np.ndarray], int]:
    if len(object_points) < 6:
        return None, 0
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        camera_matrix,
        None,
        reprojectionError=4.0,
        iterationsCount=100,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return None, 0
    rotation, _ = cv2.Rodrigues(rvec)
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, :3] = rotation
    Tcw[:3, 3] = tvec.reshape(3)
    return np.linalg.inv(Tcw), int(len(inliers))


@dataclass
class TrackingResult:
    pose: np.ndarray
    inliers: int
    matched_points: list[MapPoint]


class LocalMapTracker:
    """ORB-SLAM2-style local map tracking using visible map-point projection matches."""

    def __init__(self, camera_matrix: np.ndarray, radius_px: float = 35.0, min_inliers: int = 20):
        self.camera_matrix = camera_matrix
        self.radius_px = radius_px
        self.min_inliers = min_inliers
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def collect_local_map(
        self,
        current_kf: Optional[KeyFrame],
        keyframes: list[KeyFrame],
        covisibility: CovisibilityGraph,
        max_points: int = 2000,
    ) -> list[MapPoint]:
        if not keyframes:
            return []
        candidate_kfs: list[KeyFrame] = []
        if current_kf is not None:
            local_ids = covisibility.get_local_window(current_kf.id, radius=2)
            candidate_kfs.extend([covisibility.kfs[kid] for kid in local_ids if kid in covisibility.kfs])
        candidate_kfs.extend(keyframes[-5:])

        unique: dict[int, MapPoint] = {}
        for kf in candidate_kfs:
            for mp in kf.map_points:
                if mp.valid and mp.descriptor is not None:
                    unique[mp.id] = mp
        points = sorted(unique.values(), key=lambda mp: len(mp.observations), reverse=True)
        return points[:max_points]

    def refine_pose(
        self,
        pose_guess: np.ndarray,
        keypoints: list,
        descriptors: Optional[np.ndarray],
        local_points: list[MapPoint],
    ) -> TrackingResult:
        if descriptors is None or len(keypoints) < 6:
            return TrackingResult(pose_guess, 0, [])

        map_desc, usable_points = _as_descriptor_matrix(local_points)
        if len(usable_points) < 6:
            return TrackingResult(pose_guess, 0, [])

        raw_matches = self.matcher.knnMatch(map_desc, descriptors, k=2)
        object_points = []
        image_points = []
        matched_points = []
        Tcw_guess = np.linalg.inv(pose_guess)

        for candidates in raw_matches:
            if len(candidates) < 2:
                continue
            best, second = candidates
            if best.distance >= 0.75 * second.distance:
                continue

            mp = usable_points[best.queryIdx]
            pt_cam = Tcw_guess[:3, :3] @ mp.position + Tcw_guess[:3, 3]
            if pt_cam[2] <= 0.1:
                continue
            projected = self.camera_matrix @ pt_cam
            projected = projected[:2] / projected[2]
            observed = np.asarray(keypoints[best.trainIdx].pt)
            if np.linalg.norm(projected - observed) > self.radius_px:
                continue
            object_points.append(mp.position)
            image_points.append(observed)
            matched_points.append(mp)

        pose, inliers = _pose_from_pnp(
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
            self.camera_matrix,
            self.min_inliers,
        )
        if pose is None:
            return TrackingResult(pose_guess, 0, [])
        for mp in matched_points:
            mp.matched_count += 1
        return TrackingResult(pose, inliers, matched_points)


class Relocalizer:
    """BoW candidate retrieval plus 3D-2D PnP relocalization."""

    def __init__(self, camera_matrix: np.ndarray, min_inliers: int = 20):
        self.camera_matrix = camera_matrix
        self.min_inliers = min_inliers
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def relocalize(
        self,
        bow_vector: Optional[np.ndarray],
        keypoints: list,
        descriptors: Optional[np.ndarray],
        keyframes: list[KeyFrame],
        vocabulary,
        top_k: int = 5,
    ) -> Optional[TrackingResult]:
        if bow_vector is None or descriptors is None or not keyframes:
            return None
        candidates = []
        for kf in keyframes:
            if kf.bow_vector is None or not kf.feature_map_points:
                continue
            sim = vocabulary.get_similarity(bow_vector, kf.bow_vector)
            candidates.append((sim, kf))
        candidates.sort(key=lambda item: item[0], reverse=True)

        best: Optional[TrackingResult] = None
        for _, candidate in candidates[:top_k]:
            object_points = []
            image_points = []
            matched_points = []
            if candidate.descriptors is None:
                continue
            raw_matches = self.matcher.knnMatch(candidate.descriptors, descriptors, k=2)
            for pair in raw_matches:
                if len(pair) < 2:
                    continue
                match, second = pair
                if match.distance >= 0.75 * second.distance:
                    continue
                mp = candidate.feature_map_points.get(match.queryIdx)
                if mp is None or not mp.valid:
                    continue
                object_points.append(mp.position)
                image_points.append(keypoints[match.trainIdx].pt)
                matched_points.append(mp)

            pose, inliers = _pose_from_pnp(
                np.asarray(object_points, dtype=np.float32),
                np.asarray(image_points, dtype=np.float32),
                self.camera_matrix,
                self.min_inliers,
            )
            if pose is not None and (best is None or inliers > best.inliers):
                best = TrackingResult(pose, inliers, matched_points)
        return best


class Sim3LoopFusion:
    """Compute a Sim3 correction from matched map points and fuse duplicate points."""

    def __init__(self, min_pairs: int = 12):
        self.min_pairs = min_pairs
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _matched_map_point_pairs(self, current: KeyFrame, loop: KeyFrame) -> list[tuple[MapPoint, MapPoint]]:
        if current.descriptors is None or loop.descriptors is None:
            return []
        pairs = []
        raw_matches = self.matcher.knnMatch(current.descriptors, loop.descriptors, k=2)
        for candidates in raw_matches:
            if len(candidates) < 2:
                continue
            best, second = candidates
            if best.distance >= 0.75 * second.distance:
                continue
            mp_current = current.feature_map_points.get(best.queryIdx)
            mp_loop = loop.feature_map_points.get(best.trainIdx)
            if mp_current is None or mp_loop is None or not mp_current.valid or not mp_loop.valid:
                continue
            pairs.append((mp_current, mp_loop))
        return pairs

    def fuse(self, current: KeyFrame, loop: KeyFrame, keyframes: list[KeyFrame]) -> Optional[np.ndarray]:
        pairs = self._matched_map_point_pairs(current, loop)
        if len(pairs) < self.min_pairs:
            return None

        pts_current = np.asarray([a.position for a, _ in pairs], dtype=np.float64)
        pts_loop = np.asarray([b.position for _, b in pairs], dtype=np.float64)
        sim3, _ = solve_sim3(pts_current, pts_loop)
        scale = float(np.cbrt(max(np.linalg.det(sim3[:3, :3]), 1e-12)))
        rotation = sim3[:3, :3] / max(scale, 1e-12)
        translation = sim3[:3, 3]

        affected_ids = {kf.id for kf in keyframes if kf.id >= current.id}
        transformed_points: set[int] = set()
        for kf in keyframes:
            if kf.id not in affected_ids:
                continue
            kf.pose[:3, :3] = rotation @ kf.pose[:3, :3]
            kf.pose[:3, 3] = scale * (rotation @ kf.pose[:3, 3]) + translation
            for mp in kf.map_points:
                if mp.id in transformed_points:
                    continue
                mp.position = scale * (rotation @ mp.position) + translation
                transformed_points.add(mp.id)

        for mp_current, mp_loop in pairs:
            if mp_current.id == mp_loop.id:
                continue
            mp_current.valid = False
            for obs in mp_current.observations:
                if obs not in mp_loop.observations:
                    mp_loop.observations.append(obs)
        return sim3


class GlobalBundleAdjuster:
    """Bounded global BA used after loop correction."""

    def __init__(self, camera_matrix: np.ndarray, max_points: int = 250, max_iter: int = 15):
        self.camera_matrix = camera_matrix
        self.max_points = max_points
        self.max_iter = max_iter

    def optimize(self, keyframes: list[KeyFrame]) -> bool:
        unique: dict[int, MapPoint] = {}
        for kf in keyframes:
            for mp in kf.map_points:
                if mp.valid and len(mp.observations) >= 2:
                    unique[mp.id] = mp
        points = sorted(unique.values(), key=lambda mp: len(mp.observations), reverse=True)
        if len(points) < 20 or len(keyframes) < 2:
            return False
        return solve_local_ba(keyframes, points[:self.max_points], max_iter=self.max_iter, camera_matrix=self.camera_matrix)
